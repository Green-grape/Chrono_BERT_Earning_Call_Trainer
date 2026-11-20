from __future__ import annotations

import contextlib
import gc
from dataclasses import replace
import importlib.metadata
import json
import math
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from .common import (
    digest,
    file_hash,
    job_lock,
    model_id,
    read_json,
    validate_year,
    write_json,
)
from .data import BlockStore, build_cache
from .export import export_checkpoint


def base_snapshot(cfg, year: int) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model_id(year),
            revision=cfg.base_revision,
            cache_dir=cfg.base_cache_dir,
            local_files_only=cfg.local_files_only,
        )
    )


def rng_state():
    state = {"python": random.getstate(), "cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["cpu"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def autocast(cfg):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if cfg.precision == "bf16"
        else contextlib.nullcontext()
    )


def fixed_validation(store, collator, cfg):
    # One block per original document; fixed CPU masking isolated from training RNG.
    one_per_doc = {}
    for idx, doc_id in enumerate(store.doc_ids):
        one_per_doc.setdefault(int(doc_id), idx)
    candidates = list(one_per_doc.values())
    selector = random.Random(cfg.seed + 111)
    indices = selector.sample(candidates, min(cfg.validation_samples, len(candidates)))
    batches = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed + 112)
        for index in indices:
            batch = collator([store.example(index)])
            # Short blocks can draw no targets. Guarantee a finite held-out metric.
            if not (batch["labels"] != -100).any():
                ids = store.example(index)["input_ids"]
                eligible = [i for i, tid in enumerate(ids) if tid not in store.special]
                if not eligible:
                    continue
                pos = eligible[0]
                batch["labels"][0, pos] = ids[pos]
                batch["input_ids"][0, pos] = collator.tokenizer.mask_token_id
            batches.append(batch)
    if not batches:
        raise ValueError("Validation contains no maskable text")
    return batches


@torch.inference_mode()
def evaluate(model, batches, cfg):
    saved = rng_state()
    was_training = model.training
    weighted, count = 0.0, 0
    try:
        model.eval()
        for cpu in batches:
            batch = {k: value.to(cfg.device) for k, value in cpu.items()}
            n = int((batch["labels"] != -100).sum().item())
            with autocast(cfg):
                loss = model(**batch).loss.float().item()
            if not math.isfinite(loss):
                raise FloatingPointError("Non-finite held-out loss")
            weighted += loss * n
            count += n
    finally:
        model.train(was_training)
        restore_rng(saved)
    return {"mlm_ce": weighted / count, "masked_targets": count, "blocks": len(batches)}


def training_identity(cfg, year, section, dataset, cache, base):
    settings = {
        name: getattr(cfg, name)
        for name in (
            "target_tokens",
            "save_tokens",
            "max_length",
            "min_text_tokens",
            "mlm_probability",
            "learning_rate",
            "weight_decay",
            "gradient_accumulation",
            "warmup_tokens",
            "seed",
            "validation_samples",
            "attention_implementation",
            "precision",
            "device",
            "gradient_checkpointing",
        )
    }
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "numpy")
    }
    code = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return {
        "version": 1,
        "base_model": model_id(year),
        "base_revision": base.name,
        "base_config_hash": file_hash(base / "config.json"),
        "year": year,
        "section": section,
        "dataset_id": dataset["dataset_id"],
        "cache_signature": cache["signature"],
        "settings": settings,
        "versions": versions,
        "code": code,
    }


def make_optimizer(model, cfg):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (
                no_decay if name.endswith("bias") or "norm" in name.lower() else decay
            ).append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(0.9, 0.99),
        eps=1e-6,
    )


# Only these audited pre-extension sources may migrate to this implementation.
# All other source files, data, versions and training settings must still match.
LEGACY_RESUME_CODE = {
    "training.py": "c14e18bcbd133356779207d9a9cff2ab1dd057e45ee36366de6964a8405d2154",
    "cli.py": "9aca64a4063e36f9bd908a2b7882e3feb8cfb6a425682f89e9b4958a93fa38b0",
}

# Audited predecessor: diagnostics and cleanup only; training math is unchanged.
PRE_DIAGNOSTICS_CODE = {
    "training.py": "a3de5228375ecfd317a9521d72255be8559afe00e59dd972bc29d26388cf8179",
}

PRE_PRESSURE_CLEANUP_CODE = {
    "training.py": "9a08f83a8832f3ee225351b22597c167edaa0ea2d6653830994a108eab3469e6",
}


def compatible_resume_identity(saved, current):
    saved = {**saved, "code": dict(saved["code"])}
    for name, old_hash in LEGACY_RESUME_CODE.items():
        if saved["code"].get(name) == old_hash:
            saved["code"][name] = current["code"][name]
    for name, old_hash in PRE_DIAGNOSTICS_CODE.items():
        if saved["code"].get(name) == old_hash:
            saved["code"][name] = current["code"][name]
    for name, old_hash in PRE_PRESSURE_CLEANUP_CODE.items():
        if saved["code"].get(name) == old_hash:
            saved["code"][name] = current["code"][name]
    return digest(saved) == digest(current)


def log_cuda_memory(cfg, phase, year, section, **extra):
    if cfg.device != "cuda" or not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    stats = torch.cuda.memory_stats()
    mib = 1024 ** 2
    print(json.dumps({
        "event": "cuda_memory", "phase": phase, "year": year, "section": section,
        "allocated_mib": round(torch.cuda.memory_allocated() / mib, 1),
        "reserved_mib": round(torch.cuda.memory_reserved() / mib, 1),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / mib, 1),
        "inactive_split_mib": round(stats.get("inactive_split_bytes.all.current", 0) / mib, 1),
        "allocation_retries": stats.get("num_alloc_retries", 0),
        "device_free_mib": round(free / mib, 1),
        "device_total_mib": round(total / mib, 1), **extra,
    }), flush=True)


def trim_cuda_cache_if_needed(cfg, year, section, step):
    """Return idle cache before variable-length batches exhaust device headroom.

    This is not a tensor-memory limit: live tensors and inactive split blocks
    may prevent reclamation. The before/after records make that visible.
    """
    if cfg.device != "cuda" or not torch.cuda.is_available():
        return False
    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    if reserved < total * 0.70 or reserved - allocated < 1024 ** 3:
        return False
    log_cuda_memory(cfg, "pressure_before_cleanup", year, section, step=step)
    started = time.perf_counter()
    torch.cuda.empty_cache()
    log_cuda_memory(
        cfg, "pressure_after_cleanup", year, section, step=step,
        reclaimed_mib=round((reserved - torch.cuda.memory_reserved()) / 1024 ** 2, 1),
        cleanup_seconds=round(time.perf_counter() - started, 4),
    )
    return True


def run_checkpoints(run):
    return sorted(
        (p for p in (run / "checkpoints").glob("*")
         if not p.name.startswith(".") and (p / "trainer_state.pt").exists()),
        key=lambda p: read_json(p / "training_manifest.json")["steps"],
    )


def _train_job(
    cfg,
    year: int,
    section: str,
    data_dir: Path,
    *,
    resume=False,
    smoke_steps=0,
    target_tokens: int | None = None,
    base_dir: Path | None = None,
):
    """Independent initialization for every year/section. Resume only within this job."""
    cfg.validate()
    if target_tokens is not None and target_tokens < cfg.target_tokens:
        raise ValueError("target_tokens must be at least the configured target_tokens")
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA required; CPU is available only for small fp32 smoke runs"
            )
        if cfg.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 is not supported by this GPU")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    base = base_dir or base_snapshot(cfg, year)
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    cache_root = build_cache(cfg, data_dir, section, tokenizer)
    dataset = read_json(data_dir / "manifest.json")
    if dataset["signature"]["year"] != year:
        raise ValueError("Prepared dataset year does not match base checkpoint year")
    cache = read_json(cache_root / "manifest.json")
    identity = training_identity(cfg, year, section, dataset, cache, base)
    fingerprint = digest(identity)
    run = (
        Path(cfg.output_root)
        / "runs"
        / str(year)
        / section
        / f"seed_{cfg.seed}"
        / fingerprint[:16]
    )
    if resume:
        matches = []
        for identity_path in run.parent.glob("*/identity.json"):
            saved_identity = read_json(identity_path)
            if (compatible_resume_identity(saved_identity, identity)
                    and run_checkpoints(identity_path.parent)):
                matches.append((identity_path.parent, saved_identity))
        if len(matches) > 1:
            raise ValueError("Multiple compatible resume runs found: "
                             + ", ".join(str(path) for path, _ in matches))
        if matches:
            run, identity = matches[0]
            fingerprint = digest(identity)
        elif target_tokens is not None:
            raise FileNotFoundError(f"No compatible checkpoint to extend under {run.parent}")
    completed = run_checkpoints(run)
    if completed and not resume:
        raise FileExistsError(f"Run already has checkpoints; use --resume: {run}")
    checkpoint = completed[-1] if completed else None
    previous_target = (
        read_json(checkpoint / "training_manifest.json")["config"]["target_tokens"]
        if checkpoint else cfg.target_tokens
    )
    if target_tokens is not None and target_tokens < previous_target:
        raise ValueError(f"Cannot reduce previous token goal {previous_target:,} to {target_tokens:,}")
    # The override is mutable run metadata, not part of the experiment identity.
    # A later plain --resume retains the extended goal, including smoke runs.
    cfg = replace(cfg, target_tokens=target_tokens or max(cfg.target_tokens, previous_target))
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "resolved_config.json", cfg.to_dict())
    write_json(run / "identity.json", identity)
    model = AutoModelForMaskedLM.from_pretrained(
        str(checkpoint or base),
        local_files_only=True,
        attn_implementation=cfg.attention_implementation,
    )
    if model.config.max_position_embeddings < cfg.max_length:
        raise ValueError("Model position capacity is shorter than max_length")
    model.config.output_hidden_states = False
    if model.config.model_type == "modernbert":
        model.config.sparse_prediction = True
        model.sparse_prediction = True
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(cfg.device)
    optimizer = make_optimizer(model, cfg)
    train = BlockStore(cache_root, "train", tokenizer)
    val = BlockStore(cache_root, "val", tokenizer)
    collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=True, mlm_probability=cfg.mlm_probability
    )
    fixed = fixed_validation(val, collator, cfg)
    fixture_hash = digest(
        [{key: value.tolist() for key, value in batch.items()} for batch in fixed]
    )
    write_json(
        run / "evaluation_fixture.json",
        {
            "hash": fixture_hash,
            "batches": [
                {key: value.tolist() for key, value in batch.items()} for batch in fixed
            ],
        },
    )
    generator = torch.Generator().manual_seed(cfg.seed + 99)
    permutation = torch.randperm(len(train), generator=generator)
    cursor, steps, processed, trained_seconds, masked_targets = 0, 0, 0, 0.0, 0
    baseline = None
    if checkpoint:
        state = torch.load(
            checkpoint / "trainer_state.pt", map_location="cpu", weights_only=True
        )
        if state["fingerprint"] != fingerprint or state["fixture_hash"] != fixture_hash:
            raise ValueError("Resume fingerprint or evaluation fixture changed")
        optimizer.load_state_dict(state["optimizer"])
        permutation, cursor = state["permutation"], state["cursor"]
        generator.set_state(state["shuffle_generator"])
        steps, processed = state["steps"], state["processed_tokens"]
        trained_seconds, masked_targets = (
            state["train_seconds"],
            state["masked_targets"],
        )
        baseline = state["baseline"]
        restore_rng(state["rng"])
        del state
    else:
        baseline = evaluate(model, fixed, cfg)
        write_json(run / "baseline.json", baseline)
    print(
        json.dumps(
            {
                "year": year,
                "section": section,
                "run": str(run),
                "train": cache["stats"]["train"],
                "baseline": baseline,
                "resumed_tokens": processed,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    milestones = sorted(set([*cfg.save_tokens, cfg.target_tokens]))
    pending = [target for target in milestones if target > processed]
    steps_this_call = 0

    def save(tag, train_loss):
        metrics = evaluate(model, fixed, cfg)
        info = {
            "year": year,
            "section": section,
            "base_model": model_id(year),
            "base_revision": base.name,
            "fingerprint": fingerprint,
            "cutoff": dataset["signature"]["cutoff"],
            "availability_is_rdq_proxy": dataset["availability_is_rdq_proxy"],
            "data_start": dataset["signature"]["start"],
            "dataset_id": dataset["dataset_id"],
            "processed_tokens": processed,
            "masked_targets": masked_targets,
            "steps": steps,
            "train_seconds": trained_seconds,
            "train_loss": train_loss,
            "approx_epochs": processed
            / cache["stats"]["train"]["unique_position_tokens"],
            "baseline": baseline,
            "validation": metrics,
            "fixture_hash": fixture_hash,
            "completed": processed >= cfg.target_tokens,
            "config": cfg.to_dict(),
        }
        checkpoints = run / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        dest = checkpoints / tag
        if dest.exists():
            raise FileExistsError(f"Refusing to replace checkpoint: {dest}")
        tmp = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=checkpoints))
        try:
            model.save_pretrained(tmp, safe_serialization=True)
            tokenizer.save_pretrained(tmp)
            write_json(tmp / "training_manifest.json", info)
            # Save after evaluation; evaluate restores training RNG and model mode.
            torch.save(
                {
                    "fingerprint": fingerprint,
                    "fixture_hash": fixture_hash,
                    "optimizer": optimizer.state_dict(),
                    "rng": rng_state(),
                    "permutation": permutation,
                    "cursor": cursor,
                    "shuffle_generator": generator.get_state(),
                    "steps": steps,
                    "processed_tokens": processed,
                    "masked_targets": masked_targets,
                    "train_seconds": trained_seconds,
                    "baseline": baseline,
                },
                tmp / "trainer_state.pt",
            )
            tmp.rename(dest)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
        write_json(
            run / "latest.json",
            {
                "checkpoint": dest.relative_to(run).as_posix(),
                "processed_tokens": processed,
                "steps": steps,
            },
        )
        print(f"Saved {dest}: validation CE={metrics['mlm_ce']:.5f}", flush=True)
        return dest

    if not pending:
        if not checkpoint:
            raise RuntimeError("No completed checkpoint")
        return export_checkpoint(
            checkpoint, Path(cfg.output_root) / "exports", year, section
        )
    latest = checkpoint
    model.train()
    log_tokens, log_seconds = processed, trained_seconds
    session_tokens, session_seconds = processed, trained_seconds
    last_cache_trim_step = steps - 25
    log_cuda_memory(cfg, "training_start", year, section, resumed_tokens=processed)
    while pending:
        # Rate-limit unsuccessful reclamation too; never flush every step.
        if steps - last_cache_trim_step >= 25:
            cleanup_started = time.perf_counter()
            if trim_cuda_cache_if_needed(cfg, year, section, steps):
                trained_seconds += time.perf_counter() - cleanup_started
                last_cache_trim_step = steps
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, step_tokens, step_targets = 0.0, 0, 0
        microbatches = []
        for _ in range(cfg.gradient_accumulation):
            # 무작위 샘플링
            if cursor >= len(permutation):
                permutation = torch.randperm(len(train), generator=generator)
                cursor = 0
            example = train.example(int(permutation[cursor]))
            cursor += 1

            # 학습 데이터 생성
            cpu = collator([example])
            if not (
                cpu["labels"] != -100
            ).any():  # MLM target이 없는 경우, 첫 번째 maskable token을 강제로 선택
                positions = [
                    i
                    for i, value in enumerate(example["special_tokens_mask"])
                    if not value
                ]
                if not positions:
                    raise ValueError("Training block has no maskable tokens")
                pos = positions[0]
                cpu["labels"][0, pos] = example["input_ids"][pos]
                cpu["input_ids"][0, pos] = tokenizer.mask_token_id
            step_tokens += int(cpu["attention_mask"].sum().item())
            step_targets += int((cpu["labels"] != -100).sum().item())
            microbatches.append(cpu)

        # Weight variable-length microbatches by their actual MLM targets.
        for cpu in microbatches:
            weight = int((cpu["labels"] != -100).sum().item()) / step_targets
            batch = {key: value.to(cfg.device) for key, value in cpu.items()}
            with autocast(cfg):
                loss = (
                    model(**batch).loss * weight
                )  # 가변 길이 마이크로배치를 실제 MLM 타겟 수로 가중치 적용
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"Non-finite loss at step {steps + 1}")
            loss.backward()
            loss_sum += loss.detach().float().item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        # Token-based warmup has no separate scheduler state to lose on resume.
        scale = (
            min(1.0, (processed + step_tokens) / cfg.warmup_tokens)
            if cfg.warmup_tokens
            else 1.0
        )
        for group in optimizer.param_groups:
            group["lr"] = cfg.learning_rate * scale
        optimizer.step()
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        trained_seconds += time.perf_counter() - started
        processed += step_tokens
        masked_targets += step_targets
        steps += 1
        steps_this_call += 1
        if steps <= 5 or steps % cfg.log_every == 0:
            recent_speed = (processed - log_tokens) / max(trained_seconds - log_seconds, 1e-9)
            session_speed = (processed - session_tokens) / max(trained_seconds - session_seconds, 1e-9)
            print(
                f"{year}/{section} step={steps} tokens={processed:,}/{cfg.target_tokens:,} "
                f"loss={loss_sum:.5f} tokens/sec={processed / max(trained_seconds, 1e-9):.0f} "
                f"recent_tokens/sec={recent_speed:.0f} session_tokens/sec={session_speed:.0f}",
                flush=True,
            )
            log_cuda_memory(cfg, "step", year, section, step=steps)
            log_tokens, log_seconds = processed, trained_seconds
        crossed = [target for target in pending if processed >= target]
        if crossed:
            # A whole optimizer step can cross multiple very small smoke milestones.
            latest = save(f"step_{steps:08d}_tokens_{processed:012d}", loss_sum)
            pending = [target for target in pending if target > processed]
        if smoke_steps and steps_this_call >= smoke_steps and pending:
            if not crossed:
                latest = save(f"step_{steps:08d}_tokens_{processed:012d}", loss_sum)
            print("Smoke run stopped; resume this job with --resume", flush=True)
            return latest
    return export_checkpoint(latest, Path(cfg.output_root) / "exports", year, section)


def train_job(
    cfg,
    year: int,
    section: str,
    data_dir: Path,
    *,
    resume=False,
    smoke_steps=0,
    target_tokens: int | None = None,
    base_dir: Path | None = None,
):
    validate_year(year)
    if section not in ("presentation", "qna"):
        raise ValueError("Invalid section")
    with job_lock(Path(cfg.output_root) / "locks" / f"{year}_{section}.lock"):
        log_cuda_memory(cfg, "before_cleanup", year, section)
        gc.collect()
        if cfg.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        log_cuda_memory(cfg, "after_cleanup", year, section)
        try:
            return _train_job(
                cfg, year, section, data_dir, resume=resume,
                smoke_steps=smoke_steps, target_tokens=target_tokens, base_dir=base_dir,
            )
        finally:
            log_cuda_memory(cfg, "job_end_before_cleanup", year, section)
            gc.collect()
            if cfg.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_cuda_memory(cfg, "job_end_after_cleanup", year, section)
