from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .common import digest, file_hash, model_id, read_json, validate_year, write_json


def export_checkpoint(checkpoint: Path, export_root: Path, year: int, section: str) -> Path:
    """Publish a self-contained, HF-cache-compatible tree. Never touch the base cache."""
    validate_year(year)
    if section not in ("presentation", "qna"):
        raise ValueError("Invalid section")
    metadata = read_json(checkpoint / "training_manifest.json")
    if metadata["year"] != year or metadata["section"] != section or metadata["base_model"] != model_id(year):
        raise ValueError("Checkpoint identity does not match requested export")
    files = {p.relative_to(checkpoint).as_posix(): file_hash(p)
             for p in sorted(checkpoint.rglob("*")) if p.is_file() and p.name != "trainer_state.pt"}
    if "config.json" not in files or not any(name.endswith(".safetensors") for name in files):
        raise ValueError("Export requires save_pretrained config and safetensors weights")
    # A 40-character local content revision, separate from the upstream commit.
    revision = digest(files)[:40]
    repo = export_root / section / ("models--" + model_id(year).replace("/", "--"))
    snapshots = repo / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshot = snapshots / revision
    if not snapshot.exists():
        tmp = Path(tempfile.mkdtemp(prefix=".export-", dir=snapshots))
        try:
            for name, sha in files.items():
                source = checkpoint / name
                blob = repo / "blobs" / sha
                blob.parent.mkdir(parents=True, exist_ok=True)
                if blob.exists() and file_hash(blob) != sha:
                    raise ValueError(f"Corrupt export blob: {blob}")
                if not blob.exists():
                    staging = blob.with_suffix(".tmp")
                    shutil.copyfile(source, staging)
                    os.replace(staging, blob)
                dest = tmp / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Relative links remain valid when the whole exports folder is moved.
                dest.symlink_to(os.path.relpath(blob, dest.parent))
            tmp.rename(snapshot)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
    for name, sha in files.items():
        if file_hash(snapshot / name) != sha:
            raise ValueError(f"Invalid snapshot: {snapshot / name}")
    # Some Transformers versions probe PEFT adapters even for full-model loading.
    # Record the known absence so loading by repository id remains offline, too.
    if "adapter_config.json" not in files:
        absent = repo / ".no_exist" / revision / "adapter_config.json"
        absent.parent.mkdir(parents=True, exist_ok=True)
        absent.touch()
    (repo / "refs").mkdir(exist_ok=True)
    ref_tmp = repo / "refs" / "main.tmp"
    ref_tmp.write_text(revision, encoding="ascii")
    os.replace(ref_tmp, repo / "refs" / "main")
    entry = {"year": year, "section": section, "base_model": model_id(year),
             "local_revision": revision, "snapshot": snapshot.relative_to(export_root).as_posix(),
             "processed_tokens": metadata["processed_tokens"], "cutoff": metadata["cutoff"],
             "available_for_prediction_from": f"{year + 1}-01-01",
             "availability_is_rdq_proxy": metadata["availability_is_rdq_proxy"],
             "files": files}
    write_json(export_root / "manifests" / f"{year}_{section}.json", entry)
    return snapshot


def resolve_export(export_root: Path, year: int, section: str) -> Path:
    validate_year(year)
    if section not in ("presentation", "qna"):
        raise ValueError("Invalid section")
    root = export_root.resolve()
    entry = read_json(root / "manifests" / f"{year}_{section}.json")
    snapshot = root / entry["snapshot"]
    if not snapshot.resolve().is_relative_to(root):
        raise ValueError("Snapshot must remain inside the export folder")
    for name, sha in entry["files"].items():
        path = snapshot / name
        if not path.resolve().is_relative_to(root) or file_hash(path) != sha:
            raise ValueError(f"Missing or corrupt exported model file: {path}")
    return snapshot
