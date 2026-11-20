from __future__ import annotations

import dataclasses
import importlib.util
import json
import shutil
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from chrono_dapt.common import digest, file_hash, read_json, write_json
from chrono_dapt.config import Config, load_config
from chrono_dapt.data import prepare_year, temporal_split, token_blocks
from chrono_dapt.export import export_checkpoint, resolve_export


def make_db(path):
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE call_sections(call_id INTEGER PRIMARY KEY, rdq TEXT, presentation TEXT, qna TEXT)")
        for year in (2012, 2013, 2014):
            for day in range(1, 11):
                text = f"company earnings revenue growth margin profit {day} " * 8
                conn.execute("INSERT INTO call_sections VALUES(?,?,?,?)",
                             (year * 100 + day, f"{year}-01-{day:02d}", text + "presentation", text + "question answer"))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / "calls.db"
        make_db(self.db)
        self.cfg = Config(db_path=str(self.db), output_root=str(self.root / "artifacts"), validation_ratio=0.2)

    def tearDown(self):
        self.tmp.cleanup()

    def test_shared_date_split_and_cutoff(self):
        folder = prepare_year(self.cfg, 2013)
        manifest = read_json(folder / "manifest.json")
        split = read_json(folder / "split.json")["calls"]
        self.assertTrue(all(c["available_date"].startswith("2013") for c in split))
        ids = {}
        for section in ("presentation", "qna"):
            for part in ("train", "val"):
                rows = [json.loads(line) for line in (folder / f"{section}_{part}.jsonl").read_text().splitlines()]
                ids[section, part] = {r["call_id"] for r in rows}
        self.assertFalse(ids["presentation", "train"] & ids["qna", "val"])
        self.assertEqual(ids["presentation", "val"], ids["qna", "val"])
        self.assertEqual(prepare_year(self.cfg, 2013), folder)
        self.assertTrue(manifest["availability_is_rdq_proxy"])
        # Cumulative mode includes older data, but never future data.
        cumulative = prepare_year(dataclasses.replace(self.cfg, lookback_years=0), 2013)
        years = {c["available_date"][:4] for c in read_json(cumulative / "split.json")["calls"]}
        self.assertEqual(years, {"2012", "2013"})

    def test_duplicate_text_cannot_leak_to_validation(self):
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("UPDATE call_sections SET presentation='same financial statement' WHERE rdq IN ('2013-01-01','2013-01-10')")
        folder = prepare_year(self.cfg, 2013)
        texts = {}
        for part in ("train", "val"):
            texts[part] = {json.loads(line)["text"] for line in (folder / f"presentation_{part}.jsonl").read_text().splitlines()}
        self.assertFalse(texts["train"] & texts["val"])
        self.assertIn("same financial statement", texts["val"])

    def test_equal_date_is_never_split(self):
        split = temporal_split([("1", "2013-01-01"), ("2", "2013-02-01"), ("3", "2013-02-01")], 0.5)
        self.assertEqual(split["2"], split["3"])

    def test_changed_prepared_file_is_rejected(self):
        folder = prepare_year(self.cfg, 2013)
        with (folder / "qna_train.jsonl").open("a") as stream:
            stream.write("changed")
        with self.assertRaisesRegex(ValueError, "Prepared data changed"):
            prepare_year(self.cfg, 2013)

    def test_block_boundaries_preserve_tokens(self):
        blocks = list(token_blocks(list(range(10, 30)), 10, 2, 1, 2))
        self.assertEqual([token for block in blocks for token in block[1:-1]], list(range(10, 30)))
        self.assertTrue(all(len(block) <= 10 for block in blocks))

    def test_default_config_has_24_independent_jobs(self):
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs/default.json")
        self.assertEqual((cfg.end_year - cfg.start_year + 1) * len(cfg.sections), 24)
        self.assertTrue(Path(cfg.db_path).is_absolute())

    def test_target_tokens_cli_validation(self):
        from chrono_dapt.cli import main, parser

        for command in ("train", "run"):
            args = parser().parse_args([command, "--resume", "--target-tokens", "20000000"])
            self.assertTrue(args.resume)
            self.assertEqual(args.target_tokens, 20_000_000)
        with self.assertRaises(SystemExit) as error:
            main(["train", "--resume", "--target-tokens", "0"])
        self.assertEqual(error.exception.code, 2)


class ExportTests(unittest.TestCase):
    def test_portable_cache_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            write_json(checkpoint / "config.json", {"model_type": "bert"})
            (checkpoint / "model.safetensors").write_bytes(b"test weights")
            (checkpoint / "trainer_state.pt").write_bytes(b"not for deployment")
            write_json(checkpoint / "training_manifest.json", {
                "year": 2013, "section": "qna", "base_model": "manelalab/chrono-bert-v1-20131231",
                "processed_tokens": 10_000_002, "cutoff": "2013-12-31", "availability_is_rdq_proxy": True,
            })
            exports = root / "exports"
            snapshot = export_checkpoint(checkpoint, exports, 2013, "qna")
            self.assertFalse((snapshot / "trainer_state.pt").exists())
            self.assertTrue((snapshot / "model.safetensors").is_symlink())
            moved = root / "moved" / "exports"
            moved.parent.mkdir()
            shutil.move(exports, moved)
            resolved = resolve_export(moved, 2013, "qna")
            self.assertEqual((resolved / "model.safetensors").read_bytes(), b"test weights")
            (resolved / "model.safetensors").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                resolve_export(moved, 2013, "qna")


@unittest.skipUnless(importlib.util.find_spec("transformers"), "transformers environment required")
class TrainingTests(unittest.TestCase):
    def test_training_resume_and_external_loading(self):
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast, AutoModel, AutoModelForMaskedLM, AutoTokenizer
        from huggingface_hub import hf_hub_download
        from chrono_dapt.training import train_job

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "calls.db"
            make_db(db)
            base = root / "tiny_base"
            base.mkdir()
            vocab = {word: i for i, word in enumerate([
                "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "company", "earnings", "revenue",
                "growth", "margin", "profit", "presentation", "question", "answer", *map(str, range(1, 11))])}
            backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
            backend.pre_tokenizer = pre_tokenizers.Whitespace()
            backend.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]",
                                                                    special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                                                cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
            tokenizer.save_pretrained(base)
            torch.manual_seed(9)
            BertForMaskedLM(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                                      num_attention_heads=2, intermediate_size=32,
                                      max_position_embeddings=64)).save_pretrained(base)
            cfg = Config(db_path=str(db), output_root=str(root / "full"), device="cpu", precision="fp32",
                         max_length=32, min_text_tokens=4, target_tokens=256, save_tokens=(128, 256),
                         validation_samples=2, gradient_accumulation=2, warmup_tokens=64,
                         gradient_checkpointing=False)
            data = prepare_year(cfg, 2013)
            full_snapshot = train_job(cfg, 2013, "qna", data, base_dir=base)
            interrupted = dataclasses.replace(cfg, output_root=str(root / "interrupted"))
            interrupted_data = prepare_year(interrupted, 2013)
            smoke = train_job(interrupted, 2013, "qna", interrupted_data, base_dir=base, smoke_steps=1)
            self.assertTrue((smoke / "trainer_state.pt").exists())
            resumed_snapshot = train_job(interrupted, 2013, "qna", interrupted_data, base_dir=base, resume=True)
            full_model = AutoModelForMaskedLM.from_pretrained(full_snapshot, local_files_only=True)
            resumed_model = AutoModelForMaskedLM.from_pretrained(resumed_snapshot, local_files_only=True)
            for name, value in full_model.state_dict().items():
                self.assertTrue(torch.equal(value, resumed_model.state_dict()[name]), name)
            with self.assertRaises(FileExistsError):
                train_job(interrupted, 2013, "qna", interrupted_data, base_dir=base)
            exports = root / "interrupted" / "exports"
            moved = root / "other_project" / "exports"
            moved.parent.mkdir()
            shutil.move(exports, moved)
            snapshot = resolve_export(moved, 2013, "qna")
            tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
            encoder = AutoModel.from_pretrained(snapshot, local_files_only=True).eval()
            with torch.inference_mode():
                output = encoder(**tok("company earnings", return_tensors="pt"))
            self.assertEqual(output.last_hidden_state.shape[-1], 16)
            # The cache can also be resolved using the original HF repository id.
            cached = hf_hub_download("manelalab/chrono-bert-v1-20131231", "config.json",
                                     cache_dir=moved / "qna", local_files_only=True)
            self.assertEqual(file_hash(Path(cached)), file_hash(snapshot / "config.json"))
            with patch("socket.socket.connect", side_effect=AssertionError("Unexpected network access")):
                cached_model = AutoModelForMaskedLM.from_pretrained(
                    "manelalab/chrono-bert-v1-20131231", cache_dir=moved / "qna", local_files_only=True,
                    adapter_kwargs={"cache_dir": moved / "qna", "local_files_only": True})
            self.assertEqual(cached_model.config.hidden_size, 16)

            # Migrate a real optimizer/RNG checkpoint from the pre-extension code.
            from chrono_dapt.training import LEGACY_RESUME_CODE, compatible_resume_identity
            from chrono_dapt.cli import main

            run = smoke.parent.parent
            identity = read_json(run / "identity.json")
            legacy = {**identity, "code": {**identity["code"], **LEGACY_RESUME_CODE}}
            self.assertTrue(compatible_resume_identity(legacy, identity))
            changed = {**legacy, "settings": {**legacy["settings"], "learning_rate": 1e-4}}
            self.assertFalse(compatible_resume_identity(changed, identity))
            changed_code = {**legacy, "code": {**legacy["code"], "training.py": "unknown"}}
            self.assertFalse(compatible_resume_identity(changed_code, identity))
            legacy_fingerprint = digest(legacy)
            write_json(run / "identity.json", legacy)
            for path in (run / "checkpoints").iterdir():
                state = torch.load(path / "trainer_state.pt", weights_only=True)
                state["fingerprint"] = legacy_fingerprint
                torch.save(state, path / "trainer_state.pt")
                manifest = read_json(path / "training_manifest.json")
                manifest["fingerprint"] = legacy_fingerprint
                write_json(path / "training_manifest.json", manifest)
            legacy_run = run.with_name(legacy_fingerprint[:16])
            run.rename(legacy_run)
            old_weights = {p: file_hash(p) for p in legacy_run.glob("checkpoints/*/model.safetensors")}
            config_path = root / "extension.json"
            write_json(config_path, interrupted.to_dict())
            with patch("chrono_dapt.training.base_snapshot", return_value=base):
                main(["--config", str(config_path), "train", "--years", "2013", "--sections", "qna",
                      "--resume", "--target-tokens", "512", "--smoke-steps", "1"])
            latest = legacy_run / read_json(legacy_run / "latest.json")["checkpoint"]
            info = read_json(latest / "training_manifest.json")
            self.assertGreater(info["processed_tokens"], 256)
            self.assertEqual(info["config"]["target_tokens"], 512)
            self.assertFalse(info["completed"])
            # A plain resume keeps the extension goal after interruption.
            extended = train_job(interrupted, 2013, "qna", interrupted_data, base_dir=base, resume=True)
            self.assertGreaterEqual(read_json(extended / "training_manifest.json")["processed_tokens"], 512)
            self.assertEqual(len(list(legacy_run.parent.iterdir())), 1)
            self.assertEqual(old_weights, {p: file_hash(p) for p in old_weights})
            reference_cfg = dataclasses.replace(cfg, output_root=str(root / "reference_extended"))
            reference_data = prepare_year(reference_cfg, 2013)
            reference = train_job(reference_cfg, 2013, "qna", reference_data, base_dir=base, target_tokens=512)
            extended_model = AutoModelForMaskedLM.from_pretrained(extended, local_files_only=True)
            reference_model = AutoModelForMaskedLM.from_pretrained(reference, local_files_only=True)
            for name, value in reference_model.state_dict().items():
                self.assertTrue(torch.equal(value, extended_model.state_dict()[name]), name)
            with self.assertRaisesRegex(ValueError, "Cannot reduce"):
                train_job(interrupted, 2013, "qna", interrupted_data, base_dir=base,
                          resume=True, target_tokens=384)
            missing = dataclasses.replace(cfg, output_root=str(root / "missing"))
            with self.assertRaisesRegex(FileNotFoundError, "No compatible checkpoint"):
                train_job(missing, 2013, "qna", data, base_dir=base, resume=True, target_tokens=512)
            duplicate = legacy_run.with_name("duplicate_run")
            shutil.copytree(legacy_run, duplicate)
            with self.assertRaisesRegex(ValueError, "Multiple compatible resume runs"):
                train_job(interrupted, 2013, "qna", interrupted_data, base_dir=base,
                          resume=True, target_tokens=1024)
            # Exercise ChronoBERT's actual architecture, including sparse MLM and checkpointing.
            from transformers import ModernBertConfig, ModernBertForMaskedLM

            modern_base = root / "tiny_modernbert"
            tokenizer.save_pretrained(modern_base)
            ModernBertForMaskedLM(ModernBertConfig(
                vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=64,
                local_attention=16, layer_types=["full_attention", "sliding_attention"],
                pad_token_id=0, cls_token_id=2, sep_token_id=3, bos_token_id=2, eos_token_id=3,
                reference_compile=False,
            )).save_pretrained(modern_base)
            modern_cfg = dataclasses.replace(cfg, output_root=str(root / "modern"), target_tokens=64,
                                             save_tokens=(64,), gradient_checkpointing=True)
            modern_data = prepare_year(modern_cfg, 2013)
            modern_snapshot = train_job(modern_cfg, 2013, "presentation", modern_data, base_dir=modern_base)
            encoder = AutoModel.from_pretrained(modern_snapshot, local_files_only=True).eval()
            with torch.inference_mode():
                output = encoder(**tok("company earnings", return_tensors="pt"))
            self.assertEqual(output.last_hidden_state.shape[-1], 16)


if __name__ == "__main__":
    unittest.main()
