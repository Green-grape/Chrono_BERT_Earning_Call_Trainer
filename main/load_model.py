"""Copy this file AND chrono_dapt/ to another project, or use the printed snapshot path."""

from pathlib import Path

from chrono_dapt.export import resolve_export


def load_adapted_model(exports_dir, year, section, *, device="cpu", for_mlm=False):
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    snapshot = resolve_export(Path(exports_dir), year, section)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    cls = AutoModelForMaskedLM if for_mlm else AutoModel
    model = cls.from_pretrained(str(snapshot), local_files_only=True, attn_implementation="sdpa")
    model.config.output_hidden_states = False
    return tokenizer, model.to(device).eval()
