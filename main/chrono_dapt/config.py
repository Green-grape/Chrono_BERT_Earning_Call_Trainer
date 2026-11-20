from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .common import read_json


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    db_path: str = "../../call_sections_rebuilt_new.db"
    output_root: str = "../artifacts"
    base_cache_dir: str | None = None
    base_revision: str = "main"
    start_year: int = 2013
    end_year: int = 2024
    sections: tuple = ("presentation", "qna")
    lookback_years: int = 1
    availability_column: str = "rdq"
    validation_ratio: float = 0.1
    max_calls: int | None = None
    target_tokens: int = 10_000_000
    save_tokens: tuple = (5_000_000, 10_000_000)
    max_length: int = 8192
    min_text_tokens: int = 64
    mlm_probability: float = 0.3
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    gradient_accumulation: int = 2
    warmup_tokens: int = 500_000
    seed: int = 42
    validation_samples: int = 64
    log_every: int = 25
    attention_implementation: str = "sdpa"
    precision: str = "bf16"
    device: str = "cuda"
    local_files_only: bool = True
    gradient_checkpointing: bool = True

    def validate(self):
        if not 2013 <= self.start_year <= self.end_year <= 2024:
            raise ValueError("Expected 2013 <= start_year <= end_year <= 2024")
        if not self.sections or len(set(self.sections)) != len(self.sections) or any(
            s not in ("presentation", "qna") for s in self.sections
        ):
            raise ValueError("sections must contain presentation and/or qna, without duplicates")
        if self.lookback_years < 0 or not 0 < self.validation_ratio < 1:
            raise ValueError("lookback_years >= 0 and 0 < validation_ratio < 1 required")
        if not 4 <= self.max_length <= 8192 or not 1 <= self.min_text_tokens <= self.max_length - 2:
            raise ValueError("Invalid block lengths")
        for name in ("target_tokens", "gradient_accumulation", "validation_samples", "log_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_calls is not None and self.max_calls < 2:
            raise ValueError("max_calls must be >= 2")
        if self.warmup_tokens < 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if not 0 < self.mlm_probability < 1:
            raise ValueError("mlm_probability must be between 0 and 1")
        if any(t <= 0 or t > self.target_tokens for t in self.save_tokens):
            raise ValueError("save_tokens must be positive and <= target_tokens")
        if self.precision not in ("bf16", "fp32") or self.device not in ("cuda", "cpu"):
            raise ValueError("Supported precision: bf16/fp32; devices: cuda/cpu")
        if self.device == "cpu" and self.precision != "fp32":
            raise ValueError("CPU smoke runs require precision=fp32")

    def to_dict(self):
        return asdict(self)


def load_config(path: Path) -> Config:
    path = path.resolve()
    values = read_json(path)
    unknown = set(values) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    cfg = Config(**values)
    for key in ("db_path", "output_root", "base_cache_dir"):
        value = getattr(cfg, key)
        if value is not None:
            setattr(cfg, key, str((path.parent / value).resolve()))
    cfg.sections = tuple(cfg.sections)
    cfg.save_tokens = tuple(sorted(set(cfg.save_tokens)))
    cfg.validate()
    return cfg
