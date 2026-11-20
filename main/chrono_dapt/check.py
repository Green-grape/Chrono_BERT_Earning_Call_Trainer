"""Read-only preflight. Does not import models or download any files."""

from contextlib import closing
from pathlib import Path
import sqlite3

from .common import model_id, read_json


def preflight(cfg, years):
    from huggingface_hub import snapshot_download

    with closing(sqlite3.connect(Path(cfg.db_path).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(call_sections)")}
        if cfg.availability_column not in columns:
            raise ValueError(f"Missing availability column: {cfg.availability_column}")
        column = '"' + cfg.availability_column.replace('"', '""') + '"'
        rows = conn.execute(f"SELECT substr({column},1,4), count(*) FROM call_sections GROUP BY 1").fetchall()
        counts = {str(year): count for year, count in rows}
    jobs = []
    for year in years:
        item = {"year": year, "model": model_id(year), "calls_in_checkpoint_year": counts.get(str(year), 0)}
        try:
            snapshot = Path(snapshot_download(model_id(year), revision=cfg.base_revision,
                                              cache_dir=cfg.base_cache_dir, local_files_only=True))
            config = read_json(snapshot / "config.json")
            weights = list(snapshot.glob("*.safetensors"))
            index = snapshot / "model.safetensors.index.json"
            if index.exists():
                weights = [snapshot / name for name in set(read_json(index)["weight_map"].values())]
            item.update(snapshot=str(snapshot), model_type=config.get("model_type"),
                        position_capacity=config.get("max_position_embeddings"),
                        weights_present=bool(weights) and all(p.is_file() for p in weights),
                        tokenizer_present=(snapshot / "tokenizer.json").is_file())
            item["ready"] = (item["weights_present"] and item["tokenizer_present"]
                             and (item["position_capacity"] or 0) >= cfg.max_length)
        except (OSError, ValueError, KeyError) as exc:
            item.update(ready=False, error=str(exc))
        jobs.append(item)
    return {"availability_column": cfg.availability_column, "calls_by_year": counts,
            "base_models": jobs, "all_models_cached": all(item["ready"] for item in jobs)}
