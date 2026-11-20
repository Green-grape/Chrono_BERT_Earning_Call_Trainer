from __future__ import annotations

import datetime as dt
import hashlib
from contextlib import closing
import json
import math
import shutil
import sqlite3
import tempfile
from pathlib import Path

from .common import digest, file_hash, read_json, validate_year, write_json


def temporal_split(calls: list[tuple[str, str]], ratio: float) -> dict[str, str]:
    """Put whole calendar dates in one split, shared by both sections."""
    if len({call_id for call_id, _ in calls}) != len(calls):
        raise ValueError("Duplicate call ids")
    ordered = sorted(calls, key=lambda x: (x[1], x[0]))
    dates = sorted({date for _, date in ordered})
    if len(dates) < 2:
        raise ValueError("At least two distinct available dates are required")
    index = min(len(ordered) - 1, max(1, math.floor(len(ordered) * (1 - ratio))))
    boundary = ordered[index][1]
    if boundary == dates[0]:
        boundary = dates[1]
    return {
        call_id: "val" if date >= boundary else "train" for call_id, date in ordered
    }


def prepare_year(cfg, year: int) -> Path:
    validate_year(year)
    db = Path(cfg.db_path)
    stat = db.stat()
    source = {
        "path": str(db.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    # SQLite WAL files can contain newer data than the main DB file.
    wal = Path(str(db) + "-wal")
    if wal.exists():
        source["wal"] = {"size": wal.stat().st_size, "mtime_ns": wal.stat().st_mtime_ns}
    cutoff = f"{year}-12-31"
    start = (
        f"{year - cfg.lookback_years + 1}-01-01" if cfg.lookback_years else "0001-01-01"
    )
    signature = {
        "version": 1,
        "source": source,
        "year": year,
        "start": start,
        "cutoff": cutoff,
        "availability_column": cfg.availability_column,
        "validation_ratio": cfg.validation_ratio,
    }
    out = Path(cfg.output_root) / "data" / str(year) / digest(signature)[:16]
    if (out / "manifest.json").exists():
        manifest = read_json(out / "manifest.json")
        for name, sha in manifest["files"].items():
            if file_hash(out / name) != sha:
                raise ValueError(f"Prepared data changed: {out / name}")
        return out

    with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        conn.execute("BEGIN")  # Consistent view across the metadata and text reads.
        columns = {r[1] for r in conn.execute("PRAGMA table_info(call_sections)")}
        required = {"call_id", "presentation", "qna", cfg.availability_column}
        if not required <= columns:
            raise ValueError(f"Missing database columns: {sorted(required - columns)}")
        # Interpolate only a column name checked against the actual schema.
        column = '"' + cfg.availability_column.replace('"', '""') + '"'
        where = f"substr({column},1,10) >= ? AND substr({column},1,10) <= ?"
        query = f"SELECT call_id, {column} FROM call_sections WHERE {where} ORDER BY {column}, call_id"
        rows = conn.execute(query, (start, cutoff)).fetchall()
        calls = []
        for call_id, date in rows:
            date = str(date)[:10]
            dt.date.fromisoformat(date)
            calls.append((str(call_id), date))

        # 같은 날짜의 call은 같은 split으로 배정
        unique_dates = sorted({date for _, date in calls})

        boundary_idx = int(len(unique_dates) * (1 - cfg.validation_ratio))
        boundary_idx = max(1, min(boundary_idx, len(unique_dates) - 1))
        val_start = unique_dates[boundary_idx]

        split = {cid: "train" if date < val_start else "val" for cid, date in calls}
        selected = {call_id for call_id, _ in calls}
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".prepare-", dir=out.parent))
        handles = {}
        try:
            counts = {
                f"{s}_{p}": 0 for s in ("presentation", "qna") for p in ("train", "val")
            }
            for key in counts:
                handles[key] = (tmp / f"{key}.jsonl").open("w", encoding="utf-8")
            seen = set()
            duplicates = 0
            # Read held-out calls first: duplicate text in earlier training calls is dropped.
            query = (
                f"SELECT call_id, {column}, presentation, qna FROM call_sections WHERE {where} "
                f"ORDER BY {column} DESC, call_id ASC"  # DESC so val is first, train is last (최근 call이 train에 들어가도록)
            )
            for call_id, date, presentation, qna in conn.execute(
                query, (start, cutoff)
            ):
                call_id = str(call_id)
                if call_id not in selected:
                    continue
                for section, text in (("presentation", presentation), ("qna", qna)):
                    text = (text or "").strip()
                    if not text:
                        continue
                    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    if sha in seen:
                        duplicates += 1
                        continue
                    seen.add(sha)
                    key = f"{section}_{split[call_id]}"
                    row = {
                        "call_id": call_id,
                        "available_date": str(date)[:10],
                        "section": section,
                        "text_sha256": sha,
                        "text": text,
                    }
                    handles[key].write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[key] += 1
            for stream in handles.values():
                stream.close()
            for section in cfg.sections:
                if not counts[f"{section}_train"] or not counts[f"{section}_val"]:
                    raise ValueError(f"Empty {section} split; increase data window")
            write_json(
                tmp / "split.json",
                {
                    "calls": [
                        {"call_id": cid, "available_date": date, "split": split[cid]}
                        for cid, date in calls
                    ]
                },
            )
            files = {p.name: file_hash(p) for p in sorted(tmp.iterdir())}
            manifest = {
                "signature": signature,
                "dataset_id": digest(files),
                "files": files,
                "counts": counts,
                "duplicates_dropped": duplicates,
                "validation_start": min(
                    date for cid, date in calls if split[cid] == "val"
                ),
                "availability_is_rdq_proxy": cfg.availability_column == "rdq",
            }
            write_json(tmp / "manifest.json", manifest)
            tmp.rename(out)
        finally:
            for stream in handles.values():
                stream.close()
            if tmp.exists():
                shutil.rmtree(tmp)
    return out


def tokenizer_signature(tokenizer) -> str:
    return digest(
        {
            "vocab": tokenizer.get_vocab(),
            "special": tokenizer.special_tokens_map,
            "backend": tokenizer.backend_tokenizer.to_str(),
        }
    )


def token_blocks(
    ids: list[int], max_length: int, min_text_tokens: int, cls: int, sep: int
):
    capacity = max_length - 2
    for offset in range(0, len(ids), capacity):
        part = ids[offset : offset + capacity]
        if len(part) >= min_text_tokens:
            yield [cls, *part, sep]


def build_cache(cfg, data_dir: Path, section: str, tokenizer) -> Path:
    import numpy as np

    data = read_json(data_dir / "manifest.json")
    signature = {
        "version": 1,
        "dataset_id": data["dataset_id"],
        "section": section,
        "tokenizer": tokenizer_signature(tokenizer),
        "max_length": cfg.max_length,
        "min_text_tokens": cfg.min_text_tokens,
    }
    root = Path(cfg.output_root) / "cache" / digest(signature)
    if (root / "manifest.json").exists():
        cached = read_json(root / "manifest.json")
        if cached["signature"] != signature:
            raise ValueError("Cache signature mismatch")
        for name, info in cached["files"].items():
            path = root / name
            if (
                not path.exists()
                or path.stat().st_size != info["size"]
                or file_hash(path) != info["sha256"]
            ):
                raise ValueError(f"Incomplete or changed token cache: {path}")
        return root
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("A CLS/SEP tokenizer is required")
    probe = tokenizer("Company earnings", add_special_tokens=False)["input_ids"]
    if tokenizer("Company earnings", add_special_tokens=True)["input_ids"] != [
        tokenizer.cls_token_id,
        *probe,
        tokenizer.sep_token_id,
    ]:
        raise ValueError("Tokenizer does not use the expected CLS/text/SEP layout")
    root.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".tokens-", dir=root.parent))
    stats = {}
    try:
        for split in ("train", "val"):
            offsets, lengths, doc_ids = [], [], []
            total, docs = 0, 0
            with (data_dir / f"{section}_{split}.jsonl").open(
                encoding="utf-8"
            ) as source, (tmp / f"{split}.bin").open("wb") as dest:
                for doc_id, line in enumerate(source):
                    row = json.loads(line)
                    ids = tokenizer(
                        row["text"],
                        add_special_tokens=False,
                        truncation=False,
                        verbose=False,
                    )["input_ids"]
                    docs += 1
                    for block in token_blocks(
                        ids,
                        cfg.max_length,
                        cfg.min_text_tokens,
                        tokenizer.cls_token_id,
                        tokenizer.sep_token_id,
                    ):
                        offsets.append(total)
                        lengths.append(len(block))
                        doc_ids.append(doc_id)
                        np.asarray(block, dtype=np.int32).tofile(dest)
                        total += len(block)
            if not lengths:
                raise ValueError(f"No retained {section}/{split} blocks")
            np.savez(
                tmp / f"{split}.npz",
                offsets=np.asarray(offsets, dtype=np.int64),
                lengths=np.asarray(lengths, dtype=np.int32),
                doc_ids=np.asarray(doc_ids, dtype=np.int64),
            )
            stats[split] = {
                "documents": docs,
                "retained_documents": len(set(doc_ids)),
                "blocks": len(lengths),
                "unique_position_tokens": total,
            }
        files = {
            p.name: {"size": p.stat().st_size, "sha256": file_hash(p)}
            for p in tmp.iterdir()
        }
        write_json(
            tmp / "manifest.json",
            {"signature": signature, "stats": stats, "files": files},
        )
        tmp.rename(root)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return root


class BlockStore:
    def __init__(self, root: Path, split: str, tokenizer):
        import numpy as np

        with np.load(root / f"{split}.npz", allow_pickle=False) as index:
            self.offsets = index["offsets"]
            self.lengths = index["lengths"]
            self.doc_ids = index["doc_ids"]
        self.flat = np.memmap(root / f"{split}.bin", mode="r", dtype=np.int32)
        self.special = set(tokenizer.all_special_ids)

    def __len__(self):
        return len(self.lengths)

    def example(self, index):
        offset, length = int(self.offsets[index]), int(self.lengths[index])
        ids = self.flat[offset : offset + length].tolist()
        return {
            "input_ids": ids,
            "special_tokens_mask": [int(t in self.special) for t in ids],
        }
