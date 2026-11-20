from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def model_id(year: int) -> str:
    return f"manelalab/chrono-bert-v1-{year}1231"


def validate_year(year: int):
    if not 2013 <= year <= 2024:
        raise ValueError("Supported annual checkpoints are 2013 through 2024")


@contextmanager
def job_lock(path: Path):
    """Linux/WSL advisory lock; automatically released on interruption or process exit."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is using this job: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
