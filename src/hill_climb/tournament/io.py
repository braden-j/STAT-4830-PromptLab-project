"""Small IO helpers shared across tournament scripts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def ensure_parent(path: str | Path) -> Path:
    """Create a file's parent directory and return the normalized path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into memory."""

    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write rows to JSONL."""

    path = ensure_parent(path)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a formatted JSON file."""

    path = ensure_parent(path)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    """Load a CSV file as a list of dict rows."""

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def file_sha256(path: str | Path) -> str:
    """Hash a file so runs can log immutable dataset manifests."""

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
