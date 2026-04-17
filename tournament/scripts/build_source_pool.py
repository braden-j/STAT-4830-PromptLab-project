#!/usr/bin/env python3
"""Build the frozen phase-1 source pool from Pangram, Kaggle, and DefAn."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hill_climb.tournament.config import SourcePoolConfig, load_yaml_config
from hill_climb.tournament.data import (
    assign_splits,
    download_defan,
    load_defan_records,
    load_kaggle_essays,
    normalize_text,
    trim_word_window,
)
from hill_climb.tournament.io import write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the tournament source pool")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default="tournament/data/source_pool.jsonl")
    p.add_argument("--manifest-out", default="tournament/data/source_pool_manifest.json")
    return p.parse_args()


def gather_pangram(limit: int, min_words: int, max_words: int) -> list[dict[str, object]]:
    """Stream Pangram human-written records."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required; install with `pip install .[tournament]`") from exc

    rows = []
    seen: set[str] = set()
    ds = load_dataset("pangram/editlens_iclr", split="train", streaming=True)
    for idx, row in enumerate(ds):
        if row.get("text_type") != "human_written":
            continue
        text = trim_word_window(normalize_text(row.get("text")), min_words, max_words)
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append(
            {
                "id": f"pangram-{idx}",
                "domain": "essay",
                "source_dataset": f"pangram_{row.get('source', 'unknown')}",
                "text": text,
                "word_count": len(text.split()),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config, "source") if args.config else SourcePoolConfig()
    total = cfg.train_size + cfg.val_size + cfg.test_size
    pangram_target = int(total * cfg.pangram_share)
    kaggle_target = int(total * cfg.kaggle_share)
    defan_target = total - pangram_target - kaggle_target

    pangram_rows = gather_pangram(pangram_target, cfg.min_words, cfg.max_words)
    kaggle_rows = load_kaggle_essays(cfg.kaggle_path, cfg.min_words, cfg.max_words)[:kaggle_target]
    defan_path = download_defan(cfg.defan_path)
    defan_rows = load_defan_records(defan_path, cfg.min_words, cfg.max_words)[:defan_target]

    combined = list(pangram_rows) + list(kaggle_rows) + list(defan_rows)
    if len(combined) < total:
        backfill_needed = total - len(combined)
        extra = gather_pangram(pangram_target + backfill_needed * 2, cfg.min_words, cfg.max_words)
        existing_texts = {row["text"] for row in combined}
        for row in extra:
            if row["text"] not in existing_texts:
                combined.append(row)
                existing_texts.add(row["text"])
            if len(combined) >= total:
                break

    rng = random.Random(cfg.seed)
    rng.shuffle(combined)
    combined = combined[:total]
    stamped = assign_splits(combined, cfg.train_size, cfg.val_size, cfg.test_size, cfg.seed)

    manifest = {
        "requested_total": total,
        "built_total": len(stamped),
        "pangram_rows": sum(1 for row in stamped if str(row["source_dataset"]).startswith("pangram")),
        "kaggle_rows": sum(1 for row in stamped if row["source_dataset"] == "aeon_essays"),
        "defan_rows": sum(1 for row in stamped if row["source_dataset"] == "defan_public"),
    }
    write_jsonl(args.out, stamped)
    write_json(args.manifest_out, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
