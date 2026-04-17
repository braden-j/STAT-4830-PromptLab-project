#!/usr/bin/env python3
"""Add curriculum difficulty buckets to the P0 base pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hill_climb.tournament.io import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the P1 curriculum pack")
    p.add_argument("--p0", default="tournament/data/packs/p0_pairs.jsonl")
    p.add_argument("--out", default="tournament/data/packs/p1_curriculum.jsonl")
    p.add_argument("--manifest-out", default="tournament/data/packs/p1_manifest.json")
    return p.parse_args()


def bucket(row: dict[str, object]) -> str:
    """Apply the agreed curriculum thresholds."""

    target = float(row["target_score_roberta"])
    source = float(row["input_score_roberta"])
    delta = source - target
    if target <= 0.10 and source >= 0.80:
        return "easy"
    if source >= 0.60 and delta >= 0.20:
        return "medium"
    return "hard"


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.p0)
    out = []
    counts = {"easy": 0, "medium": 0, "hard": 0}
    for row in rows:
        stamped = dict(row)
        stamped["difficulty_bucket"] = bucket(row)
        counts[stamped["difficulty_bucket"]] += 1
        out.append(stamped)
    write_jsonl(args.out, out)
    write_json(args.manifest_out, counts)
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
