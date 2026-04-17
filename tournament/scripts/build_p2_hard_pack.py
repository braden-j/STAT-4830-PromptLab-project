#!/usr/bin/env python3
"""Build the hard-negative weighted pack from P0 and optional M2 failures."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hill_climb.tournament.io import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the P2 hard pack")
    p.add_argument("--p0", default="tournament/data/packs/p0_pairs.jsonl")
    p.add_argument("--m2-failures", default=None)
    p.add_argument("--out", default="tournament/data/packs/p2_hard.jsonl")
    p.add_argument("--manifest-out", default="tournament/data/packs/p2_manifest.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.p0)
    if not rows:
        raise FileNotFoundError(f"No P0 pack found at {args.p0}")

    by_score = sorted(rows, key=lambda r: float(r["input_score_roberta"]), reverse=True)
    by_delta = sorted(rows, key=lambda r: float(r["input_score_roberta"]) - float(r["target_score_roberta"]))
    top_count = max(1, math.floor(len(rows) * 0.50))
    low_delta_count = max(1, math.floor(len(rows) * 0.30))

    picked: dict[str, dict[str, object]] = {}
    for row in by_score[:top_count]:
        stamped = dict(row)
        stamped["sample_weight"] = 2.0
        stamped["hard_reason"] = "top_pangram_input"
        picked[stamped["example_id"]] = stamped
    for row in by_delta[:low_delta_count]:
        stamped = dict(row)
        stamped["sample_weight"] = 1.5
        stamped["hard_reason"] = "lowest_delta"
        picked[stamped["example_id"]] = stamped

    failure_rows = read_jsonl(args.m2_failures) if args.m2_failures else []
    for row in failure_rows:
        source_id = row.get("source_id")
        matched = next((candidate for candidate in rows if candidate["source_id"] == source_id), None)
        if matched is None:
            continue
        stamped = dict(matched)
        stamped["sample_weight"] = 1.2
        stamped["hard_reason"] = "m2_failure"
        picked[stamped["example_id"]] = stamped

    out = list(picked.values())
    manifest = {
        "rows": len(out),
        "top_pangram": sum(1 for row in out if row["hard_reason"] == "top_pangram_input"),
        "lowest_delta": sum(1 for row in out if row["hard_reason"] == "lowest_delta"),
        "m2_failure": sum(1 for row in out if row["hard_reason"] == "m2_failure"),
    }
    write_jsonl(args.out, out)
    write_json(args.manifest_out, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
