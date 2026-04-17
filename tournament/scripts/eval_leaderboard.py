#!/usr/bin/env python3
"""Aggregate checkpoint eval summaries into a leaderboard."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hill_climb.tournament.io import read_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a leaderboard from eval outputs")
    p.add_argument("--eval-json", nargs="+", required=True)
    p.add_argument("--out-csv", default="tournament/outputs/leaderboards/leaderboard.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for path in args.eval_json:
        with open(path) as f:
            payload = json.load(f)
        payload["run_name"] = Path(path).stem
        rows.append(payload)

    rows.sort(
        key=lambda row: (
            -row.get("mean_delta_editlens", 0.0),
            row.get("mean_output_score", 1.0),
            -row.get("mean_similarity", 0.0),
        )
    )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["run_name", "mean_delta_editlens"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"rows": len(rows), "out_csv": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
