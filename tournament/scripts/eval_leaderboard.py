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
            -row.get("pass_rate", 0.0),
            -row.get("mean_delta_editlens", 0.0),
            row.get("mean_output_score", 1.0),
            -row.get("mean_similarity", 0.0),
            row.get("mean_length_gap", 1.0),
        )
    )

    if rows:
        leader_delta = rows[0].get("mean_delta_editlens", 0.0)
        leader_pass = rows[0].get("pass_rate", 0.0)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
            row["delta_from_leader"] = row.get("mean_delta_editlens", 0.0) - leader_delta
            row["pass_rate_from_leader"] = row.get("pass_rate", 0.0) - leader_pass

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "rank",
        "run_name",
        "rows",
        "rows_passed",
        "pass_rate",
        "mean_input_score",
        "mean_delta_editlens",
        "delta_from_leader",
        "mean_delta_editlens_passed",
        "mean_output_score",
        "mean_similarity",
        "median_length_ratio",
        "mean_length_gap",
        "invalid_rate",
        "pass_rate_from_leader",
        "eval_split",
        "max_eval_rows",
        "sort_key",
    ]
    fieldnames = preferred if rows else ["run_name", "mean_delta_editlens"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"rows": len(rows), "out_csv": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
