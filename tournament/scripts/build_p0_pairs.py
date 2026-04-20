#!/usr/bin/env python3
"""Build the base pair pack by sloppifying clean source-pool texts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hill_climb.tournament.data import sloppify_candidates
from hill_climb.tournament.io import read_jsonl, write_json, write_jsonl
from hill_climb.tournament.scoring import load_canonical_pangram_module, length_ratio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the P0 base pair pack")
    p.add_argument("--source-pool", default="tournament/data/source_pool.jsonl")
    p.add_argument("--out", default="tournament/data/packs/p0_pairs.jsonl")
    p.add_argument("--manifest-out", default="tournament/data/packs/p0_manifest.json")
    p.add_argument("--max-candidates-per-source", type=int, default=3)
    p.add_argument("--min-length-ratio", type=float, default=0.65)
    p.add_argument("--max-length-ratio", type=float, default=1.60)
    p.add_argument("--min-delta", type=float, default=-1.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.source_pool)
    if not rows:
        raise FileNotFoundError(f"No source pool found at {args.source_pool}")

    module = load_canonical_pangram_module()
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer_tok, scorer_mdl = module.load_editlens(device)

    candidates = []
    for idx, row in enumerate(rows):
        target_text = row["text"]
        target_score = module.editlens_score(scorer_tok, scorer_mdl, target_text, device)
        row_candidates = []
        for mode_name, candidate in sloppify_candidates(target_text, seed=idx * 17):
            cand_score = module.editlens_score(scorer_tok, scorer_mdl, candidate, device)
            ratio = length_ratio(target_text, candidate)
            delta = cand_score - target_score
            if not (args.min_length_ratio <= ratio <= args.max_length_ratio):
                continue
            if delta < args.min_delta:
                continue
            row_candidates.append(
                {
                    "example_id": f"p0-{idx}-{mode_name}",
                    "source_id": row["id"],
                    "split": row["split"],
                    "input_text": candidate,
                    "target_text": target_text,
                    "input_score_roberta": cand_score,
                    "target_score_roberta": target_score,
                    "generation_mode": mode_name,
                    "length_ratio": ratio,
                    "delta": delta,
                }
            )
        if not row_candidates:
            continue
        row_candidates.sort(
            key=lambda item: (item["delta"], item["input_score_roberta"]),
            reverse=True,
        )
        candidates.extend(row_candidates[: args.max_candidates_per_source])

    kept = []
    for row in candidates:
        stamped = dict(row)
        if row["delta"] >= 0.15 and 0.80 <= row["length_ratio"] <= 1.30:
            stamped["acceptance_reason"] = "strong"
        elif row["delta"] >= 0.10:
            stamped["acceptance_reason"] = "fallback"
        else:
            stamped["acceptance_reason"] = "no_threshold_topk"
        kept.append(stamped)

    write_jsonl(args.out, kept)
    manifest = {
        "accepted_rows": len(kept),
        "source_rows": len(rows),
        "candidate_rows": len(candidates),
        "acceptance_mode": "topk_no_threshold",
        "max_candidates_per_source": args.max_candidates_per_source,
        "min_length_ratio": args.min_length_ratio,
        "max_length_ratio": args.max_length_ratio,
        "min_delta": args.min_delta,
        "mean_delta": sum(row["delta"] for row in kept) / len(kept) if kept else 0.0,
    }
    write_json(args.manifest_out, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
