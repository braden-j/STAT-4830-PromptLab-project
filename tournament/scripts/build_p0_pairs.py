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
        best = None
        for mode_name, candidate in sloppify_candidates(target_text, seed=idx * 17):
            cand_score = module.editlens_score(scorer_tok, scorer_mdl, candidate, device)
            ratio = length_ratio(target_text, candidate)
            delta = cand_score - target_score
            if best is None or delta > best["delta"] or (
                delta == best["delta"] and cand_score > best["input_score_roberta"]
            ):
                best = {
                    "example_id": f"p0-{idx}",
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
        if best is None:
            continue
        candidates.append(best)

    strict_kept = []
    permissive_kept = []
    for row in candidates:
        strong = row["delta"] >= 0.15 and 0.80 <= row["length_ratio"] <= 1.30
        fallback = row["delta"] >= 0.10
        permissive = (
            row["input_text"] != row["target_text"]
            and 0.70 <= row["length_ratio"] <= 1.50
        )
        if strong:
            stamped = dict(row)
            stamped["acceptance_reason"] = "strong"
            strict_kept.append(stamped)
        elif fallback:
            stamped = dict(row)
            stamped["acceptance_reason"] = "fallback"
            strict_kept.append(stamped)
        elif permissive:
            stamped = dict(row)
            stamped["acceptance_reason"] = "permissive_fallback"
            permissive_kept.append(stamped)

    kept = strict_kept if strict_kept else permissive_kept
    mode = "strict" if strict_kept else "permissive_fallback"

    write_jsonl(args.out, kept)
    manifest = {
        "accepted_rows": len(kept),
        "source_rows": len(rows),
        "candidate_rows": len(candidates),
        "strict_rows": len(strict_kept),
        "permissive_rows": len(permissive_kept),
        "acceptance_mode": mode,
    }
    write_json(args.manifest_out, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
