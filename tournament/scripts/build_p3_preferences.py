#!/usr/bin/env python3
"""Generate Pangram-ranked rewrite preferences for DPO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hill_climb.tournament.io import read_jsonl, write_json, write_jsonl
from hill_climb.tournament.prompts import format_rewrite_prompt
from hill_climb.tournament.scoring import length_ratio, load_canonical_pangram_module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the P3 preference pack")
    p.add_argument("--p0", default="tournament/data/packs/p0_pairs.jsonl")
    p.add_argument("--model-name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--out", default="tournament/data/packs/p3_preferences.jsonl")
    p.add_argument("--manifest-out", default="tournament/data/packs/p3_manifest.json")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
        from datasets import Dataset
        from sentence_transformers import SentenceTransformer, util
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Missing tournament dependencies. Install with `pip install .[tournament]`.") from exc

    rows = read_jsonl(args.p0)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise FileNotFoundError(f"No P0 rows found at {args.p0}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pangram = load_canonical_pangram_module()
    scorer_tok, scorer_mdl = pangram.load_editlens(device)
    sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = model.to(device).eval()

    candidate_settings = [
        {"temperature": 0.7, "top_p": 0.9},
        {"temperature": 0.8, "top_p": 0.95},
        {"temperature": 0.6, "top_p": 0.92},
        {"temperature": 0.9, "top_p": 0.85},
    ]

    out = []
    for idx, row in enumerate(rows):
        prompt = format_rewrite_prompt(row["input_text"])
        prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)
        candidates = []
        reference = row["target_text"]
        ref_emb = sim_model.encode(reference, convert_to_tensor=True)
        for setting in candidate_settings:
            with torch.no_grad():
                generated = model.generate(
                    **prompt_ids,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=setting["temperature"],
                    top_p=setting["top_p"],
                    pad_token_id=tokenizer.pad_token_id,
                )
            completion_ids = generated[0][prompt_ids["input_ids"].shape[1] :]
            candidate = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            if not candidate:
                continue
            cand_emb = sim_model.encode(candidate, convert_to_tensor=True)
            similarity = float(util.cos_sim(ref_emb, cand_emb).item())
            ratio = length_ratio(reference, candidate)
            if similarity < 0.88 or not (0.80 <= ratio <= 1.25):
                continue
            score = pangram.editlens_score(scorer_tok, scorer_mdl, candidate, device)
            candidates.append({"text": candidate, "score": score, "similarity": similarity})
        if len(candidates) < 2:
            continue
        ranked = sorted(candidates, key=lambda item: item["score"])
        chosen = ranked[0]
        rejected = ranked[-1]
        out.append(
            {
                "example_id": f"p3-{idx}",
                "source_id": row["source_id"],
                "split": row["split"],
                "input_text": row["input_text"],
                "chosen_text": chosen["text"],
                "rejected_text": rejected["text"],
                "chosen_score_roberta": chosen["score"],
                "rejected_score_roberta": rejected["score"],
                "similarity_chosen": chosen["similarity"],
                "similarity_rejected": rejected["similarity"],
            }
        )
    write_jsonl(args.out, out)
    write_json(args.manifest_out, {"rows": len(out)})
    print(json.dumps({"rows": len(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
