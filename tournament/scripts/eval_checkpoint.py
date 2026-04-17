#!/usr/bin/env python3
"""Generate rewrites from a checkpoint and score them with Pangram and similarity."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hill_climb.tournament.config import EvalConfig, load_yaml_config
from hill_climb.tournament.evaluation import classify_eval_row
from hill_climb.tournament.io import read_jsonl, write_json, write_jsonl
from hill_climb.tournament.prompts import format_rewrite_prompt
from hill_climb.tournament.scoring import load_canonical_pangram_module, summarize_gate_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on a rewrite set")
    p.add_argument("--config", required=True)
    p.add_argument("--out-jsonl", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config, "eval")
    assert isinstance(cfg, EvalConfig)

    try:
        import torch
        from sentence_transformers import SentenceTransformer, util
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Missing tournament dependencies. Install with `pip install .[tournament]`.") from exc

    rows = read_jsonl(cfg.eval_path)
    if not rows:
        raise FileNotFoundError(f"No eval rows found at {cfg.eval_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pangram = load_canonical_pangram_module()
    scorer_tok, scorer_mdl = pangram.load_editlens(device)
    sim_model = SentenceTransformer(cfg.similarity_model)

    tokenizer = AutoTokenizer.from_pretrained(cfg.run_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.run_dir).to(device).eval()

    scored_rows = []
    for idx, row in enumerate(rows):
        if row.get("split") == "train":
            continue
        input_text = row.get("input_text") or row["text"]
        target_text = row.get("target_text") or row["text"]
        prompt = format_rewrite_prompt(input_text)
        prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **prompt_ids,
                max_new_tokens=cfg.generation.max_new_tokens,
                do_sample=cfg.generation.do_sample,
                temperature=cfg.generation.temperature,
                top_p=cfg.generation.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        output_ids = generated[0][prompt_ids["input_ids"].shape[1] :]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        input_score = pangram.editlens_score(scorer_tok, scorer_mdl, input_text, device)
        output_score = pangram.editlens_score(scorer_tok, scorer_mdl, output_text, device)
        target_emb = sim_model.encode(target_text, convert_to_tensor=True)
        output_emb = sim_model.encode(output_text, convert_to_tensor=True)
        sim_value = float(util.cos_sim(target_emb, output_emb).item())
        scored_rows.append(
            classify_eval_row(
                {
                    "id": row.get("id", f"eval-{idx}"),
                    "source_id": row.get("source_id", row.get("id", f"eval-{idx}")),
                    "split": row.get("split", "val"),
                    "input_text": input_text,
                    "target_text": target_text,
                    "output_text": output_text,
                    "input_score": input_score,
                    "output_score": output_score,
                    "similarity": sim_value,
                },
                mean_similarity_gate=cfg.mean_similarity_gate,
                min_length_ratio=cfg.min_length_ratio,
                max_length_ratio=cfg.max_length_ratio,
            )
        )

    gate_metrics = summarize_gate_metrics(
        similarities=[row["similarity"] for row in scored_rows],
        ratios=[row["length_ratio"] for row in scored_rows],
        invalid_flags=[row["invalid"] for row in scored_rows],
    )
    summary = {
        "rows": len(scored_rows),
        "mean_delta_editlens": statistics.fmean(row["delta_editlens"] for row in scored_rows) if scored_rows else 0.0,
        "mean_output_score": statistics.fmean(row["output_score"] for row in scored_rows) if scored_rows else 1.0,
        **gate_metrics,
    }
    write_json(cfg.output_path, summary)
    write_jsonl(args.out_jsonl or str(Path(cfg.output_path).with_suffix(".jsonl")), scored_rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
