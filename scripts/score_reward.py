#!/usr/bin/env python3
"""CLI: load reward model checkpoint and score raw text or a text file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from slop_minimization.scoring import SlopRewardModel, RewardConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score text with slop reward model")
    p.add_argument("--checkpoint", "-c", default="outputs/classifier", help="Checkpoint directory")
    p.add_argument("--config", type=str, default=None, help="YAML with reward/model config (optional)")
    p.add_argument("--text", "-t", type=str, default=None, help="Raw input text")
    p.add_argument("--input", "-i", type=str, default=None, help="Input text file (one example per line or single blob)")
    p.add_argument("--output", "-o", type=str, default=None, help="Write JSON results here")
    p.add_argument("--token-scores", action="store_true", help="Include per-token scores in output")
    p.add_argument("--diagnostics", action="store_true", help="Include anti-hacking diagnostics")
    p.add_argument("--aggregation", choices=["mean", "max", "topk"], default="mean")
    p.add_argument("--topk-fraction", type=float, default=0.1)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    texts = []
    if args.text:
        texts.append(args.text)
    if args.input:
        path = Path(args.input)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            sys.exit(1)
        with open(path) as f:
            content = f.read()
        if args.text:
            texts.append(content)
        else:
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            texts = lines if lines else [content]

    if not texts:
        print("Provide --text or --input.", file=sys.stderr)
        sys.exit(1)

    config_dict = {
        "checkpoint_path": args.checkpoint,
        "config_path": args.config,
        "device": args.device,
        "batch_size": args.batch_size,
        "aggregation_mode": args.aggregation,
        "topk_fraction": args.topk_fraction,
    }
    if args.config and Path(args.config).exists():
        import yaml
        with open(args.config) as f:
            full = yaml.safe_load(f) or {}
        reward_section = full.get("reward")
        if isinstance(reward_section, dict):
            config_dict.update({k: v for k, v in reward_section.items() if k in RewardConfig.__dataclass_fields__})

    config = RewardConfig(**{k: v for k, v in config_dict.items() if k in RewardConfig.__dataclass_fields__})
    model = SlopRewardModel(config)
    model.load()

    results = model.score_batch(
        texts,
        return_token_scores=args.token_scores,
        return_diagnostics=args.diagnostics,
    )

    for i, text in enumerate(texts):
        doc_score = results["doc_slop_score"][i]
        reward = results["reward"][i]
        print(f"doc_slop_score={doc_score:.4f} reward={reward:.4f}")
        if args.diagnostics and results.get("diagnostics"):
            d = results["diagnostics"][i]
            print(f"  diagnostics: very_short={d['very_short']} rep_ratio={d['repetition_ratio']:.3f} punct_ratio={d['punctuation_ratio']:.3f}")

    if args.output:
        out = {
            "doc_slop_scores": results["doc_slop_score"],
            "rewards": results["reward"],
        }
        if results.get("token_scores"):
            out["token_scores"] = results["token_scores"]
        if results.get("diagnostics"):
            out["diagnostics"] = results["diagnostics"]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
