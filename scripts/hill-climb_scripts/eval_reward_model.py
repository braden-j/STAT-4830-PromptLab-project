#!/usr/bin/env python3
"""Batch evaluation: score a JSONL dataset with reward model, report stats and optional comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from hill_climb.dataset_io import load_jsonl
from hill_climb.scoring import SlopRewardModel, RewardConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate reward model on a JSONL dataset")
    p.add_argument("--data", "-d", required=True, help="JSONL with 'text' field (and optional 'difficulty')")
    p.add_argument("--checkpoint", "-c", default="outputs/classifier", help="Checkpoint directory")
    p.add_argument("--checkpoint-baseline", type=str, default=None, help="Second checkpoint for comparison")
    p.add_argument("--config", type=str, default=None, help="YAML config for model/reward")
    p.add_argument("--text-column", type=str, default="text")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--show-examples", type=int, default=3, help="Number of top/bottom reward examples to print")
    p.add_argument("--output", "-o", type=str, default=None, help="Write full results JSON here")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data = load_jsonl(args.data)
    if not data:
        print(f"No data in {args.data}")
        return
    if args.max_samples:
        data = data[: args.max_samples]

    texts = []
    difficulties = []
    for ex in data:
        t = ex.get(args.text_column, ex.get("text", ""))
        if isinstance(t, str):
            texts.append(t)
            difficulties.append(ex.get("difficulty", "unknown"))
        else:
            texts.append("")
            difficulties.append("unknown")

    config_dict = {
        "checkpoint_path": args.checkpoint,
        "config_path": args.config,
        "batch_size": args.batch_size,
        "device": args.device,
    }
    config = RewardConfig(**{k: v for k, v in config_dict.items() if k in RewardConfig.__dataclass_fields__})

    def run_one(checkpoint_path: str) -> tuple[list[float], list[float]]:
        config.checkpoint_path = checkpoint_path
        model = SlopRewardModel(config)
        model.load()
        out = model.score_batch(texts, return_token_scores=False, return_diagnostics=False)
        return out["doc_slop_score"], out["reward"]

    print(f"Scoring {len(texts)} examples with checkpoint {args.checkpoint}")
    doc_scores, rewards = run_one(args.checkpoint)
    mean_doc = sum(doc_scores) / len(doc_scores)
    mean_reward = sum(rewards) / len(rewards)
    print(f"  mean doc_slop_score = {mean_doc:.4f}")
    print(f"  mean reward         = {mean_reward:.4f}")

    if any(d != "unknown" for d in difficulties):
        by_diff = {}
        for d, sc, r in zip(difficulties, doc_scores, rewards):
            by_diff.setdefault(d, []).append((sc, r))
        print("  By difficulty:")
        for diff in sorted(by_diff.keys()):
            lst = by_diff[diff]
            m_d = sum(x[0] for x in lst) / len(lst)
            m_r = sum(x[1] for x in lst) / len(lst)
            print(f"    {diff}: n={len(lst)} mean_doc_slop={m_d:.4f} mean_reward={m_r:.4f}")

    if args.checkpoint_baseline:
        print(f"\nBaseline checkpoint {args.checkpoint_baseline}")
        doc_scores_b, rewards_b = run_one(args.checkpoint_baseline)
        mean_doc_b = sum(doc_scores_b) / len(doc_scores_b)
        mean_reward_b = sum(rewards_b) / len(rewards_b)
        print(f"  mean doc_slop_score = {mean_doc_b:.4f}")
        print(f"  mean reward         = {mean_reward_b:.4f}")
        print(f"  Delta reward (main - baseline) = {mean_reward - mean_reward_b:.4f}")

    if args.show_examples and texts:
        indexed = list(zip(rewards, doc_scores, texts, difficulties))
        indexed.sort(key=lambda x: x[0], reverse=True)
        print(f"\nTop {args.show_examples} by reward:")
        for r, d, t, diff in indexed[: args.show_examples]:
            snippet = (t[:80] + "…") if len(t) > 80 else t
            print(f"  reward={r:.4f} doc_slop={d:.4f} [{diff}] {snippet!r}")
        print(f"Bottom {args.show_examples} by reward:")
        for r, d, t, diff in indexed[-args.show_examples :]:
            snippet = (t[:80] + "…") if len(t) > 80 else t
            print(f"  reward={r:.4f} doc_slop={d:.4f} [{diff}] {snippet!r}")

    if args.output:
        out = {
            "n": len(texts),
            "mean_doc_slop_score": mean_doc,
            "mean_reward": mean_reward,
            "doc_slop_scores": doc_scores,
            "rewards": rewards,
            "difficulties": difficulties,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
