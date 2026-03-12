#!/usr/bin/env python3
"""Slop generator: (1) Generate data/slop_pairs.jsonl via rule-based sloppifier, (2) Train T5 rewriter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from slop_minimization.data.dataset import load_jsonl
from slop_minimization.slop_gen import RuleSloppifier
from slop_minimization.slop_gen.train_rewriter import train_rewriter


def load_good_text(paths: list[str], text_key: str = "text", label_key: str = "labels", min_len: int = 20) -> list[str]:
    """Load lines of good (non-slop) text from JSONL or plain text."""
    out = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        if p.suffix == ".jsonl":
            for item in load_jsonl(p):
                t = item.get(text_key) or item.get("human", "")
                if isinstance(t, str) and len(t.strip()) >= min_len:
                    labels = item.get(label_key, [])
                    if not labels or (isinstance(labels[0], (int, float)) and labels[0] == 0):
                        out.append(t.strip())
                elif isinstance(t, str) and t.strip():
                    out.append(t.strip())
        else:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if len(line) >= min_len:
                        out.append(line)
    return out


def generate_slop_pairs(
    input_paths: list[str],
    output_path: str,
    num_per_text: int = 2,
    seed: int = 42,
    **sloppifier_kw,
) -> None:
    """Generate (human, slop) pairs using rule-based sloppifier and write JSONL."""
    texts = load_good_text(input_paths)
    if not texts:
        raise FileNotFoundError(
            f"No good text found in {input_paths}. Use data/train.jsonl or data/raw/good.txt."
        )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sloppifier = RuleSloppifier(seed=seed, **sloppifier_kw)
    pairs = []
    for i, human in enumerate(texts):
        for _ in range(num_per_text):
            slop = sloppifier.sloppify(human)
            pairs.append({"human": human, "slop": slop})
    with open(out_path, "w") as f:
        for rec in pairs:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(pairs)} pairs to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Slop generator: generate pairs and/or train T5 rewriter")
    sub = p.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Generate data/slop_pairs.jsonl with rule-based sloppifier")
    gen.add_argument("--input", nargs="+", default=["data/train.jsonl", "data/raw/good.txt"])
    gen.add_argument("--output", default="data/slop_pairs.jsonl")
    gen.add_argument("--num-per-text", type=int, default=2)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--filler-prob", type=float, default=0.25)
    gen.add_argument("--hedge-prob", type=float, default=0.2)
    gen.add_argument("--repeat-prob", type=float, default=0.15)
    gen.add_argument("--generic-noun-prob", type=float, default=0.3)
    gen.add_argument("--template-prob", type=float, default=0.2)
    train = sub.add_parser("train", help="Train T5 rewriter on data/slop_pairs.jsonl")
    train.add_argument("--train-path", default="data/slop_pairs.jsonl")
    train.add_argument("--val-path", default=None)
    train.add_argument("--output-dir", default="outputs/slop_rewriter")
    train.add_argument("--model-name", default="t5-small")
    train.add_argument("--max-source-length", type=int, default=256)
    train.add_argument("--max-target-length", type=int, default=256)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--lr", type=float, default=5e-5)
    train.add_argument("--warmup-ratio", type=float, default=0.1)
    train.add_argument("--fp16", action="store_true", default=True)
    train.add_argument("--no-fp16", action="store_false", dest="fp16")
    train.add_argument("--use-wandb", action="store_true")
    train.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        generate_slop_pairs(
            input_paths=args.input,
            output_path=args.output,
            num_per_text=args.num_per_text,
            seed=args.seed,
            filler_prob=args.filler_prob,
            hedge_prob=args.hedge_prob,
            repeat_sentence_prob=args.repeat_prob,
            generic_noun_prob=args.generic_noun_prob,
            template_prob=args.template_prob,
        )
    elif args.command == "train":
        train_rewriter(
            train_path=args.train_path,
            val_path=args.val_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_ratio=args.warmup_ratio,
            fp16=args.fp16,
            use_wandb=args.use_wandb,
            seed=args.seed,
        )
        print(f"Rewriter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
