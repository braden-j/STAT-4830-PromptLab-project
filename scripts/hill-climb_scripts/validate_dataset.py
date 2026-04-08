#!/usr/bin/env python3
"""Quick validation of token-level slop dataset: sample examples and check alignment."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "slop_src"))

from slop.dataset_io import load_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect train/val JSONL and verify label alignment")
    p.add_argument("--train", default="data/train.jsonl", help="Train JSONL path")
    p.add_argument("--val", default="data/val.jsonl", help="Val JSONL path")
    p.add_argument("--sample", type=int, default=10, help="Number of random examples to print per split")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_path = Path(args.train)
    val_path = Path(args.val)
    if not train_path.exists():
        print(f"Missing {train_path}", file=sys.stderr)
        sys.exit(1)
    if not val_path.exists():
        print(f"Missing {val_path}", file=sys.stderr)
        sys.exit(1)

    train_data = load_jsonl(train_path)
    val_data = load_jsonl(val_path)
    rng = random.Random(args.seed)

    def check_and_report(name: str, data: list[dict]) -> None:
        if not data:
            print(f"\n{name}: no examples")
            return
        n = len(data)
        has_slop = sum(1 for ex in data if any(ex.get("labels", [])))
        print(f"\n{name}: {n} examples, {has_slop} with ≥1 slop token ({100 * has_slop / n:.1f}%)")

        sample_size = min(args.sample, n)
        indices = rng.sample(range(n), sample_size)
        misaligned = 0
        for i, idx in enumerate(indices):
            ex = data[idx]
            text = ex.get("text", "")
            labels = ex.get("labels", [])
            words = text.split()
            ok = len(labels) == len(words)
            if not ok:
                misaligned += 1
            print(f"\n  --- Example {i + 1} (index {idx}) ---")
            print(f"  text: {text}")
            print(f"  labels: {labels}")
            print(f"  len(labels)==len(text.split()): {len(labels)} == {len(words)} -> {'OK' if ok else 'MISMATCH'}")
        if misaligned:
            print(f"\n  MISALIGNED: {misaligned}/{sample_size} sampled examples")
        else:
            print(f"\n  All {sample_size} sampled examples aligned.")

    check_and_report("Train", train_data)
    check_and_report("Val", val_data)
    print()


if __name__ == "__main__":
    main()
