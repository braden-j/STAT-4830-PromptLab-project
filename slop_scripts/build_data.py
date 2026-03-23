#!/usr/bin/env python3
"""Build training data for slop classifier and generator."""

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build slop minimization datasets")
    p.add_argument("--good-path", type=str, default="data/raw/good.txt", help="Path to good text samples")
    p.add_argument("--slop-path", type=str, default="data/raw/slop.txt", help="Path to slop text samples")
    p.add_argument("--output-dir", type=str, default="data", help="Output directory")
    p.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Val split ratio")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--min-len", type=int, default=10, help="Min chars per sample")
    p.add_argument("--max-samples", type=int, default=None, help="Max samples per split")
    return p.parse_args()


def load_lines(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        return [ln.strip() for ln in f if ln.strip()]


def create_token_labels(text: str, is_slop: bool) -> list[int]:
    """Create per-token labels: 0 = good, 1 = slop. Word-level granularity."""
    words = text.split()
    label = 1 if is_slop else 0
    return [label] * len(words)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    good = load_lines(args.good_path)
    slop = load_lines(args.slop_path)

    if not good and not slop:
        # Create placeholder data for skeleton
        good = ["This is a clear and well-written sentence with proper structure."] * 10
        slop = ["Yo so basically like um you know it's kinda like that thing right."] * 10

    samples = []
    for t in good:
        if len(t) >= args.min_len:
            samples.append({"text": t, "labels": create_token_labels(t, is_slop=False)})
    for t in slop:
        if len(t) >= args.min_len:
            samples.append({"text": t, "labels": create_token_labels(t, is_slop=True)})

    random.shuffle(samples)
    n = len(samples)
    tr_end = int(n * args.train_ratio)
    val_end = tr_end + int(n * args.val_ratio)

    train_data = samples[:tr_end]
    val_data = samples[tr_end:val_end]
    test_data = samples[val_end:]

    if args.max_samples:
        train_data = train_data[: args.max_samples]
        val_data = val_data[: max(1, args.max_samples // 10)]
        test_data = test_data[: max(1, args.max_samples // 10)]

    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        out_path = output_dir / f"{name}.jsonl"
        with open(out_path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {len(data)} examples to {out_path}")


if __name__ == "__main__":
    main()
