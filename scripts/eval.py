#!/usr/bin/env python3
"""Evaluate slop classifier and generator."""

import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from slop_minimization.data.dataset import load_jsonl
from slop_minimization.data.tokenizer import SlopTokenizer
from slop_minimization.scoring import compute_reward, aggregate_token_scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--classifier-path", type=str, default="outputs/classifier")
    p.add_argument("--test-path", type=str, default="data/test.jsonl")
    p.add_argument("--output-path", type=str, default="outputs/eval_results.json")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    test_data = load_jsonl(args.test_path)
    if not test_data:
        print(f"No test data at {args.test_path}. Run build_data.py first.")
        return

    classifier_path = Path(args.classifier_path)
    if not classifier_path.exists():
        print(f"Classifier not found at {classifier_path}. Skipping eval.")
        return

    try:
        from slop_minimization.models.token_classifier import SlopTokenClassifier
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(classifier_path)
        model = SlopTokenClassifier(
            backbone_name=str(classifier_path),
            num_labels=2,
        ).to(device)
        model.eval()

        slop_tok = SlopTokenizer(tokenizer, max_length=256)

        texts = [ex["text"] for ex in test_data]
        labels = [ex.get("labels", [0] * len(t.split())) for t, ex in zip(texts, test_data)]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            probs = model.score_tokens(inputs["input_ids"], inputs["attention_mask"])
        rewards = compute_reward(model, inputs["input_ids"], inputs["attention_mask"])
        mean_reward = rewards.mean().item()

        # Simple accuracy if we have labels
        correct = 0
        total = 0
        for i, lbls in enumerate(labels):
            preds = (probs[i].mean().item() > 0.5)
            gt = 1 if (sum(lbls) / max(len(lbls), 1) > 0.5) else 0
            if len(lbls) > 0:
                correct += int(preds == gt)
                total += 1
        acc = correct / total if total else 0

        results = {
            "mean_reward": mean_reward,
            "sequence_accuracy": acc,
            "n_samples": len(test_data),
        }
        print(results)

        import json
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_path}")
    except Exception as e:
        print(f"Eval failed: {e}")
        print("Ensure classifier is trained and compatible.")


if __name__ == "__main__":
    main()
