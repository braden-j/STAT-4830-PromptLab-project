#!/usr/bin/env python3
"""Evaluate slop classifier and generator."""

import argparse
import json
from dataclasses import fields as dc_fields
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from hill_climb.dataset_io import load_jsonl
from hill_climb.tokenizer_utils import SlopTokenizer
from hill_climb.scoring import compute_reward, aggregate_token_scores


def _load_classifier_from_saved_config(classifier_path: Path, device: torch.device):
    """Load encoder classifier (with optional PEFT) using model_config.json + pytorch_model.bin."""
    from hill_climb.config import ModelConfig
    from hill_climb.models import create_classifier_and_tokenizer
    from transformers import AutoTokenizer

    config_path = classifier_path / "model_config.json"
    state_path = classifier_path / "pytorch_model.bin"
    if not state_path.exists():
        return None, None
    if not config_path.exists():
        # Backward compat: write default encoder config so old checkpoints (e.g. classifier_curriculum) load
        default_config = {
            "backbone_name": "distilbert-base-uncased",
            "model_type": "encoder",
            "num_labels": 2,
            "dropout": 0.1,
            "max_length": 256,
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": ["q_lin", "k_lin", "v_lin"],
        }
        config_path.write_text(json.dumps(default_config, indent=2))
        print(f"Wrote default {config_path} for backward compatibility.")
    raw = json.loads(config_path.read_text())
    raw["backbone_type"] = raw.get("model_type", "encoder")
    raw.pop("model_type", None)
    # ModelConfig fields only
    allowed = {f.name for f in dc_fields(ModelConfig)}
    cfg_dict = {k: v for k, v in raw.items() if k in allowed}
    cfg = ModelConfig(**cfg_dict)

    model, _ = create_classifier_and_tokenizer(cfg)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(classifier_path)
    return model, tokenizer


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
        from transformers import AutoTokenizer

        model, tokenizer = _load_classifier_from_saved_config(classifier_path, device)
        if model is None or tokenizer is None:
            # Fallback: load as causal LM from path (requires config.json with model_type in checkpoint)
            from hill_climb.models.token_classifier import SlopTokenClassifier
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

        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_path}")
    except Exception as e:
        print(f"Eval failed: {e}")
        print("Ensure classifier is trained and compatible.")


if __name__ == "__main__":
    main()
