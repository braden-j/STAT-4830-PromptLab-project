#!/usr/bin/env python3
"""Compare baseline vs curriculum reward checkpoints on a fixed suite of clean / medium / hard slop sentences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "slop_src"))

from slop.scoring import SlopRewardModel, RewardConfig
from slop.scoring.diagnostics import compute_diagnostics


# Hardcoded test suite for checkpoint comparison
CLEAN_SENTENCES = [
    "The report was clear and well structured.",
    "We need to improve efficiency in the process.",
    "The team agreed on the next steps.",
    "The solution addresses the main problem.",
    "The data shows a clear trend.",
]

MEDIUM_SLOP_SENTENCES = [
    "The report was kind of clear and well structured.",
    "We need to basically improve efficiency in the process.",
    "The team sort of agreed on the next steps.",
    "The solution actually addresses the main situation.",
    "The data shows a fairly clear trend.",
]

HARD_SLOP_SENTENCES = [
    "In general the report was relatively clear and well structured.",
    "In practice we need to improve efficiency in the process.",
    "By and large the team agreed on the next steps.",
    "To some extent the solution addresses the main approach.",
    "In many cases the data shows a clear trend.",
]


def get_test_suite() -> list[tuple[str, str]]:
    """Return [(category, text), ...] for all test sentences."""
    out = []
    for s in CLEAN_SENTENCES:
        out.append(("clean", s))
    for s in MEDIUM_SLOP_SENTENCES:
        out.append(("medium_slop", s))
    for s in HARD_SLOP_SENTENCES:
        out.append(("hard_slop", s))
    return out


def word_level_scores_from_subword(
    text: str,
    tokenizer,
    subword_scores: list[float],
) -> list[tuple[str, float]]:
    """Map subword slop scores back to words: [(word, score), ...]. Uses mean of subword scores per word."""
    if not subword_scores or not text.strip():
        return []
    try:
        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            truncation=True,
            max_length=512,
        )
    except Exception:
        return []
    offset_mapping = enc.get("offset_mapping", [])
    special = enc.get("special_tokens_mask", [0] * len(offset_mapping))
    words = text.split()
    if not words or not offset_mapping:
        return [(w, 0.0) for w in words] if words else []

    # Character index -> word index (by scanning text)
    char_to_word = [-1] * (len(text) + 1)
    pos = 0
    for wi, w in enumerate(words):
        for _ in range(len(w)):
            if pos <= len(text):
                char_to_word[pos] = wi
            pos += 1
        while pos < len(text) and text[pos].isspace():
            pos += 1

    result = [[] for _ in words]
    for i, (start, end) in enumerate(offset_mapping):
        if i >= len(subword_scores) or special[i] == 1 or (start == 0 and end == 0):
            continue
        score = subword_scores[i]
        if start < len(char_to_word) and char_to_word[start] >= 0:
            result[char_to_word[start]].append(score)
    out = []
    for i, w in enumerate(words):
        if result[i]:
            out.append((w, sum(result[i]) / len(result[i])))
        else:
            out.append((w, 0.0))
    return out


def highlight_line(pairs: list[tuple[str, float]], threshold: float = 0.5) -> str:
    """Format word(score) with optional mark for high slop."""
    parts = []
    for w, s in pairs:
        if s >= threshold:
            parts.append(f"**{w}**({s:.2f})")
        else:
            parts.append(f"{w}({s:.2f})")
    return " ".join(parts)


def run_comparison(
    baseline_path: str,
    curriculum_path: str,
    config_path: str | None,
    device: str | None,
    token_highlight: bool,
    output_path: str | None,
) -> dict:
    base_dir = Path(baseline_path)
    curr_dir = Path(curriculum_path)
    if not base_dir.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_path}. Train with train_token_classifier.py first.")
    if not curr_dir.exists():
        raise FileNotFoundError(f"Curriculum checkpoint not found: {curriculum_path}. Train a curriculum model first.")

    suite = get_test_suite()
    texts = [t for _, t in suite]
    categories = [c for c, _ in suite]

    def load_model(checkpoint: str) -> SlopRewardModel:
        cfg = RewardConfig(
            checkpoint_path=checkpoint,
            config_path=config_path,
            device=device,
        )
        return SlopRewardModel(cfg)

    print("Loading baseline checkpoint:", baseline_path)
    baseline = load_model(baseline_path)
    baseline.load()
    print("Loading curriculum checkpoint:", curriculum_path)
    curriculum = load_model(curriculum_path)
    curriculum.load()

    print("\nScoring test suite...")
    base_out = baseline.score_batch(texts, return_token_scores=token_highlight, return_diagnostics=True)
    curr_out = curriculum.score_batch(texts, return_token_scores=token_highlight, return_diagnostics=True)

    results = {
        "baseline_checkpoint": baseline_path,
        "curriculum_checkpoint": curriculum_path,
        "examples": [],
    }

    for i, (cat, text) in enumerate(suite):
        base_doc = base_out["doc_slop_score"][i]
        base_rew = base_out["reward"][i]
        curr_doc = curr_out["doc_slop_score"][i]
        curr_rew = curr_out["reward"][i]
        base_diag = base_out["diagnostics"][i] if base_out.get("diagnostics") else None
        curr_diag = curr_out["diagnostics"][i] if curr_out.get("diagnostics") else None

        ex = {
            "category": cat,
            "text": text,
            "baseline": {"doc_slop_score": base_doc, "reward": base_rew},
            "curriculum": {"doc_slop_score": curr_doc, "reward": curr_rew},
        }
        if base_diag:
            ex["baseline"]["diagnostics"] = base_diag
        if curr_diag:
            ex["curriculum"]["diagnostics"] = curr_diag
        results["examples"].append(ex)

        # Print
        print("\n" + "=" * 60)
        print(f"[{cat}] {text[:70]}{'…' if len(text) > 70 else ''}")
        print("-" * 40)
        print(f"  Baseline:   doc_slop_score={base_doc:.4f}  reward={base_rew:.4f}")
        if base_diag:
            print(f"             diagnostics: very_short={base_diag['very_short']} "
                  f"rep_ratio={base_diag['repetition_ratio']:.3f} "
                  f"punct_ratio={base_diag['punctuation_ratio']:.3f} "
                  f"caps_ratio={base_diag['caps_ratio']:.3f} "
                  f"filler_loop={base_diag['filler_loop_score']:.3f}")
        print(f"  Curriculum: doc_slop_score={curr_doc:.4f}  reward={curr_rew:.4f}")
        if curr_diag:
            print(f"             diagnostics: very_short={curr_diag['very_short']} "
                  f"rep_ratio={curr_diag['repetition_ratio']:.3f} "
                  f"punct_ratio={curr_diag['punctuation_ratio']:.3f} "
                  f"caps_ratio={curr_diag['caps_ratio']:.3f} "
                  f"filler_loop={curr_diag['filler_loop_score']:.3f}")

        if token_highlight and base_out.get("token_scores") and curr_out.get("token_scores"):
            base_scores = base_out["token_scores"][i]
            curr_scores = curr_out["token_scores"][i]
            if base_scores and curr_scores:
                try:
                    base_pairs = word_level_scores_from_subword(text, baseline.tokenizer, base_scores)
                    curr_pairs = word_level_scores_from_subword(text, curriculum.tokenizer, curr_scores)
                    print("  Token-level (slop prob per word, ** = high):")
                    print("    Baseline:   ", highlight_line(base_pairs))
                    print("    Curriculum: ", highlight_line(curr_pairs))
                except Exception as e:
                    print("  Token-level: (skip)", e)
                ex["baseline"]["token_scores"] = base_out["token_scores"][i]
                ex["curriculum"]["token_scores"] = curr_out["token_scores"][i]

    # Summary
    base_docs = [e["baseline"]["doc_slop_score"] for e in results["examples"]]
    base_rews = [e["baseline"]["reward"] for e in results["examples"]]
    curr_docs = [e["curriculum"]["doc_slop_score"] for e in results["examples"]]
    curr_rews = [e["curriculum"]["reward"] for e in results["examples"]]
    by_cat = {}
    for ex in results["examples"]:
        c = ex["category"]
        by_cat.setdefault(c, []).append(ex)
    results["summary"] = {
        "baseline_mean_doc_slop_score": sum(base_docs) / len(base_docs),
        "baseline_mean_reward": sum(base_rews) / len(base_rews),
        "curriculum_mean_doc_slop_score": sum(curr_docs) / len(curr_docs),
        "curriculum_mean_reward": sum(curr_rews) / len(curr_rews),
        "by_category": {
            c: {
                "baseline_mean_doc_slop_score": sum(e["baseline"]["doc_slop_score"] for e in exs) / len(exs),
                "baseline_mean_reward": sum(e["baseline"]["reward"] for e in exs) / len(exs),
                "curriculum_mean_doc_slop_score": sum(e["curriculum"]["doc_slop_score"] for e in exs) / len(exs),
                "curriculum_mean_reward": sum(e["curriculum"]["reward"] for e in exs) / len(exs),
            }
            for c, exs in by_cat.items()
        },
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("-" * 40)
    print(f"  Baseline   mean doc_slop_score={results['summary']['baseline_mean_doc_slop_score']:.4f}  mean reward={results['summary']['baseline_mean_reward']:.4f}")
    print(f"  Curriculum mean doc_slop_score={results['summary']['curriculum_mean_doc_slop_score']:.4f}  mean reward={results['summary']['curriculum_mean_reward']:.4f}")
    print("  By category:")
    for c, s in results["summary"]["by_category"].items():
        print(f"    {c}: baseline reward={s['baseline_mean_reward']:.4f}  curriculum reward={s['curriculum_mean_reward']:.4f}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Serialize diagnostics (they contain floats)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare baseline vs curriculum reward checkpoints on fixed test suite")
    p.add_argument("--baseline", "-b", default="outputs/classifier", help="Baseline checkpoint directory")
    p.add_argument("--curriculum", "-C", default="outputs/classifier_curriculum", help="Curriculum checkpoint directory")
    p.add_argument("--config", type=str, default=None, help="YAML config for model (optional)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--token-highlight", action="store_true", help="Print token-level slop scores side by side")
    p.add_argument("--output", "-o", type=str, default=None, help="Save results JSON here")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_comparison(
        baseline_path=args.baseline,
        curriculum_path=args.curriculum,
        config_path=args.config,
        device=args.device,
        token_highlight=args.token_highlight,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
