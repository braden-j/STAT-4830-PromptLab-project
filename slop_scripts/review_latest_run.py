#!/usr/bin/env python3
"""Compact review of the latest TinyLlama prompt optimization run.

Prints top 5 prompts with reward, base reward, structural penalty, structural
diagnostics, and one sample output so you can see whether high reward is from
clarity or from formatting artifacts. Does not retrain or run the generator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Review top 5 prompts from latest prompt_opt run")
    p.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory (default: latest under output_dir)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/prompt_opt",
        help="Base output dir when auto-detecting latest run",
    )
    p.add_argument("--top-n", type=int, default=5, help="Number of top prompts to show")
    return p.parse_args()


def find_latest_run(base_dir: Path) -> Path | None:
    """Return the most recent run_* directory under base_dir."""
    if not base_dir.exists():
        return None
    runs = sorted(base_dir.glob("run_*"), key=lambda p: p.name, reverse=True)
    return runs[0] if runs else None


def load_leaderboard(run_dir: Path, top_n: int) -> list[dict]:
    """Load top N entries from leaderboard.jsonl or best_prompts.json."""
    best_path = run_dir / "best_prompts.json"
    leader_path = run_dir / "leaderboard.jsonl"
    if best_path.exists():
        with open(best_path) as f:
            data = json.load(f)
        return data[:top_n] if isinstance(data, list) else [data][:top_n]
    if leader_path.exists():
        rows = []
        with open(leader_path) as f:
            for i, line in enumerate(f):
                if i >= top_n:
                    break
                rows.append(json.loads(line))
        return rows
    return []


def load_sample_outputs(run_dir: Path, leaderboard: list[dict]) -> dict[str, str]:
    """Map full prompt_text -> one sample output using generations.jsonl."""
    gen_path = run_dir / "generations.jsonl"
    out_by_prefix: dict[str, str] = {}
    if not gen_path.exists() or not leaderboard:
        return {}
    with open(gen_path) as f:
        for line in f:
            try:
                gen = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt_prefix = gen.get("prompt_text", "")
            outputs = gen.get("outputs", [])
            if not outputs:
                continue
            sample = outputs[0] if isinstance(outputs[0], str) else ""
            for row in leaderboard:
                full = row["prompt_text"]
                if full in out_by_prefix:
                    continue
                if full[:500] == prompt_prefix or (prompt_prefix and full.startswith(prompt_prefix)):
                    out_by_prefix[full] = sample
                    break
    return out_by_prefix


def main() -> None:
    args = parse_args()
    base = Path(args.output_dir)
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"Run dir not found: {run_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        run_dir = find_latest_run(base)
        if not run_dir:
            print(f"No run_* directories under {base}", file=sys.stderr)
            sys.exit(1)
    print(f"Run: {run_dir}")
    print()

    leaderboard = load_leaderboard(run_dir, args.top_n)
    if not leaderboard:
        print("No leaderboard or best_prompts in run dir.", file=sys.stderr)
        sys.exit(1)

    sample_outputs = load_sample_outputs(run_dir, leaderboard)

    for i, row in enumerate(leaderboard, 1):
        prompt_text = row.get("prompt_text", "")
        avg_reward = row.get("avg_reward", 0.0)
        base_reward = row.get("avg_base_reward", avg_reward)
        struct_penalty = row.get("structural_penalty_contribution", row.get("avg_structural_penalty", 0.0))
        semantic_penalty = row.get("semantic_penalty_contribution", row.get("avg_semantic_penalty", 0.0))
        quality_reward = row.get("quality_reward_contribution", row.get("avg_quality_score", 0.0))
        struct_diag = row.get("structural_diagnostics", {})
        semantic_diag = row.get("semantic_diagnostics", {})
        quality_diag = row.get("quality_diagnostics", {})
        provenance = row.get("provenance", "unknown")
        sample = sample_outputs.get(prompt_text, "(no sample in generations.jsonl)")

        print("=" * 72)
        print(
            f"#{i}  Total: {avg_reward:.4f}  |  Base: {base_reward:.4f}"
            f"  |  Struct: -{struct_penalty:.4f}  |  Semantic: -{semantic_penalty:.4f}  |  Quality: +{quality_reward:.4f}"
            f"  |  Source: {provenance}"
        )
        print("=" * 72)
        print("PROMPT:")
        print(prompt_text[:700] + ("..." if len(prompt_text) > 700 else ""))
        print()
        print("STRUCTURAL (high = list/bullet/formatting):")
        if struct_diag:
            for k, v in sorted(struct_diag.items()):
                bar = "█" * int(round(v * 10)) + "░" * (10 - int(round(v * 10)))
                print(f"  {k}: {v:.3f}  {bar}")
        else:
            print("  (none)")
        print("SEMANTIC (high = meta/advice/off-task; task_relevance high = on-task):")
        if semantic_diag:
            for k, v in sorted(semantic_diag.items()):
                bar = "█" * int(round(v * 10)) + "░" * (10 - int(round(v * 10)))
                print(f"  {k}: {v:.3f}  {bar}")
        else:
            print("  (none)")
        print("QUALITY (higher = more useful; bounded heuristics):")
        if quality_diag:
            for k, v in sorted(quality_diag.items()):
                bar = "█" * int(round(v * 10)) + "░" * (10 - int(round(v * 10)))
                print(f"  {k}: {v:.3f}  {bar}")
        else:
            print("  (none)")
        print()
        print("ONE SAMPLE OUTPUT:")
        print(sample[:500] + ("..." if len(sample) > 500 else ""))
        print()

    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    print("Base = content/clarity. Structural penalty = list/bullet/formatting. Semantic penalty =")
    print("instruction echo, writing advice, prompt copy, off-task. task_relevance_score high = on-task.")
    print("Total reward = base - structural - semantic + quality.")
    print("High total with low semantic penalty, low structural artifacts, and strong quality scores is")
    print("more likely to reflect real clarity rather than meta/format reward hacking.")
    print()


if __name__ == "__main__":
    main()
