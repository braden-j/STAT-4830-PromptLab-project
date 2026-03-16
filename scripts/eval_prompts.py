#!/usr/bin/env python3
"""Compare seed prompts vs optimized prompts from a run directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml
from slop_minimization.scoring import SlopRewardModel, RewardConfig
from slop_minimization.prompt_opt import FrozenGenerator, GeneratorConfig, compare_seed_vs_optimized


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare seed vs optimized prompts on same task")
    p.add_argument("--run-dir", type=str, required=True, help="Run directory (e.g. outputs/prompt_opt/run_20250101_120000)")
    p.add_argument("--config", type=str, default="configs/prompt_opt.yaml", help="Config for reward/generator if not in run dir")
    p.add_argument("--n-samples", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run dir not found: {run_dir}", file=sys.stderr)
        sys.exit(1)
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = run_dir / "config_used.yaml"
    if not config_path.exists():
        print("No config found.", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    task = cfg.get("task_instruction", "Explain the given topic.")
    reward_cfg = cfg.get("reward", {})
    gen_cfg = cfg.get("generator", {})
    if not reward_cfg:
        root = Path(__file__).resolve().parent.parent
        with open(root / "configs" / "prompt_opt.yaml") as f:
            full = yaml.safe_load(f) or {}
            reward_cfg = full.get("reward", {})
            gen_cfg = full.get("generator", {})
    reward_model = SlopRewardModel(RewardConfig(**{k: v for k, v in reward_cfg.items() if k in RewardConfig.__dataclass_fields__}))
    reward_model.load()
    generator = FrozenGenerator(GeneratorConfig(**{k: v for k, v in gen_cfg.items() if k in GeneratorConfig.__dataclass_fields__}))
    generator.load()
    best_path = run_dir / "best_prompts.json"
    if not best_path.exists():
        best_path = run_dir / "leaderboard.jsonl"
        if best_path.exists():
            lines = best_path.read_text().strip().split("\n")
            best = json.loads(lines[0]) if lines else {}
        else:
            print("No best_prompts.json or leaderboard.jsonl in run dir.", file=sys.stderr)
            sys.exit(1)
    else:
        best_list = json.loads(best_path.read_text())
        best = best_list[0] if best_list else {}
    optimized_text = best.get("prompt_text", "")
    if not optimized_text:
        print("No prompt_text in best.", file=sys.stderr)
        sys.exit(1)
    min_length = cfg.get("min_output_length", 20)
    render_mode = cfg.get("render_mode", "structured")
    result = compare_seed_vs_optimized(
        task_instruction=task,
        generator=generator,
        reward_model=reward_model,
        optimized_prompt_text=optimized_text,
        n_samples=args.n_samples,
        min_length=min_length,
        render_mode=render_mode,
    )
    print("Task:", task)
    print(f"Seed mean reward:     {result['seed_mean_reward']:.4f}")
    print(f"Optimized mean reward: {result['optimized_mean_reward']:.4f}")
    delta = result["optimized_mean_reward"] - result["seed_mean_reward"]
    print(f"Delta (optimized - seed): {delta:+.4f}")
    out_path = run_dir / "eval_seed_vs_optimized.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
