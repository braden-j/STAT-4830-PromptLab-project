#!/usr/bin/env python3
"""Compare structure styles (prose_preferred vs mixed vs list_friendly) on the same task: reward, structural diagnostics, examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "slop_src"))

import yaml
from slop.scoring import SlopRewardModel, RewardConfig
from slop.prompt_opt import (
    FrozenGenerator,
    GeneratorConfig,
    get_seeds_for_task,
    evaluate_prompt,
    STRUCTURE_PREFERENCE_VALUES,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare prose_preferred / mixed / list_friendly structure styles (same task, seeds, generator, reward)"
    )
    p.add_argument("--config", type=str, default="slop_configs/prompt_opt.yaml", help="YAML config path")
    p.add_argument("--task", type=str, default=None, help="Task instruction (overrides config)")
    p.add_argument("--n-samples", type=int, default=3, help="Samples per seed per style")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None, help="Write JSON results here")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    reward_cfg = cfg.get("reward", {})
    reward_config = RewardConfig(
        **{k: v for k, v in reward_cfg.items() if k in RewardConfig.__dataclass_fields__}
    )
    reward_model = SlopRewardModel(reward_config)
    reward_model.load()

    gen_cfg = cfg.get("generator", {})
    generator_config = GeneratorConfig(
        **{k: v for k, v in gen_cfg.items() if k in GeneratorConfig.__dataclass_fields__}
    )
    generator = FrozenGenerator(generator_config)
    generator.load()

    task = args.task or cfg.get("default_task", "Explain inflation clearly to a college student.")
    search = cfg.get("search", {})
    render_mode = search.get("render_mode", "simple")
    min_length = search.get("min_output_length", 20)
    lambda_structural = search.get("lambda_structural", 0.15)
    structural_threshold = search.get("structural_threshold", 0.25)

    import random
    rng = random.Random(args.seed)
    base_seeds = get_seeds_for_task(task)
    seeds_per_style = {style: [s.copy() for s in base_seeds[:3]] for style in STRUCTURE_PREFERENCE_VALUES}
    for style in STRUCTURE_PREFERENCE_VALUES:
        for spec in seeds_per_style[style]:
            spec.structure_preference = style

    results = {
        "task_instruction": task,
        "render_mode": render_mode,
        "n_samples": args.n_samples,
        "lambda_structural": lambda_structural,
        "structural_threshold": structural_threshold,
        "styles": {},
    }

    for style in STRUCTURE_PREFERENCE_VALUES:
        seeds = seeds_per_style[style]
        rewards = []
        all_structural = []
        example_outputs = []
        for spec in seeds:
            res = evaluate_prompt(
                spec,
                generator,
                reward_model,
                n_samples=args.n_samples,
                min_length=min_length,
                rng=rng,
                render_mode=render_mode,
                lambda_structural=lambda_structural,
                structural_threshold=structural_threshold,
            )
            rewards.append(res["avg_reward"])
            all_structural.append(res.get("structural_diagnostics", {}))
            if res.get("valid_outputs"):
                example_outputs.append(res["valid_outputs"][0][:400])
            elif res.get("outputs"):
                example_outputs.append(res["outputs"][0][:400] if res["outputs"] else "")
            else:
                example_outputs.append("")
        avg_reward = sum(rewards) / len(rewards) if rewards else -1.0
        # Aggregate structural diagnostics (mean per key)
        struct_agg = {}
        for d in all_structural:
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    struct_agg.setdefault(k, []).append(v)
        struct_summary = {k: sum(v) / len(v) for k, v in struct_agg.items()}
        results["styles"][style] = {
            "avg_reward": avg_reward,
            "per_seed_rewards": rewards,
            "structural_diagnostics": struct_summary,
            "example_outputs": example_outputs,
        }

    print("Task:", task)
    print("Render mode:", render_mode)
    print("n_samples per seed:", args.n_samples)
    print("lambda_structural:", lambda_structural)
    print()
    for style in STRUCTURE_PREFERENCE_VALUES:
        data = results["styles"][style]
        print("=" * 60)
        print(f"Structure style: {style}")
        print("=" * 60)
        print(f"Average reward: {data['avg_reward']:.4f}")
        print("Per-seed rewards:", [f"{r:.4f}" for r in data["per_seed_rewards"]])
        print("Structural diagnostics (mean):", data["structural_diagnostics"])
        print("Example outputs (one per seed):")
        for i, ex in enumerate(data["example_outputs"], 1):
            print(f"  [{i}] {ex[:350]}...")
        print()

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
