#!/usr/bin/env python3
"""Compare the same task and seed prompts across multiple generator models (same reward model and render mode)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import yaml
from hill_climb.scoring import SlopRewardModel, RewardConfig
from hill_climb.prompt_opt import (
    FrozenGenerator,
    GeneratorConfig,
    compare_generators,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare seed prompts across multiple generator models (same reward model and render mode)"
    )
    p.add_argument("--config", type=str, default="configs/hill-climb_configs/prompt_opt.yaml", help="YAML config path")
    p.add_argument("--task", type=str, default=None, help="Task instruction (overrides config)")
    p.add_argument(
        "--generators",
        type=str,
        default="gpt2",
        help="Comma-separated model names, e.g. gpt2,TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )
    p.add_argument("--n-samples", type=int, default=3, help="Samples per seed per generator")
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

    base_gen_cfg = cfg.get("generator", {})
    model_names = [s.strip() for s in args.generators.split(",") if s.strip()]
    if not model_names:
        print("At least one generator model name required (e.g. --generators gpt2)", file=sys.stderr)
        sys.exit(1)

    generators: list[tuple[str, FrozenGenerator]] = []
    for name in model_names:
        gen_dict = {**base_gen_cfg, "model_name": name}
        gen_config = GeneratorConfig(
            **{k: v for k, v in gen_dict.items() if k in GeneratorConfig.__dataclass_fields__}
        )
        gen = FrozenGenerator(gen_config)
        gen.load()
        generators.append((name, gen))

    task = args.task or cfg.get("default_task", "Explain inflation clearly to a college student.")
    render_mode = cfg.get("search", {}).get("render_mode", "simple")
    min_length = cfg.get("search", {}).get("min_output_length", 20)
    import random
    rng = random.Random(args.seed)

    result = compare_generators(
        task_instruction=task,
        generators=generators,
        reward_model=reward_model,
        render_mode=render_mode,
        n_samples=args.n_samples,
        min_length=min_length,
        rng=rng,
    )

    print("Task:", task)
    print("Render mode:", render_mode)
    print("n_samples per seed:", args.n_samples)
    print()
    for label, data in result["generators"].items():
        print("=" * 60)
        print(f"Generator: {label}")
        print("=" * 60)
        print(f"Average reward: {data['avg_reward']:.4f}")
        print("Per-seed rewards:", [f"{r:.4f}" for r in data["per_seed_rewards"]])
        print("Example outputs (one per seed):")
        for i, ex in enumerate(data["example_outputs"], 1):
            print(f"  [{i}] {ex[:350]}...")
        print()

    if args.output:
        out_path = Path(args.output)
        out_data = {
            "task_instruction": result["task_instruction"],
            "render_mode": result["render_mode"],
            "n_samples": result["n_samples"],
            "generators": {
                label: {
                    "avg_reward": d["avg_reward"],
                    "per_seed_rewards": d["per_seed_rewards"],
                    "example_outputs": d["example_outputs"],
                }
                for label, d in result["generators"].items()
            },
        }
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
