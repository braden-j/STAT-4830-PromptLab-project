#!/usr/bin/env python3
"""Compare structured vs simple vs compact prompt rendering on the same seeds: reward and example outputs."""

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
    compare_rendering_modes,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare seed prompts under structured / simple / compact rendering"
    )
    p.add_argument("--config", type=str, default="configs/hill-climb_configs/prompt_opt.yaml", help="YAML config path")
    p.add_argument("--task", type=str, default=None, help="Task instruction (overrides config)")
    p.add_argument("--n-samples", type=int, default=3, help="Samples per seed per mode")
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
    import random
    rng = random.Random(args.seed)

    result = compare_rendering_modes(
        task_instruction=task,
        generator=generator,
        reward_model=reward_model,
        n_samples=args.n_samples,
        min_length=cfg.get("search", {}).get("min_output_length", 20),
        rng=rng,
    )

    print("Task:", task)
    print("n_samples per seed:", args.n_samples)
    print()
    for mode, data in result["modes"].items():
        print("=" * 60)
        print(f"Mode: {mode}")
        print("=" * 60)
        print(f"Average reward: {data['avg_reward']:.4f}")
        print("Per-seed rewards:", [f"{r:.4f}" for r in data["per_seed_rewards"]])
        print("\nExample prompt (first 300 chars):")
        print(data["example_prompts"][0] if data["example_prompts"] else "(none)")
        print("\nExample outputs (one per seed):")
        for i, ex in enumerate(data["example_outputs"], 1):
            print(f"  [{i}] {ex[:350]}...")
        print()

    if args.output:
        out_path = Path(args.output)
        # Make result JSON-serializable (no RNG)
        out_data = {
            "task_instruction": result["task_instruction"],
            "n_samples": result["n_samples"],
            "modes": {
                mode: {
                    "avg_reward": d["avg_reward"],
                    "per_seed_rewards": d["per_seed_rewards"],
                    "example_prompts": d["example_prompts"],
                    "example_outputs": d["example_outputs"],
                }
                for mode, d in result["modes"].items()
            },
        }
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
