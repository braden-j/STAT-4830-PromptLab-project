#!/usr/bin/env python3
"""Run prompt optimization: hill climbing to minimize slop using the reward model."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml
from slop_minimization.scoring import SlopRewardModel, RewardConfig
from slop_minimization.prompt_opt import (
    FrozenGenerator,
    GeneratorConfig,
    run_hill_climbing,
    HillClimbConfig,
    get_seeds_for_task,
)
from slop_minimization.prompt_opt.templates import PromptSpec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimize prompts via hill climbing to minimize slop")
    p.add_argument("--config", type=str, default="configs/prompt_opt.yaml", help="YAML config path")
    p.add_argument("--task", type=str, default=None, help="Task/topic instruction (overrides config)")
    p.add_argument("--reward-checkpoint", type=str, default=None, help="Reward model checkpoint (overrides config)")
    p.add_argument("--reward-config", type=str, default=None, help="Reward model config YAML (overrides config)")
    p.add_argument("--generator-model", type=str, default=None, help="Generator model name (overrides config)")
    p.add_argument("--output-dir", type=str, default=None, help="Base output dir (default: outputs/prompt_opt)")
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--population-size", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--samples-per-prompt", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
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
    if args.reward_checkpoint:
        reward_cfg["checkpoint_path"] = args.reward_checkpoint
    if args.reward_config:
        reward_cfg["config_path"] = args.reward_config
    reward_config = RewardConfig(**{k: v for k, v in reward_cfg.items() if k in RewardConfig.__dataclass_fields__})
    reward_model = SlopRewardModel(reward_config)
    reward_model.load()

    gen_cfg = cfg.get("generator", {})
    if args.generator_model:
        gen_cfg["model_name"] = args.generator_model
    generator_config = GeneratorConfig(**{k: v for k, v in gen_cfg.items() if k in GeneratorConfig.__dataclass_fields__})
    generator = FrozenGenerator(generator_config)
    generator.load()

    task = args.task or cfg.get("default_task", "Explain the given topic in 2-3 short paragraphs.")
    search_cfg = cfg.get("search", {})
    if args.iterations is not None:
        search_cfg["num_iterations"] = args.iterations
    if args.population_size is not None:
        search_cfg["population_size"] = args.population_size
    if args.top_k is not None:
        search_cfg["top_k"] = args.top_k
    if args.samples_per_prompt is not None:
        search_cfg["samples_per_prompt"] = args.samples_per_prompt
    if args.seed is not None:
        search_cfg["random_seed"] = args.seed
    hill_config = HillClimbConfig(**{k: v for k, v in search_cfg.items() if k in HillClimbConfig.__dataclass_fields__})

    base_out = args.output_dir or cfg.get("output_dir", "outputs/prompt_opt")
    run_dir = Path(base_out) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("Task:", task)
    print("Reward checkpoint:", reward_config.checkpoint_path)
    print("Generator:", generator_config.model_name)
    print("Output dir:", run_dir)
    print("Running hill climbing...")
    result = run_hill_climbing(
        task_instruction=task,
        generator=generator,
        reward_model=reward_model,
        config=hill_config,
        output_dir=run_dir,
    )
    print("\nTop 3 prompts by reward:")
    for i, row in enumerate(result["leaderboard"][:3], 1):
        print(f"\n--- {i}. avg_reward={row['avg_reward']:.4f} ---")
        print(row["prompt_text"][:600])
    print(f"\nBest reward: {result['best_avg_reward']:.4f}")
    print(f"Results saved to {result['output_dir']}")
    if result.get("invalid_count"):
        print(f"Invalid (too short) generations: {result['invalid_count']}")


if __name__ == "__main__":
    main()
