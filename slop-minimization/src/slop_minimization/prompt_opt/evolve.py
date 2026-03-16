"""Hill climbing / evolutionary search over PromptSpecs using reward model."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .templates import PromptSpec, render_prompt, get_seeds_for_task, prompt_spec_to_dict, dict_to_prompt_spec, RENDER_MODES
from .mutations import mutate_spec


def evaluate_prompt(
    prompt_spec: PromptSpec,
    generator: Any,
    reward_model: Any,
    n_samples: int = 2,
    min_length: int = 20,
    rng: Any = None,
    render_mode: str = "structured",
) -> dict[str, Any]:
    """Generate n_samples from the prompt, score with reward model, return averaged metrics and diagnostics.

    Outputs below min_length (word count) are rejected and not scored; they are counted in invalid_count.
    """
    import random
    rng = rng or random.Random(42)
    if render_mode not in RENDER_MODES:
        render_mode = "structured"
    prompt_text = render_prompt(prompt_spec, mode=render_mode)
    # Generate multiple times (no batching across samples to keep interface simple)
    outputs: list[str] = []
    for _ in range(n_samples):
        out = generator.generate_one(prompt_text)
        outputs.append(out)

    # Anti-degenerate: reject too short
    valid_outputs = [t for t in outputs if len(t.split()) >= min_length]
    invalid_count = len(outputs) - len(valid_outputs)

    if not valid_outputs:
        return {
            "prompt_spec": prompt_spec_to_dict(prompt_spec),
            "prompt_text": prompt_text,
            "outputs": outputs,
            "valid_outputs": [],
            "invalid_count": invalid_count,
            "n_samples": n_samples,
            "avg_reward": -1.0,
            "avg_doc_slop_score": 1.0,
            "diagnostics_summary": {},
            "error": "all outputs below min_length",
        }

    reward_model.load()
    result = reward_model.score_batch(valid_outputs, return_diagnostics=True)
    rewards = result["reward"]
    doc_scores = result["doc_slop_score"]
    diagnostics = result.get("diagnostics") or []

    avg_reward = sum(rewards) / len(rewards)
    avg_doc_slop_score = sum(doc_scores) / len(doc_scores)
    # Summarize diagnostics (e.g. mean of key fields)
    diag_summary: dict[str, float] = {}
    if diagnostics:
        keys = ["repetition_ratio", "punctuation_ratio", "caps_ratio", "filler_loop_score"]
        for k in keys:
            if k in diagnostics[0]:
                vals = [d[k] for d in diagnostics if isinstance(d.get(k), (int, float))]
                if vals:
                    diag_summary[k] = sum(vals) / len(vals)

    return {
        "prompt_spec": prompt_spec_to_dict(prompt_spec),
        "prompt_text": prompt_text,
        "outputs": outputs,
        "valid_outputs": valid_outputs,
        "invalid_count": invalid_count,
        "n_samples": n_samples,
        "avg_reward": avg_reward,
        "avg_doc_slop_score": avg_doc_slop_score,
        "diagnostics_summary": diag_summary,
        "error": None,
    }


@dataclass
class HillClimbConfig:
    population_size: int = 12
    top_k: int = 4
    children_per_parent: int = 2
    num_iterations: int = 10
    samples_per_prompt: int = 2
    mutation_strength: str = "medium"
    random_seed: int = 42
    min_output_length: int = 20
    keep_random_explore: int = 1  # number of random mutants to add each iteration
    render_mode: str = "structured"  # structured | simple | compact


def run_hill_climbing(
    task_instruction: str,
    generator: Any,
    reward_model: Any,
    config: HillClimbConfig | dict | None = None,
    output_dir: Path | str | None = None,
    seed_specs: list[PromptSpec] | None = None,
) -> dict[str, Any]:
    """Run hill climbing over PromptSpecs. Returns summary and saves leaderboard, best_prompts, generations, config."""
    import random
    import json
    import yaml

    if config is None:
        config = HillClimbConfig()
    elif isinstance(config, dict):
        config = HillClimbConfig(**{k: v for k, v in config.items() if k in HillClimbConfig.__dataclass_fields__})
    rng = random.Random(config.random_seed)

    seeds = seed_specs or get_seeds_for_task(task_instruction)
    population: list[PromptSpec] = []
    # Start with seeds; if we need more to fill population_size, duplicate randomly
    while len(population) < config.population_size:
        population.append(rng.choice(seeds).copy())
    population = population[: config.population_size]

    all_candidates: list[dict[str, Any]] = []
    invalid_generations: list[dict[str, Any]] = []
    iteration_found: dict[str, int] = {}  # prompt_text -> first iteration where seen
    best_initial_prompt_text: str = ""
    best_initial_reward: float = -1.0

    out_path = None
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "config_used.yaml", "w") as f:
            cfg_dict = {
                "task_instruction": task_instruction,
                "population_size": config.population_size,
                "top_k": config.top_k,
                "children_per_parent": config.children_per_parent,
                "num_iterations": config.num_iterations,
                "samples_per_prompt": config.samples_per_prompt,
                "mutation_strength": config.mutation_strength,
                "random_seed": config.random_seed,
                "min_output_length": config.min_output_length,
                "render_mode": config.render_mode,
            }
            yaml.dump(cfg_dict, f, default_flow_style=False)

    for iteration in range(config.num_iterations):
        # Evaluate current population
        scores: list[tuple[float, dict[str, Any]]] = []
        for spec in population:
            res = evaluate_prompt(
                spec,
                generator,
                reward_model,
                n_samples=config.samples_per_prompt,
                min_length=config.min_output_length,
                rng=rng,
                render_mode=config.render_mode,
            )
            if res.get("invalid_count", 0) > 0 and res.get("outputs"):
                for i, out in enumerate(res["outputs"]):
                    if len(out.split()) < config.min_output_length:
                        invalid_generations.append({
                            "iteration": iteration,
                            "prompt_text": res["prompt_text"][:200],
                            "output": out[:500],
                            "word_count": len(out.split()),
                        })
            res["iteration"] = iteration
            res["generation_count"] = len(res.get("valid_outputs", []))
            all_candidates.append(res)
            prompt_key = res["prompt_text"]
            if prompt_key not in iteration_found:
                iteration_found[prompt_key] = iteration
            scores.append((res["avg_reward"], res))

        # Sort by reward descending (higher = better)
        scores.sort(key=lambda x: x[0], reverse=True)
        top = [s[1] for s in scores[: config.top_k]]
        if iteration == 0 and top:
            best_initial_prompt_text = top[0]["prompt_text"]
            best_initial_reward = top[0]["avg_reward"]
        top_specs = [dict_to_prompt_spec(s["prompt_spec"]) for s in top]

        # New population: top_k specs + their mutated children + optional random explore
        population = []
        for spec in top_specs:
            population.append(spec.copy())
            for _ in range(config.children_per_parent - 1):
                population.append(mutate_spec(spec.copy(), rng, config.mutation_strength))
        # Random explorations
        for _ in range(config.keep_random_explore):
            base = rng.choice(seeds).copy()
            population.append(mutate_spec(base, rng, config.mutation_strength))
        # Trim to population_size
        population = population[: config.population_size]

    # Build leaderboard from all_candidates (dedupe by prompt_text, keep best reward)
    by_prompt: dict[str, dict] = {}
    for c in all_candidates:
        key = c["prompt_text"]
        if key not in by_prompt or c["avg_reward"] > by_prompt[key]["avg_reward"]:
            by_prompt[key] = c
    leaderboard = sorted(by_prompt.values(), key=lambda x: x["avg_reward"], reverse=True)

    # One example generation from the best prompt
    best_prompt_text = leaderboard[0]["prompt_text"] if leaderboard else ""
    example_generation = ""
    if best_prompt_text:
        example_generation = generator.generate_one(best_prompt_text)

    if out_path:
        with open(out_path / "leaderboard.jsonl", "w") as f:
            for row in leaderboard:
                out_row = {
                    "prompt_text": row["prompt_text"],
                    "prompt_spec": row["prompt_spec"],
                    "avg_reward": row["avg_reward"],
                    "avg_doc_slop_score": row["avg_doc_slop_score"],
                    "generation_count": row.get("generation_count", 0),
                    "iteration_found": iteration_found.get(row["prompt_text"], -1),
                }
                f.write(json.dumps(out_row) + "\n")
        best = leaderboard[0] if leaderboard else {}
        with open(out_path / "best_prompts.json", "w") as f:
            json.dump(leaderboard[:10], f, indent=2)
        with open(out_path / "generations.jsonl", "w") as f:
            for c in all_candidates:
                f.write(json.dumps({
                    "prompt_text": c["prompt_text"][:500],
                    "avg_reward": c["avg_reward"],
                    "outputs": c.get("valid_outputs", [])[:3],
                }) + "\n")
        if invalid_generations:
            with open(out_path / "invalid_generations.jsonl", "w") as f:
                for g in invalid_generations:
                    f.write(json.dumps(g) + "\n")
        with open(out_path / "report.md", "w") as f:
            f.write("# Prompt optimization report\n\n")
            f.write(f"Task: {task_instruction}\n\n")
            f.write(f"Iterations: {config.num_iterations}\n\n")
            f.write("## Top 5 prompts by reward\n\n")
            for i, row in enumerate(leaderboard[:5], 1):
                f.write(f"### {i}. reward={row['avg_reward']:.4f}\n\n")
                f.write(f"```\n{row['prompt_text']}\n```\n\n")

    return {
        "leaderboard": leaderboard,
        "best_prompt_text": leaderboard[0]["prompt_text"] if leaderboard else "",
        "best_avg_reward": leaderboard[0]["avg_reward"] if leaderboard else -1.0,
        "best_initial_prompt_text": best_initial_prompt_text,
        "best_initial_reward": best_initial_reward,
        "example_generation": example_generation,
        "output_dir": str(out_path) if out_path else None,
        "invalid_count": len(invalid_generations),
    }


def compare_seed_vs_optimized(
    task_instruction: str,
    generator: Any,
    reward_model: Any,
    optimized_prompt_text: str,
    n_samples: int = 5,
    min_length: int = 20,
    rng: Any = None,
    render_mode: str = "structured",
) -> dict[str, Any]:
    """Compare seed prompts vs one optimized prompt on the same task. Returns mean reward for seeds and for optimized."""
    import random
    rng = rng or random.Random(42)
    if render_mode not in RENDER_MODES:
        render_mode = "structured"
    seeds = get_seeds_for_task(task_instruction)
    seed_rewards = []
    for spec in seeds[:3]:
        res = evaluate_prompt(
            spec, generator, reward_model,
            n_samples=n_samples, min_length=min_length, rng=rng, render_mode=render_mode,
        )
        seed_rewards.append(res["avg_reward"])
    outputs = [generator.generate_one(optimized_prompt_text) for _ in range(n_samples)]
    valid = [t for t in outputs if len(t.split()) >= min_length]
    if not valid:
        opt_reward = -1.0
    else:
        reward_model.load()
        r = reward_model.score_batch(valid, return_diagnostics=False)
        opt_reward = sum(r["reward"]) / len(r["reward"])
    return {
        "seed_mean_reward": sum(seed_rewards) / len(seed_rewards) if seed_rewards else -1.0,
        "seed_rewards": seed_rewards,
        "optimized_mean_reward": opt_reward,
        "n_samples": n_samples,
    }


def compare_rendering_modes(
    task_instruction: str,
    generator: Any,
    reward_model: Any,
    seed_specs: list[PromptSpec] | None = None,
    n_samples: int = 3,
    min_length: int = 20,
    rng: Any = None,
) -> dict[str, Any]:
    """Test the same seed prompts under structured, simple, and compact rendering.

    Returns per-mode average reward and example outputs for each mode.
    """
    import random
    rng = rng or random.Random(42)
    seeds = seed_specs or get_seeds_for_task(task_instruction)
    seeds = seeds[:3]  # use first 3 seeds

    reward_model.load()
    results: dict[str, Any] = {
        "task_instruction": task_instruction,
        "n_samples": n_samples,
        "modes": {},
    }

    for mode in RENDER_MODES:
        mode_rewards: list[float] = []
        mode_examples: list[str] = []
        for spec in seeds:
            res = evaluate_prompt(
                spec,
                generator,
                reward_model,
                n_samples=n_samples,
                min_length=min_length,
                rng=rng,
                render_mode=mode,
            )
            mode_rewards.append(res["avg_reward"])
            if res.get("valid_outputs"):
                mode_examples.append(res["valid_outputs"][0][:400])
            elif res.get("outputs"):
                mode_examples.append(res["outputs"][0][:400] if res["outputs"] else "")
            else:
                mode_examples.append("")
        avg_reward = sum(mode_rewards) / len(mode_rewards) if mode_rewards else -1.0
        results["modes"][mode] = {
            "avg_reward": avg_reward,
            "per_seed_rewards": mode_rewards,
            "example_outputs": mode_examples,
            "example_prompts": [render_prompt(spec, mode=mode)[:300] for spec in seeds],
        }

    return results


def compare_generators(
    task_instruction: str,
    generators: list[tuple[str, Any]],  # [(label, generator), ...]
    reward_model: Any,
    render_mode: str = "simple",
    seed_specs: list[PromptSpec] | None = None,
    n_samples: int = 3,
    min_length: int = 20,
    rng: Any = None,
) -> dict[str, Any]:
    """Compare the same task and seed prompts across multiple generator models.

    Uses the same reward model and render_mode for each. For each generator, returns
    average reward, per-seed rewards, and example outputs.
    """
    import random
    rng = rng or random.Random(42)
    if render_mode not in RENDER_MODES:
        render_mode = "structured"
    seeds = seed_specs or get_seeds_for_task(task_instruction)
    seeds = seeds[:3]

    reward_model.load()
    results: dict[str, Any] = {
        "task_instruction": task_instruction,
        "render_mode": render_mode,
        "n_samples": n_samples,
        "generators": {},
    }

    for label, generator in generators:
        mode_rewards: list[float] = []
        mode_examples: list[str] = []
        for spec in seeds:
            res = evaluate_prompt(
                spec,
                generator,
                reward_model,
                n_samples=n_samples,
                min_length=min_length,
                rng=rng,
                render_mode=render_mode,
            )
            mode_rewards.append(res["avg_reward"])
            if res.get("valid_outputs"):
                mode_examples.append(res["valid_outputs"][0][:400])
            elif res.get("outputs"):
                mode_examples.append(res["outputs"][0][:400] if res["outputs"] else "")
            else:
                mode_examples.append("")
        avg_reward = sum(mode_rewards) / len(mode_rewards) if mode_rewards else -1.0
        results["generators"][label] = {
            "avg_reward": avg_reward,
            "per_seed_rewards": mode_rewards,
            "example_outputs": mode_examples,
        }

    return results
