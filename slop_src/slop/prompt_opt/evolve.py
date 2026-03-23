"""Hill climbing / evolutionary search over PromptSpecs using reward model."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .templates import PromptSpec, render_prompt, get_seeds_for_task, prompt_spec_to_dict, dict_to_prompt_spec, RENDER_MODES
from .mutations import mutate_spec
from slop.scoring.diagnostics import compute_semantic_diagnostics, compute_quality_diagnostics


def _task_keywords_from_instruction(task_instruction: str, max_words: int = 10) -> list[str]:
    """Extract lightweight task keywords from task instruction for relevance scoring."""
    import re
    stop = {"the", "a", "an", "to", "in", "on", "for", "of", "and", "or", "is", "are", "be", "by", "with", "as", "your", "you", "that", "this", "it", "if", "when", "how", "what", "why", "can", "should", "use", "write", "explain", "give", "provide"}
    words = re.findall(r"\b[a-z]{3,}\b", task_instruction.lower())
    seen = set()
    out = []
    for w in words:
        if w not in stop and w not in seen:
            seen.add(w)
            out.append(w)
            if len(out) >= max_words:
                break
    return out


def _structural_penalty_from_diagnostics(
    diagnostics: list[dict],
    threshold: float = 0.25,
) -> tuple[float, dict[str, float]]:
    """Compute a smooth structural penalty (0..1) from per-output diagnostics. Above threshold ramps up."""
    if not diagnostics:
        return 0.0, {}
    densities = []
    for d in diagnostics:
        v = d.get("abnormal_punctuation_density")
        if isinstance(v, (int, float)):
            densities.append(float(v))
    if not densities:
        return 0.0, {}
    avg_density = sum(densities) / len(densities)
    # Penalty: 0 below threshold, then linear ramp to 1 at density=1
    denom = 1.0 - threshold
    penalty = max(0.0, (avg_density - threshold) / denom) if denom > 0 else (1.0 if avg_density > threshold else 0.0)
    summary = {}
    for key in ["bullet_like_line_ratio", "repeated_dash_ratio", "list_marker_ratio", "abnormal_punctuation_density"]:
        vals = [d[key] for d in diagnostics if key in d and isinstance(d.get(key), (int, float))]
        if vals:
            summary[key] = sum(vals) / len(vals)
    return penalty, summary


def _semantic_penalty_from_outputs(
    valid_outputs: list[str],
    prompt_text: str,
    task_keywords: list[str],
) -> tuple[float, dict[str, float]]:
    """Compute semantic penalty (0..1) from meta-instructional and off-task scores. Returns (penalty, summary)."""
    if not valid_outputs:
        return 0.0, {}
    diags = [
        compute_semantic_diagnostics(out, prompt_text=prompt_text, task_keywords=task_keywords)
        for out in valid_outputs
    ]
    # Combined penalty: meta/advice/echo + low task relevance
    penalties = []
    for d in diags:
        meta = d.get("semantic_meta_score", 0.0)
        off_task = d.get("semantic_off_task_score", 1.0 - d.get("task_relevance_score", 0.0))
        penalties.append(min(1.0, 0.6 * meta + 0.4 * off_task))
    penalty = sum(penalties) / len(penalties) if penalties else 0.0
    summary = {}
    for key in ["instruction_echo_ratio", "writing_advice_ratio", "prompt_copy_ratio", "off_task_generic_ratio", "task_relevance_score", "semantic_meta_score"]:
        vals = [d[key] for d in diags if key in d and isinstance(d.get(key), (int, float))]
        if vals:
            summary[key] = sum(vals) / len(vals)
    return penalty, summary


def _quality_reward_from_outputs(
    valid_outputs: list[str],
    structural_diag_summary: dict[str, float],
    semantic_diag_summary: dict[str, float],
    task_keywords: list[str],
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute average quality score (0..1) and summary diagnostics (means)."""
    if not valid_outputs:
        return 0.0, {}
    diags = [
        compute_quality_diagnostics(
            out,
            prompt_text=None,
            task_keywords=task_keywords,
            structural_diag=structural_diag_summary,
            semantic_diag=semantic_diag_summary,
            weights=weights,
        )
        for out in valid_outputs
    ]
    # Aggregate (mean)
    summary: dict[str, float] = {}
    for key in ["information_density_score", "clarity_score", "completeness_score", "quality_score"]:
        vals = [d[key] for d in diags if isinstance(d.get(key), (int, float))]
        if vals:
            summary[key] = sum(vals) / len(vals)
    avg_quality = summary.get("quality_score", 0.0)
    return float(avg_quality), summary


def evaluate_prompt(
    prompt_spec: PromptSpec,
    generator: Any,
    reward_model: Any,
    n_samples: int = 2,
    min_length: int = 20,
    rng: Any = None,
    render_mode: str = "structured",
    lambda_structural: float = 0.0,
    structural_threshold: float = 0.25,
    lambda_semantic: float = 0.0,
    lambda_quality: float = 0.0,
    task_instruction: str | None = None,
    task_keywords: list[str] | None = None,
    quality_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Generate n_samples from the prompt, score with reward model, return averaged metrics and diagnostics.

    Outputs below min_length (word count) are rejected and not scored; they are counted in invalid_count.
    Total reward:
      base_reward
      - lambda_structural * structural_penalty
      - lambda_semantic * semantic_penalty
      + lambda_quality * quality_score
    task_keywords used for task relevance; if None and task_instruction provided, derived from task_instruction.
    """
    import random
    rng = rng or random.Random(42)
    if render_mode not in RENDER_MODES:
        render_mode = "structured"
    prompt_text = render_prompt(prompt_spec, mode=render_mode)
    keywords = task_keywords if task_keywords is not None else (_task_keywords_from_instruction(task_instruction) if task_instruction else [])

    outputs: list[str] = []
    for _ in range(n_samples):
        out = generator.generate_one(prompt_text)
        outputs.append(out)

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
            "avg_base_reward": -1.0,
            "avg_structural_penalty": 0.0,
            "structural_penalty_contribution": 0.0,
            "avg_semantic_penalty": 0.0,
            "semantic_penalty_contribution": 0.0,
            "avg_quality_score": 0.0,
            "quality_reward_contribution": 0.0,
            "avg_doc_slop_score": 1.0,
            "diagnostics_summary": {},
            "structural_diagnostics": {},
            "semantic_diagnostics": {},
            "quality_diagnostics": {},
            "error": "all outputs below min_length",
        }

    reward_model.load()
    result = reward_model.score_batch(valid_outputs, return_diagnostics=True)
    base_rewards = result["reward"]
    doc_scores = result["doc_slop_score"]
    diagnostics = result.get("diagnostics") or []

    avg_base_reward = sum(base_rewards) / len(base_rewards)
    avg_doc_slop_score = sum(doc_scores) / len(doc_scores)
    diag_summary: dict[str, float] = {}
    if diagnostics:
        for k in ["repetition_ratio", "punctuation_ratio", "caps_ratio", "filler_loop_score"]:
            if k in diagnostics[0]:
                vals = [d[k] for d in diagnostics if isinstance(d.get(k), (int, float))]
                if vals:
                    diag_summary[k] = sum(vals) / len(vals)

    structural_penalty, structural_diagnostics = _structural_penalty_from_diagnostics(
        diagnostics, threshold=structural_threshold
    )
    structural_contribution = lambda_structural * structural_penalty

    semantic_penalty, semantic_diagnostics = _semantic_penalty_from_outputs(
        valid_outputs, prompt_text, keywords
    )
    semantic_contribution = lambda_semantic * semantic_penalty

    avg_quality_score, quality_diagnostics = _quality_reward_from_outputs(
        valid_outputs,
        structural_diag_summary={**diag_summary, **structural_diagnostics},
        semantic_diag_summary=semantic_diagnostics,
        task_keywords=keywords,
        weights=quality_weights,
    )
    quality_contribution = lambda_quality * avg_quality_score

    avg_reward = avg_base_reward - structural_contribution - semantic_contribution + quality_contribution

    return {
        "prompt_spec": prompt_spec_to_dict(prompt_spec),
        "prompt_text": prompt_text,
        "outputs": outputs,
        "valid_outputs": valid_outputs,
        "invalid_count": invalid_count,
        "n_samples": n_samples,
        "avg_reward": avg_reward,
        "avg_base_reward": avg_base_reward,
        "avg_structural_penalty": structural_penalty,
        "structural_penalty_contribution": structural_contribution,
        "avg_semantic_penalty": semantic_penalty,
        "semantic_penalty_contribution": semantic_contribution,
        "avg_quality_score": avg_quality_score,
        "quality_reward_contribution": quality_contribution,
        "avg_doc_slop_score": avg_doc_slop_score,
        "diagnostics_summary": diag_summary,
        "structural_diagnostics": structural_diagnostics,
        "semantic_diagnostics": semantic_diagnostics,
        "quality_diagnostics": quality_diagnostics,
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
    # Structural penalty: discourage punctuation-heavy / list-heavy outputs (A8 refinement)
    lambda_structural: float = 0.15
    structural_threshold: float = 0.25  # penalty ramps above this abnormal_punctuation_density
    preserve_one_unmutated_best: bool = True  # keep one copy of best spec unmutated in next population
    # Semantic penalty: discourage meta-instructional, writing-advice, off-task outputs (A8 refinement)
    lambda_semantic: float = 0.12
    task_keywords: list[str] = field(default_factory=list)  # for task relevance; empty => derive from task_instruction
    # Quality reward (lightweight heuristics): modest positive signal for useful outputs
    lambda_quality: float = 0.10
    quality_weights: dict[str, float] = field(default_factory=lambda: {
        "information_density": 0.4,
        "clarity": 0.3,
        "completeness": 0.3,
    })
    # Search extensions
    semantic_mutation_probability: float = 0.25
    exploration_rate: float = 0.15  # fraction of next-gen reserved for exploration/immigrants
    exploration_epsilon: float = 0.10  # chance to pick lower-ranked parent for mutation
    num_random_immigrants: int | None = None  # if set, overrides exploration_rate
    enable_eval_cache: bool = True  # in-memory eval cache within a run


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

    # In-memory eval cache (per-run) for identical evaluation contexts.
    eval_cache: dict[tuple, dict[str, Any]] = {}
    cache_hits = 0
    cache_misses = 0

    seeds = seed_specs or get_seeds_for_task(task_instruction)
    population: list[PromptSpec] = []
    # Start with seeds; if we need more to fill population_size, duplicate randomly
    while len(population) < config.population_size:
        population.append(rng.choice(seeds).copy())
    population = population[: config.population_size]

    # Provenance / mutation-type tracking keyed by rendered prompt text.
    # This is stable across copies of the same logical prompt and extensible for future operators.
    provenance_by_prompt: dict[str, str] = {}
    mutation_info_by_prompt: dict[str, dict[str, Any]] = {}

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
                "lambda_structural": getattr(config, "lambda_structural", 0.0),
                "structural_threshold": getattr(config, "structural_threshold", 0.25),
                "preserve_one_unmutated_best": getattr(config, "preserve_one_unmutated_best", True),
                "lambda_semantic": getattr(config, "lambda_semantic", 0.0),
                "task_keywords": getattr(config, "task_keywords", []) or [],
                "lambda_quality": getattr(config, "lambda_quality", 0.0),
                "quality_weights": getattr(config, "quality_weights", {}) or {},
                "semantic_mutation_probability": getattr(config, "semantic_mutation_probability", 0.0),
                "exploration_rate": getattr(config, "exploration_rate", 0.0),
                "exploration_epsilon": getattr(config, "exploration_epsilon", 0.0),
                "num_random_immigrants": getattr(config, "num_random_immigrants", None),
                "enable_eval_cache": getattr(config, "enable_eval_cache", True),
            }
            yaml.dump(cfg_dict, f, default_flow_style=False)

    # Initial population comes directly from seeds.
    for spec in population:
        pt = render_prompt(spec, mode=config.render_mode)
        provenance_by_prompt.setdefault(pt, "seed")
        mutation_info_by_prompt.setdefault(pt, {"mutation_type": "none", "mutation_helper": None})

    for iteration in range(config.num_iterations):
        # Evaluate current population
        scores: list[tuple[float, dict[str, Any]]] = []
        for spec in population:
            # Cache key: rendered prompt + evaluation knobs + generator knobs.
            prompt_text = render_prompt(spec, mode=config.render_mode)
            cache_key = (
                prompt_text,
                config.samples_per_prompt,
                config.min_output_length,
                config.render_mode,
                getattr(config, "lambda_structural", 0.0),
                getattr(config, "structural_threshold", 0.25),
                getattr(config, "lambda_semantic", 0.0),
                tuple(getattr(config, "task_keywords", []) or []),
                getattr(config, "lambda_quality", 0.0),
                tuple(sorted((getattr(config, "quality_weights", {}) or {}).items())),
                # Generator knobs that affect stochastic outputs
                getattr(generator.config, "model_name", None),
                getattr(generator.config, "temperature", None),
                getattr(generator.config, "top_p", None),
                getattr(generator.config, "max_new_tokens", None),
                getattr(generator.config, "repetition_penalty", None),
                getattr(generator.config, "no_repeat_ngram_size", None),
            )

            if getattr(config, "enable_eval_cache", True) and cache_key in eval_cache:
                res = dict(eval_cache[cache_key])
                cache_hits += 1
            else:
                res = evaluate_prompt(
                    spec,
                    generator,
                    reward_model,
                    n_samples=config.samples_per_prompt,
                    min_length=config.min_output_length,
                    rng=rng,
                    render_mode=config.render_mode,
                    lambda_structural=getattr(config, "lambda_structural", 0.0),
                    structural_threshold=getattr(config, "structural_threshold", 0.25),
                    lambda_semantic=getattr(config, "lambda_semantic", 0.0),
                    lambda_quality=getattr(config, "lambda_quality", 0.0),
                    task_instruction=task_instruction,
                    task_keywords=config.task_keywords if getattr(config, "task_keywords", None) else None,
                    quality_weights=getattr(config, "quality_weights", None),
                )
                if getattr(config, "enable_eval_cache", True):
                    eval_cache[cache_key] = dict(res)
                cache_misses += 1

            # Attach provenance / mutation metadata for later inspection.
            res["provenance"] = provenance_by_prompt.get(prompt_text, "unknown")
            mi = mutation_info_by_prompt.get(prompt_text, {})
            res["mutation_type"] = mi.get("mutation_type", "none")
            res["mutation_helper"] = mi.get("mutation_helper")
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
        preserve_best = getattr(config, "preserve_one_unmutated_best", True)
        if preserve_best and top_specs:
            best_spec = top_specs[0].copy()
            population.append(best_spec)  # one unmutated best
            pt = render_prompt(best_spec, mode=config.render_mode)
            provenance_by_prompt[pt] = "elite_carryover"
            mutation_info_by_prompt[pt] = {"mutation_type": "none", "mutation_helper": None}

        # Exploration: reserve some slots for “random immigrants”.
        exploration_rate = float(getattr(config, "exploration_rate", 0.0) or 0.0)
        num_random_immigrants = getattr(config, "num_random_immigrants", None)
        if num_random_immigrants is not None:
            n_immigrants = max(0, int(num_random_immigrants))
        else:
            n_immigrants = int(round(config.population_size * exploration_rate)) if exploration_rate > 0 else 0
        n_immigrants = min(n_immigrants, max(0, config.population_size - 1))

        # Mutated children (epsilon exploration chooses lower-ranked parent sometimes).
        eps = float(getattr(config, "exploration_epsilon", 0.0) or 0.0)
        sem_p = float(getattr(config, "semantic_mutation_probability", 0.0) or 0.0)
        for i, spec in enumerate(top_specs):
            if i > 0 or not preserve_best:
                carried = spec.copy()
                population.append(carried)
                pt = render_prompt(carried, mode=config.render_mode)
                provenance_by_prompt[pt] = "elite_carryover"
                mutation_info_by_prompt[pt] = {"mutation_type": "none", "mutation_helper": None}
            for _ in range(config.children_per_parent - 1):
                parent = spec
                used_epsilon = False
                if top_specs and eps > 0 and rng.random() < eps:
                    parent = rng.choice(top_specs)
                    used_epsilon = True
                info: dict[str, Any] = {}
                child = mutate_spec(
                    parent.copy(),
                    rng,
                    config.mutation_strength,
                    semantic_mutation_probability=sem_p,
                    mutation_info=info,
                )
                population.append(child)
                pt = render_prompt(child, mode=config.render_mode)
                # Provenance taxonomy for children
                if used_epsilon:
                    provenance_by_prompt[pt] = "epsilon_parent_mutation"
                elif info.get("mutation_type") == "semantic":
                    provenance_by_prompt[pt] = "semantic_mutation"
                else:
                    provenance_by_prompt[pt] = "standard_mutation"
                mutation_info_by_prompt[pt] = {
                    "mutation_type": info.get("mutation_type", "slot"),
                    "mutation_helper": info.get("mutation_helper"),
                }

        # Keep legacy random explore (small) for backward compatibility.
        for _ in range(config.keep_random_explore):
            base = rng.choice(seeds).copy()
            info: dict[str, Any] = {}
            explorer = mutate_spec(
                base,
                rng,
                config.mutation_strength,
                semantic_mutation_probability=sem_p,
                mutation_info=info,
            )
            population.append(explorer)
            pt = render_prompt(explorer, mode=config.render_mode)
            provenance_by_prompt[pt] = "random_exploration_seed_mutant"
            mutation_info_by_prompt[pt] = {
                "mutation_type": info.get("mutation_type", "slot"),
                "mutation_helper": info.get("mutation_helper"),
            }

        # Add immigrants last; they will survive trimming if exploration is enabled.
        for _ in range(n_immigrants):
            if rng.random() < 0.5:
                imm = rng.choice(seeds).copy()
                population.append(imm)
                pt = render_prompt(imm, mode=config.render_mode)
                provenance_by_prompt[pt] = "random_immigrant_seed"
                mutation_info_by_prompt[pt] = {"mutation_type": "none", "mutation_helper": None}
            else:
                parent = rng.choice(top_specs).copy() if top_specs else rng.choice(seeds).copy()
                info: dict[str, Any] = {}
                imm = mutate_spec(
                    parent,
                    rng,
                    config.mutation_strength,
                    semantic_mutation_probability=sem_p,
                    mutation_info=info,
                )
                population.append(imm)
                pt = render_prompt(imm, mode=config.render_mode)
                provenance_by_prompt[pt] = "random_immigrant_mutated_parent"
                mutation_info_by_prompt[pt] = {
                    "mutation_type": info.get("mutation_type", "slot"),
                    "mutation_helper": info.get("mutation_helper"),
                }

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
                    "avg_base_reward": row.get("avg_base_reward", row["avg_reward"]),
                    "avg_structural_penalty": row.get("avg_structural_penalty", 0.0),
                    "structural_penalty_contribution": row.get("structural_penalty_contribution", 0.0),
                    "avg_semantic_penalty": row.get("avg_semantic_penalty", 0.0),
                    "semantic_penalty_contribution": row.get("semantic_penalty_contribution", 0.0),
                    "avg_quality_score": row.get("avg_quality_score", 0.0),
                    "quality_reward_contribution": row.get("quality_reward_contribution", 0.0),
                    "structural_diagnostics": row.get("structural_diagnostics", {}),
                    "semantic_diagnostics": row.get("semantic_diagnostics", {}),
                    "quality_diagnostics": row.get("quality_diagnostics", {}),
                    "avg_doc_slop_score": row["avg_doc_slop_score"],
                    "generation_count": row.get("generation_count", 0),
                    "iteration_found": iteration_found.get(row["prompt_text"], -1),
                    "provenance": row.get("provenance", "unknown"),
                    "mutation_type": row.get("mutation_type", "none"),
                    "mutation_helper": row.get("mutation_helper"),
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
                    "avg_base_reward": c.get("avg_base_reward", c["avg_reward"]),
                    "structural_penalty_contribution": c.get("structural_penalty_contribution", 0.0),
                    "semantic_penalty_contribution": c.get("semantic_penalty_contribution", 0.0),
                    "quality_reward_contribution": c.get("quality_reward_contribution", 0.0),
                    "structural_diagnostics": c.get("structural_diagnostics", {}),
                    "semantic_diagnostics": c.get("semantic_diagnostics", {}),
                    "quality_diagnostics": c.get("quality_diagnostics", {}),
                    "outputs": c.get("valid_outputs", [])[:3],
                    "provenance": c.get("provenance", "unknown"),
                    "mutation_type": c.get("mutation_type", "none"),
                    "mutation_helper": c.get("mutation_helper"),
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
                base = row.get("avg_base_reward", row["avg_reward"])
                sp = row.get("structural_penalty_contribution", 0.0)
                mp = row.get("semantic_penalty_contribution", 0.0)
                qp = row.get("quality_reward_contribution", 0.0)
                prov = row.get("provenance", "unknown")
                mtype = row.get("mutation_type", "none")
                mhelper = row.get("mutation_helper") or "n/a"
                f.write(
                    f"### {i}. reward={row['avg_reward']:.4f} "
                    f"(base={base:.4f}, struct_pen={sp:.4f}, semantic_pen={mp:.4f}, "
                    f"quality={qp:.4f}, source={prov}, mutation_type={mtype}, mutation_helper={mhelper})\n\n"
                )
                f.write(f"```\n{row['prompt_text']}\n```\n\n")

        # Lightweight run-level stats for cache and search settings.
        with open(out_path / "stats.json", "w") as f:
            json.dump(
                {
                    "eval_cache_hits": cache_hits,
                    "eval_cache_misses": cache_misses,
                    "cache_enabled": getattr(config, "enable_eval_cache", True),
                    "lambda_quality": getattr(config, "lambda_quality", 0.0),
                    "semantic_mutation_probability": getattr(config, "semantic_mutation_probability", 0.0),
                    "exploration_rate": getattr(config, "exploration_rate", 0.0),
                    "exploration_epsilon": getattr(config, "exploration_epsilon", 0.0),
                    "num_random_immigrants": getattr(config, "num_random_immigrants", None),
                },
                f,
                indent=2,
            )

    return {
        "leaderboard": leaderboard,
        "best_prompt_text": leaderboard[0]["prompt_text"] if leaderboard else "",
        "best_avg_reward": leaderboard[0]["avg_reward"] if leaderboard else -1.0,
        "best_initial_prompt_text": best_initial_prompt_text,
        "best_initial_reward": best_initial_reward,
        "example_generation": example_generation,
        "output_dir": str(out_path) if out_path else None,
        "invalid_count": len(invalid_generations),
        "eval_cache_hits": cache_hits,
        "eval_cache_misses": cache_misses,
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
