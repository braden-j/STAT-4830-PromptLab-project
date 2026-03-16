"""Prompt optimization: mutation + hill climbing to minimize slop using the reward model."""

from .templates import (
    PromptSpec,
    render_prompt,
    RENDER_MODES,
    prompt_spec_to_dict,
    dict_to_prompt_spec,
    get_seeds_for_task,
    SEED_PROMPT_SPECS,
)
from .mutations import mutate_spec
from .generator import FrozenGenerator, GeneratorConfig
from .evolve import (
    evaluate_prompt,
    run_hill_climbing,
    HillClimbConfig,
    compare_seed_vs_optimized,
    compare_rendering_modes,
    compare_generators,
)

__all__ = [
    "PromptSpec",
    "render_prompt",
    "RENDER_MODES",
    "prompt_spec_to_dict",
    "dict_to_prompt_spec",
    "get_seeds_for_task",
    "SEED_PROMPT_SPECS",
    "mutate_spec",
    "FrozenGenerator",
    "GeneratorConfig",
    "evaluate_prompt",
    "run_hill_climbing",
    "HillClimbConfig",
    "compare_seed_vs_optimized",
    "compare_rendering_modes",
    "compare_generators",
]
