"""Scoring wrappers that reuse canonical Pangram logic from existing scripts."""

from __future__ import annotations

import importlib.util
import statistics
from pathlib import Path
from typing import Any


def _load_script_module(name: str, relative_path: str):
    """Load a script module by path so we can reuse existing canonical functions."""

    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_canonical_pangram_module():
    """Return the validated scoring helpers from the existing script."""

    return _load_script_module("tournament_fluency_reward_validation", "scripts/fluency_reward_validation.py")


def length_ratio(input_text: str, output_text: str) -> float:
    """Compute output/input token-count ratio."""

    in_len = max(len(input_text.split()), 1)
    out_len = len(output_text.split())
    return out_len / in_len


def invalid_output(text: str) -> bool:
    """Conservative invalid-output heuristic used in evaluation gates."""

    stripped = text.strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if lowered.startswith("rewrite the following"):
        return True
    if len(stripped.split()) < 8:
        return True
    return False


def summarize_gate_metrics(
    similarities: list[float],
    ratios: list[float],
    invalid_flags: list[bool],
) -> dict[str, Any]:
    """Return the aggregate gate metrics used by eval scripts."""

    return {
        "mean_similarity": statistics.fmean(similarities) if similarities else 0.0,
        "median_length_ratio": statistics.median(ratios) if ratios else 0.0,
        "invalid_rate": (sum(1 for x in invalid_flags if x) / len(invalid_flags)) if invalid_flags else 1.0,
    }
