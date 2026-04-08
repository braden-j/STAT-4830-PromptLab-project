"""Slop generator: rule-based and model-based sloppification."""

from .rule_sloppifier import (
    DIFFICULTY_PRESETS,
    RuleSloppifier,
    sloppify,
    sloppify_with_labels,
)

__all__ = ["DIFFICULTY_PRESETS", "RuleSloppifier", "sloppify", "sloppify_with_labels"]
