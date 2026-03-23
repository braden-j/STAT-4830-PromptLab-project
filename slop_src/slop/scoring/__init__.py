"""Scoring and reward model for slop classifier."""

from .aggregation import aggregate_token_scores, compute_reward
from .reward import RewardConfig, SlopRewardModel
from .diagnostics import (
    compute_diagnostics,
    compute_structural_diagnostics,
    compute_semantic_diagnostics,
    compute_quality_diagnostics,
)

__all__ = [
    "aggregate_token_scores",
    "compute_reward",
    "RewardConfig",
    "SlopRewardModel",
    "compute_diagnostics",
    "compute_structural_diagnostics",
    "compute_semantic_diagnostics",
    "compute_quality_diagnostics",
]
