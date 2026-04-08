"""Unit tests for scoring utilities."""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from hill_climb.scoring import compute_reward, aggregate_token_scores


def test_aggregate_token_scores_mean():
    """Mean aggregation over tokens."""
    probs = torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    result = aggregate_token_scores(probs, attention_mask=mask, reduction="mean")
    assert result.shape == (2,)
    assert torch.allclose(result[0], torch.tensor(0.2))
    assert torch.allclose(result[1], torch.tensor(0.5))


def test_aggregate_token_scores_no_mask():
    """Without mask, mean over full sequence."""
    probs = torch.tensor([[0.1, 0.2, 0.3]])
    result = aggregate_token_scores(probs, attention_mask=None, reduction="mean")
    assert result.shape == (1,)
    assert torch.allclose(result[0], torch.tensor(0.2))


def test_compute_reward_interface():
    """compute_reward expects model with score_tokens - mock for interface test."""
    class MockModel:
        def score_tokens(self, input_ids, attention_mask=None):
            # Return low slop prob => high reward
            return torch.full((input_ids.shape[0], input_ids.shape[1]), 0.1)

    model = MockModel()
    input_ids = torch.randint(0, 100, (2, 10))
    mask = torch.ones(2, 10)
    rewards = compute_reward(model, input_ids, mask)
    assert rewards.shape == (2,)
    assert torch.allclose(rewards, torch.tensor([0.9, 0.9]))


def test_aggregate_token_scores_sum():
    """Sum aggregation."""
    probs = torch.tensor([[0.1, 0.2, 0.3]])
    result = aggregate_token_scores(probs, reduction="sum")
    assert result.shape == (1,)
    assert torch.allclose(result[0], torch.tensor(0.6))


def test_aggregate_token_scores_max():
    """Max aggregation: max slop prob per sequence."""
    probs = torch.tensor([[0.1, 0.9, 0.2], [0.3, 0.3, 0.0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    result = aggregate_token_scores(probs, attention_mask=mask, reduction="max")
    assert result.shape == (2,)
    assert torch.allclose(result[0], torch.tensor(0.9))
    assert torch.allclose(result[1], torch.tensor(0.3))


def test_aggregate_token_scores_topk():
    """Top-k mean: average of top 50% of token slop probs."""
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    result = aggregate_token_scores(probs, reduction="topk", topk_fraction=0.5)
    assert result.shape == (1,)
    # Top 2 of [0.1,0.2,0.3,0.4] -> mean(0.4, 0.3) = 0.35
    assert torch.allclose(result[0], torch.tensor(0.35))


def test_reward_aggregation_and_penalties():
    """Reward = -(doc_slop_score + penalties); diagnostics exposed."""
    from hill_climb.scoring.diagnostics import compute_diagnostics, repetition_ratio
    from hill_climb.scoring.reward import length_penalty_single, generic_phrase_ratio_single

    d = compute_diagnostics("Short.", token_count=1)
    assert d["very_short"] is True
    assert "repetition_ratio" in d

    # "the the the cat" has repeated bigram (the,the) at i=0 and i=1
    assert repetition_ratio("the the the cat", n=2) > 0
    assert length_penalty_single(2, 5, 100) > 0
    assert length_penalty_single(5, 5, 100) == 0.0
    assert generic_phrase_ratio_single("you know like um", ["like", "um"]) >= 0
