"""Aggregation of per-token slop scores to document-level scores."""

from __future__ import annotations

import torch


def aggregate_token_scores(
    slop_probs: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    reduction: str = "mean",
    topk_fraction: float | None = None,
) -> torch.Tensor:
    """Aggregate per-token slop probabilities into sequence-level score.

    reduction: "mean" | "max" | "topk"
    topk_fraction: used when reduction=="topk"; fraction of tokens to average (e.g. 0.1 = top 10%).
    """
    if attention_mask is not None:
        mask = (attention_mask == 1).float()
        count = mask.sum(dim=1, keepdim=True).clamp(min=1)
    else:
        mask = torch.ones_like(slop_probs)
        count = torch.tensor(slop_probs.shape[1], device=slop_probs.device, dtype=slop_probs.dtype)

    if reduction == "mean":
        if attention_mask is not None:
            return (slop_probs * mask).sum(dim=1) / count.squeeze(-1).clamp(min=1)
        return slop_probs.mean(dim=1)
    if reduction == "sum":
        if attention_mask is not None:
            return (slop_probs * mask).sum(dim=1)
        return slop_probs.sum(dim=1)
    if reduction == "max":
        p = slop_probs.clone()
        if attention_mask is not None:
            p[attention_mask != 1] = float("-inf")
        return p.max(dim=1).values
    if reduction == "topk" and topk_fraction is not None:
        p = slop_probs.clone()
        if attention_mask is not None:
            p[attention_mask != 1] = float("-inf")
        seq_len = p.shape[1]
        k = max(1, min(int(seq_len * topk_fraction), seq_len))
        topv = p.topk(k, dim=1).values
        return topv.mean(dim=1)
    return slop_probs.mean(dim=1)


def compute_reward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-sequence reward: 1 - mean(slop_prob) over non-padding tokens.
    Higher reward = less slop. Kept for backward compatibility.
    """
    slop_probs = model.score_tokens(input_ids, attention_mask)
    doc_scores = aggregate_token_scores(slop_probs, attention_mask=attention_mask, reduction="mean")
    return 1.0 - doc_scores
