"""Metrics for token-level and document-level slop classification."""

from __future__ import annotations

import torch


def token_level_f1(
    preds: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Macro F1 over token predictions, ignoring padding/special (ignore_index)."""
    mask = labels != ignore_index
    if mask.sum() == 0:
        return 0.0
    pred_flat = preds[mask].long()
    label_flat = labels[mask].long()
    # Binary: 0 vs 1
    tp = ((pred_flat == 1) & (label_flat == 1)).sum().float().item()
    fp = ((pred_flat == 1) & (label_flat == 0)).sum().float().item()
    fn = ((pred_flat == 0) & (label_flat == 1)).sum().float().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def token_level_auroc(
    probs: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """AUROC for token-level slop (class 1) probability vs binary labels."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return 0.0
    mask = labels != ignore_index
    if mask.sum() == 0:
        return 0.0
    probs_flat = probs[mask].float().cpu().numpy()
    label_flat = labels[mask].long().cpu().numpy()
    if len(set(label_flat)) < 2:
        return 0.0
    return float(roc_auc_score(label_flat, probs_flat))


def doc_level_auroc(
    doc_scores: torch.Tensor,
    doc_labels: torch.Tensor,
) -> float:
    """AUROC for document-level slop score (e.g. mean token slop prob) vs doc binary label."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return 0.0
    scores = doc_scores.float().cpu().numpy()
    labels = doc_labels.long().cpu().numpy()
    if len(set(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def doc_labels_from_token_labels(
    token_labels: torch.Tensor,
    attention_mask: torch.Tensor,
    ignore_index: int = -100,
    strategy: str = "any",
) -> torch.Tensor:
    """Aggregate token labels to one label per document.

    strategy: "any" -> 1 if any token is slop else 0; "mean" -> 1 if mean > 0.5 else 0.
    """
    batch_size = token_labels.shape[0]
    out = torch.zeros(batch_size, dtype=torch.long, device=token_labels.device)
    for i in range(batch_size):
        mask = (attention_mask[i] == 1) & (token_labels[i] != ignore_index)
        valid = token_labels[i][mask]
        if valid.numel() == 0:
            out[i] = 0
            continue
        if strategy == "any":
            out[i] = 1 if (valid == 1).any().item() else 0
        else:
            out[i] = 1 if valid.float().mean().item() > 0.5 else 0
    return out
