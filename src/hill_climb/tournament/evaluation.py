"""Evaluation helpers for checkpoint scoring and leaderboard creation."""

from __future__ import annotations

from typing import Any

from hill_climb.tournament.scoring import invalid_output, length_ratio


def compute_delta(input_score: float, output_score: float) -> float:
    """Primary tournament metric."""

    return input_score - output_score


def classify_eval_row(
    row: dict[str, Any],
    mean_similarity_gate: float,
    min_length_ratio: float,
    max_length_ratio: float,
) -> dict[str, Any]:
    """Stamp common evaluation fields on a scored row."""

    ratio = length_ratio(row["input_text"], row["output_text"])
    invalid = invalid_output(row["output_text"])
    passed = (
        row["similarity"] >= mean_similarity_gate
        and min_length_ratio <= ratio <= max_length_ratio
        and not invalid
    )
    enriched = dict(row)
    enriched["length_ratio"] = ratio
    enriched["invalid"] = invalid
    enriched["passed"] = passed
    enriched["delta_editlens"] = compute_delta(row["input_score"], row["output_score"])
    return enriched
