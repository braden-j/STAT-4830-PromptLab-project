"""Mutation operators for PromptSpec (one slot at a time, stochastic)."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .templates import PromptSpec

if TYPE_CHECKING:
    from random import Random


# Options for mutations
TONES = [
    "Neutral and direct.",
    "Professional but accessible.",
    "Clear and confident.",
    "Direct.",
    "Helpful and precise.",
    "Formal.",
    "Conversational but precise.",
]

ANTI_SLOP_LINES = [
    "Be precise and avoid generic filler.",
    "Do not use vague phrases like 'many factors' or 'in many ways'.",
    "Avoid filler phrases (e.g. 'you know', 'basically', 'kind of').",
    "Prefer direct statements over hedging.",
    "Do not use phrases like 'it goes without saying' or 'needless to say'.",
    "Justify claims concretely rather than with hand-waving.",
    "Do not hedge excessively (avoid 'might', 'perhaps', 'somewhat' unless truly uncertain).",
    "Use concrete examples instead of abstract generalities.",
    "Avoid generic phrases like 'and stuff' or 'things like that'.",
    "Prefer 'X causes Y' over 'X can often lead to Y' when the link is clear.",
    "Do not pad with unnecessary words.",
    "Use domain-specific language when appropriate.",
]

CONSTRAINT_LINES = [
    "Be concise.",
    "Stay on topic.",
    "Use concrete examples.",
    "Prefer short sentences.",
    "Do not pad with unnecessary words.",
    "Use domain-specific language when appropriate.",
    "Be specific.",
    "Avoid repetition.",
    "Use one or two concrete examples.",
    "Define terms when you introduce them.",
]

OUTPUT_FORMATS = [
    "Plain paragraphs.",
    "Short paragraphs or bullet points if helpful.",
    "Structured: introduction, body, conclusion.",
    "Numbered steps or short paragraphs.",
    "Short paragraphs; bullets for lists.",
]

REASONING_STYLES = [
    "State claims directly; support with concrete examples when needed.",
    "Lead with the main point; then justify.",
    "One idea per paragraph; no fluff.",
    "Get to the point quickly.",
    "Build from simple to specific.",
]


def _pick_mutation_target(spec: PromptSpec, rng: "Random") -> str:
    """Choose which slot to mutate (only non-empty or always allow task/role)."""
    candidates = []
    if spec.role:
        candidates.append("role")
    if spec.task:
        candidates.append("task")
    if spec.constraints:
        candidates.append("constraints")
    if spec.anti_slop:
        candidates.append("anti_slop")
    if spec.output_format:
        candidates.append("output_format")
    if spec.tone:
        candidates.append("tone")
    if spec.audience:
        candidates.append("audience")
    if spec.reasoning_style:
        candidates.append("reasoning_style")
    # Always allow adding to list slots or mutating core slots
    candidates.extend(["constraints", "anti_slop", "tone", "output_format", "reasoning_style"])
    return rng.choice(candidates)


def _mutate_constraints(spec: PromptSpec, rng: "Random", strength: str) -> None:
    if strength == "light":
        if spec.constraints and rng.random() < 0.5:
            spec.constraints.pop(rng.randint(0, len(spec.constraints) - 1))
        else:
            spec.constraints.append(rng.choice(CONSTRAINT_LINES))
    else:
        if spec.constraints and rng.random() < 0.4:
            spec.constraints.pop(rng.randint(0, len(spec.constraints) - 1))
        if rng.random() < 0.7:
            new_line = rng.choice(CONSTRAINT_LINES)
            if new_line not in spec.constraints:
                spec.constraints.append(new_line)


def _mutate_anti_slop(spec: PromptSpec, rng: "Random", strength: str) -> None:
    if strength == "light":
        if spec.anti_slop and rng.random() < 0.5:
            idx = rng.randint(0, len(spec.anti_slop) - 1)
            spec.anti_slop[idx] = rng.choice(ANTI_SLOP_LINES)
        else:
            new_line = rng.choice(ANTI_SLOP_LINES)
            if new_line not in spec.anti_slop:
                spec.anti_slop.append(new_line)
    else:
        if spec.anti_slop and rng.random() < 0.4:
            spec.anti_slop.pop(rng.randint(0, len(spec.anti_slop) - 1))
        new_line = rng.choice(ANTI_SLOP_LINES)
        if new_line not in spec.anti_slop:
            spec.anti_slop.append(new_line)


def _mutate_tone(spec: PromptSpec, rng: "Random") -> None:
    spec.tone = rng.choice(TONES)


def _mutate_output_format(spec: PromptSpec, rng: "Random") -> None:
    spec.output_format = rng.choice(OUTPUT_FORMATS)


def _mutate_reasoning_style(spec: PromptSpec, rng: "Random") -> None:
    spec.reasoning_style = rng.choice(REASONING_STYLES)


def _strengthen_specificity(spec: PromptSpec, rng: "Random") -> None:
    """Add or strengthen a specificity/concreteness instruction."""
    lines = [
        "Use concrete examples.",
        "Be specific; avoid vague generalizations.",
        "Justify claims concretely.",
    ]
    line = rng.choice(lines)
    if spec.anti_slop and line not in spec.anti_slop:
        spec.anti_slop.append(line)
    elif not spec.anti_slop:
        spec.anti_slop = [line]


def _strengthen_brevity(spec: PromptSpec, rng: "Random") -> None:
    """Add or strengthen brevity requirement."""
    if "Be concise." not in spec.constraints:
        spec.constraints.append("Be concise.")
    if "Do not pad with unnecessary words." not in spec.anti_slop:
        spec.anti_slop.append("Do not pad with unnecessary words.")


def _add_avoid_generic(spec: PromptSpec, rng: "Random") -> None:
    line = "Avoid generic phrases; use precise language."
    if line not in spec.anti_slop and not any("generic" in a.lower() for a in spec.anti_slop):
        spec.anti_slop.append(line)


def _add_justify_concretely(spec: PromptSpec, rng: "Random") -> None:
    line = "Justify claims concretely."
    if line not in spec.anti_slop and not any("justify" in a.lower() for a in spec.anti_slop):
        spec.anti_slop.append(line)


def mutate_spec(
    spec: PromptSpec,
    rng: "Random",
    mutation_strength: str = "medium",
) -> PromptSpec:
    """Return a new PromptSpec with one slot mutated. Stochastic but reproducible with same rng."""
    out = spec.copy()
    strength = mutation_strength if mutation_strength in ("light", "medium") else "medium"

    # With some probability, apply a "structural" mutation instead of slot pick
    if rng.random() < 0.25:
        op = rng.choice([
            _strengthen_specificity,
            _strengthen_brevity,
            _add_avoid_generic,
            _add_justify_concretely,
        ])
        op(out, rng)
        return out

    target = _pick_mutation_target(out, rng)
    if target == "role":
        out.role = "You are a clear and precise writer." if rng.random() < 0.5 else "You are an expert who explains concepts clearly."
    elif target == "task":
        # Slight variation; keep existing task if it's long
        if len(out.task) < 20 and rng.random() < 0.5:
            out.task = rng.choice(["Explain the given topic accurately.", "Answer the question or explain the topic.", "Write a short essay or explanation on the topic."])
    elif target == "constraints":
        _mutate_constraints(out, rng, strength)
    elif target == "anti_slop":
        _mutate_anti_slop(out, rng, strength)
    elif target == "output_format":
        _mutate_output_format(out, rng)
    elif target == "tone":
        _mutate_tone(out, rng)
    elif target == "audience":
        out.audience = rng.choice(["General reader.", "Educated non-specialist.", "Busy reader.", "Student or curious learner."])
    elif target == "reasoning_style":
        _mutate_reasoning_style(out, rng)
    return out
