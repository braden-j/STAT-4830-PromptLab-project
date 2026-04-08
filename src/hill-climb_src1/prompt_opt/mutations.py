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

# Prose-oriented formats (preferred when structure_preference is prose_preferred)
PROSE_OUTPUT_FORMATS = [
    "Plain paragraphs.",
    "Short paragraphs; avoid bullet lists.",
    "Write in continuous prose with short paragraphs.",
    "Use short paragraphs only; no lists or bullets.",
]

REASONING_STYLES = [
    "State claims directly; support with concrete examples when needed.",
    "Lead with the main point; then justify.",
    "One idea per paragraph; no fluff.",
    "Get to the point quickly.",
    "Build from simple to specific.",
]


# ---- Semantic mutation templates (PromptSpec-level; no raw prompt string edits) ----

# These templates are intentionally short and reusable to avoid brittle giant prompt blobs.
CONSTRAINT_SEMANTIC_TEMPLATES = [
    "Explain the main idea in direct language with one concrete example.",
    "Use only the detail needed to fully answer the task.",
    "Avoid generic writing advice; focus on the specific question.",
    "Define key terms briefly, then apply them to the task.",
]

ANTI_SLOP_SEMANTIC_TEMPLATES = [
    "Avoid talking about how to write; focus on answering the task itself.",
    "Do not repeat the instructions; answer as if the task is already understood.",
    "Avoid long lists or decorative formatting unless essential to the explanation.",
    "Avoid rubric language like 'be concise' or 'use short sentences' in the output.",
]

PROSE_FORMAT_TEMPLATES = [
    "Write in short, coherent paragraphs rather than bullet lists.",
    "Use 1–3 short paragraphs with one concrete example; avoid list-heavy formatting.",
]

MIXED_FORMAT_TEMPLATES = [
    "Use short paragraphs; bullets only if they make the explanation clearer (keep bullets minimal).",
    "Write mostly prose; a short bullet list is allowed if it adds clarity.",
]

LIST_FRIENDLY_FORMAT_TEMPLATES = [
    "Combine a brief paragraph with a short, focused bullet list (no long lists).",
    "You may use bullets sparingly, but keep the explanation primarily in prose.",
]

REASONING_STYLE_TEMPLATES = [
    "Answer directly, then briefly justify with one concrete example.",
    "Focus on concrete explanation rather than abstract writing advice.",
    "Reason step by step internally, but present only the final explanation.",
]


def mutate_constraints_semantically(spec: PromptSpec, rng: "Random") -> None:
    """Semantic constraint mutation: tighten vague constraints and add concrete, task-helpful constraints."""
    replacements = {
        "Be concise.": "Use only the detail needed to fully answer the task.",
        "Be specific.": "Explain the main idea in direct language with one concrete example.",
        "Stay on topic.": "Avoid generic writing advice; focus on the specific question.",
    }
    # Prefer rewriting existing vague constraints
    for i, c in enumerate(list(spec.constraints)):
        if c in replacements and rng.random() < 0.7:
            spec.constraints[i] = replacements[c]
            return
    # Otherwise add a constraint (avoid duplicates)
    candidate = rng.choice(CONSTRAINT_SEMANTIC_TEMPLATES)
    if candidate not in spec.constraints:
        spec.constraints.append(candidate)


def mutate_anti_slop_semantically(spec: PromptSpec, rng: "Random") -> None:
    """Semantic anti_slop mutation: add anti-meta/off-task guidance in natural language."""
    candidate = rng.choice(ANTI_SLOP_SEMANTIC_TEMPLATES)
    if spec.anti_slop and rng.random() < 0.6:
        idx = rng.randint(0, len(spec.anti_slop) - 1)
        spec.anti_slop[idx] = candidate
    else:
        if candidate not in spec.anti_slop:
            spec.anti_slop.append(candidate)


def mutate_output_format_semantically(spec: PromptSpec, rng: "Random") -> None:
    """Semantic output_format mutation: align formatting guidance with structure_preference."""
    pref = getattr(spec, "structure_preference", "prose_preferred")
    if pref == "prose_preferred":
        spec.output_format = rng.choice(PROSE_FORMAT_TEMPLATES)
    elif pref == "mixed":
        spec.output_format = rng.choice(MIXED_FORMAT_TEMPLATES)
    else:  # list_friendly
        spec.output_format = rng.choice(LIST_FRIENDLY_FORMAT_TEMPLATES)


def mutate_reasoning_style_semantically(spec: PromptSpec, rng: "Random") -> None:
    """Semantic reasoning_style mutation: encourage direct answers + concrete justification."""
    spec.reasoning_style = rng.choice(REASONING_STYLE_TEMPLATES)


def _apply_semantic_mutation(spec: PromptSpec, rng: "Random", mutation_info: dict | None = None) -> None:
    """Apply one semantic mutation op to a PromptSpec (field-level).

    If mutation_info is provided, it will be populated with:
    - mutation_type: \"semantic\"
    - mutation_helper: name of the semantic helper function that fired.
    """
    ops = [
        mutate_constraints_semantically,
        mutate_anti_slop_semantically,
        mutate_output_format_semantically,
        mutate_reasoning_style_semantically,
    ]
    op = rng.choice(ops)
    if mutation_info is not None:
        mutation_info["mutation_type"] = "semantic"
        mutation_info["mutation_helper"] = op.__name__
    op(spec, rng)


def _pick_mutation_target(spec: PromptSpec, rng: "Random") -> str:
    """Choose which slot to mutate. When prose_preferred, downweight output_format."""
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
    candidates.extend(["constraints", "anti_slop", "tone", "output_format", "reasoning_style"])
    pref = getattr(spec, "structure_preference", "prose_preferred")
    if pref == "prose_preferred" and "output_format" in candidates:
        # Reduce chance of mutating output_format: remove 2 of its entries so it's less likely
        for _ in range(2):
            if "output_format" in candidates:
                candidates.remove("output_format")
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
    pref = getattr(spec, "structure_preference", "prose_preferred")
    if pref == "prose_preferred" and rng.random() < 0.7:
        spec.output_format = rng.choice(PROSE_OUTPUT_FORMATS)
    else:
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
    semantic_mutation_probability: float = 0.0,
    mutation_info: dict | None = None,
) -> PromptSpec:
    """Return a new PromptSpec with one mutation applied.

    Mutation types:
    - semantic mutation (field-level, meaning-changing) with configurable probability
    - existing structural mutation (specificity/brevity/etc.)
    - slot mutation (role/constraints/anti_slop/output_format/etc.)
    """
    out = spec.copy()
    # Optional metadata channel for callers that want to log mutation provenance.
    # When provided, this dict will be cleared and then populated with:
    # - mutation_type: \"semantic\" | \"structural\" | \"slot\"
    # - mutation_helper: helper name or None
    if mutation_info is not None:
        mutation_info.clear()
        mutation_info["mutation_type"] = "slot"
        mutation_info["mutation_helper"] = None
    strength = mutation_strength if mutation_strength in ("light", "medium") else "medium"

    # Semantic mutation: higher-level edits that improve intent while staying within PromptSpec abstraction.
    if semantic_mutation_probability > 0 and rng.random() < semantic_mutation_probability:
        _apply_semantic_mutation(out, rng, mutation_info=mutation_info)
        return out

    # With some probability, apply a "structural" mutation instead of slot pick
    if rng.random() < 0.25:
        op = rng.choice([
            _strengthen_specificity,
            _strengthen_brevity,
            _add_avoid_generic,
            _add_justify_concretely,
        ])
        if mutation_info is not None:
            mutation_info["mutation_type"] = "structural"
            mutation_info["mutation_helper"] = op.__name__
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
