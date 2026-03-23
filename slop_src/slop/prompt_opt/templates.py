"""Structured prompt representation and seed templates for prompt optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


STRUCTURE_PREFERENCE_VALUES = ("prose_preferred", "mixed", "list_friendly")


@dataclass
class PromptSpec:
    """Structured prompt with slots for role, task, constraints, anti-slop, etc."""

    role: str = ""
    task: str = ""
    constraints: list[str] = field(default_factory=list)
    anti_slop: list[str] = field(default_factory=list)
    output_format: str = ""
    tone: str = ""
    audience: str = ""
    reasoning_style: str = ""
    # Structure style: prose_preferred (default) | mixed | list_friendly
    structure_preference: str = "prose_preferred"

    def copy(self) -> "PromptSpec":
        return PromptSpec(
            role=self.role,
            task=self.task,
            constraints=list(self.constraints),
            anti_slop=list(self.anti_slop),
            output_format=self.output_format,
            tone=self.tone,
            audience=self.audience,
            reasoning_style=self.reasoning_style,
            structure_preference=self.structure_preference,
        )


RENDER_MODES = ("structured", "simple", "compact")


def render_prompt(spec: PromptSpec, mode: str = "structured") -> str:
    """Turn a PromptSpec into a single string for the generator.

    Modes:
    - structured: Role / Task / Constraints / Style sections (current behavior).
    - simple: Natural instruction paragraph (task + short sentences for constraints/style).
    - compact: Minimal single-block format (task + semicolon-separated constraints/style).
    """
    if mode not in RENDER_MODES:
        mode = "structured"

    if mode == "structured":
        return _render_structured(spec)
    if mode == "simple":
        return _render_simple(spec)
    return _render_compact(spec)


def _render_structured(spec: PromptSpec) -> str:
    """Current section-based format."""
    parts = []
    if spec.role:
        parts.append(f"Role: {spec.role}")
    if spec.task:
        parts.append(f"Task: {spec.task}")
    if spec.constraints:
        parts.append("Constraints:")
        for c in spec.constraints:
            parts.append(f"- {c}")
    if spec.anti_slop:
        parts.append("Style and clarity:")
        for a in spec.anti_slop:
            parts.append(f"- {a}")
    if spec.output_format:
        parts.append(f"Output format: {spec.output_format}")
    if spec.tone:
        parts.append(f"Tone: {spec.tone}")
    if spec.audience:
        parts.append(f"Audience: {spec.audience}")
    if spec.reasoning_style:
        parts.append(f"Reasoning: {spec.reasoning_style}")
    return "\n\n".join(parts).strip() or "Write clearly and precisely."


def _render_simple(spec: PromptSpec) -> str:
    """Natural instruction paragraph: task first, then constraints and style as short sentences."""
    sentences = []
    if spec.task:
        sentences.append(spec.task.rstrip(".") + ".")
    if spec.constraints:
        for c in spec.constraints:
            s = c.strip().rstrip(".")
            if s and not s[0].isupper():
                s = s[0].upper() + s[1:]
            sentences.append(s + "." if not s.endswith(".") else s)
    if spec.anti_slop:
        for a in spec.anti_slop:
            s = a.strip().rstrip(".")
            if s and not s[0].isupper():
                s = s[0].upper() + s[1:]
            sentences.append(s + "." if not s.endswith(".") else s)
    if spec.output_format:
        sentences.append(spec.output_format.rstrip(".") + ".")
    if spec.tone:
        sentences.append(spec.tone.rstrip(".") + ".")
    if spec.audience:
        sentences.append(f"Write for {spec.audience.lower().rstrip('.')}.")
    if spec.reasoning_style:
        sentences.append(spec.reasoning_style.rstrip(".") + ".")
    return " ".join(sentences).strip() or "Write clearly and precisely."


def _render_compact(spec: PromptSpec) -> str:
    """Minimal format: task line then semicolon-separated constraints and style."""
    parts = []
    if spec.task:
        parts.append(spec.task.strip())
    extras = []
    if spec.constraints:
        extras.append("Constraints: " + "; ".join(c.strip().rstrip(".") for c in spec.constraints))
    if spec.anti_slop:
        extras.append("Style: " + "; ".join(a.strip().rstrip(".") for a in spec.anti_slop))
    if spec.output_format:
        extras.append("Format: " + spec.output_format.strip().rstrip("."))
    if spec.tone:
        extras.append("Tone: " + spec.tone.strip().rstrip("."))
    if spec.audience:
        extras.append("Audience: " + spec.audience.strip().rstrip("."))
    if spec.reasoning_style:
        extras.append("Reasoning: " + spec.reasoning_style.strip().rstrip("."))
    if extras:
        parts.append(". ".join(extras))
    return " ".join(parts).strip() or "Write clearly and precisely."


def prompt_spec_to_dict(spec: PromptSpec) -> dict[str, Any]:
    """Serialize PromptSpec to a JSON-serializable dict."""
    return {
        "role": spec.role,
        "task": spec.task,
        "constraints": list(spec.constraints),
        "anti_slop": list(spec.anti_slop),
        "output_format": spec.output_format,
        "tone": spec.tone,
        "audience": spec.audience,
        "reasoning_style": spec.reasoning_style,
        "structure_preference": getattr(spec, "structure_preference", "prose_preferred"),
    }


def dict_to_prompt_spec(d: dict[str, Any]) -> PromptSpec:
    """Build PromptSpec from a dict (e.g. from JSON)."""
    pref = d.get("structure_preference", "prose_preferred")
    if pref not in STRUCTURE_PREFERENCE_VALUES:
        pref = "prose_preferred"
    return PromptSpec(
        role=d.get("role", ""),
        task=d.get("task", ""),
        constraints=list(d.get("constraints", [])),
        anti_slop=list(d.get("anti_slop", [])),
        output_format=d.get("output_format", ""),
        tone=d.get("tone", ""),
        audience=d.get("audience", ""),
        reasoning_style=d.get("reasoning_style", ""),
        structure_preference=pref,
    )


# Seed prompt templates for explanation / essay / QA generation
SEED_PROMPT_SPECS: list[PromptSpec] = [
    PromptSpec(
        role="You are a clear and precise writer.",
        task="Explain the given topic accurately.",
        constraints=["Be concise.", "Stay on topic."],
        anti_slop=[
            "Be precise and avoid generic filler.",
            "Do not use vague phrases like 'many factors' or 'in many ways'.",
        ],
        output_format="Plain paragraphs.",
        tone="Neutral and direct.",
        audience="General reader.",
        reasoning_style="State claims directly; support with concrete examples when needed.",
    ),
    PromptSpec(
        role="You are an expert who explains concepts clearly.",
        task="Answer the question or explain the topic.",
        constraints=["Use concrete examples.", "Prefer short sentences."],
        anti_slop=[
            "Avoid filler phrases (e.g. 'you know', 'basically', 'kind of').",
            "Prefer direct statements over hedging.",
        ],
        output_format="Short paragraphs or bullet points if helpful.",
        tone="Professional but accessible.",
        audience="Educated non-specialist.",
        reasoning_style="Lead with the main point; then justify.",
    ),
    PromptSpec(
        role="You are a writer who values clarity.",
        task="Write a short essay or explanation on the topic.",
        constraints=[
            "Use domain-specific language when appropriate.",
            "Do not pad with unnecessary words.",
        ],
        anti_slop=[
            "Do not use phrases like 'it goes without saying' or 'needless to say'.",
            "Justify claims concretely rather than with hand-waving.",
        ],
        output_format="Structured: introduction, body, conclusion.",
        tone="Clear and confident.",
        audience="Reader who wants substance.",
        reasoning_style="One idea per paragraph; no fluff.",
    ),
    PromptSpec(
        role="You are a precise communicator.",
        task="Explain or answer as requested.",
        constraints=["Be specific.", "Avoid repetition."],
        anti_slop=[
            "Do not hedge excessively (avoid 'might', 'perhaps', 'somewhat' unless truly uncertain).",
            "Use concrete examples instead of abstract generalities.",
        ],
        output_format="Numbered steps or short paragraphs.",
        tone="Direct.",
        audience="Busy reader.",
        reasoning_style="Get to the point quickly.",
    ),
    PromptSpec(
        role="You are a teacher who explains clearly.",
        task="Teach the concept or answer the question.",
        constraints=[
            "Use one or two concrete examples.",
            "Define terms when you introduce them.",
        ],
        anti_slop=[
            "Avoid generic phrases like 'and stuff' or 'things like that'.",
            "Prefer 'X causes Y' over 'X can often lead to Y' when the link is clear.",
        ],
        output_format="Short paragraphs; bullets for lists.",
        tone="Helpful and precise.",
        audience="Student or curious learner.",
        reasoning_style="Build from simple to specific.",
    ),
]


def get_seeds_for_task(task_instruction: str) -> list[PromptSpec]:
    """Return seed PromptSpecs with task slot set to the given instruction."""
    seeds = []
    for s in SEED_PROMPT_SPECS:
        spec = s.copy()
        spec.task = task_instruction
        seeds.append(spec)
    return seeds
