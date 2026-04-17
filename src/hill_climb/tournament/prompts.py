"""Prompt formatting for tournament rewrite models."""

from __future__ import annotations

REWRITE_PROMPT = (
    "Rewrite the following text so it sounds more natural and human while preserving "
    "its meaning and approximate length.\n\n"
    "Original text:\n{text}\n\n"
    "Rewritten text:\n"
)


def format_rewrite_prompt(text: str) -> str:
    """Format the shared rewrite instruction prompt."""

    return REWRITE_PROMPT.format(text=text.strip())
