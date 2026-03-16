"""Anti-reward-hacking diagnostics: expose metrics without blocking scoring.

Includes structural punctuation metrics to detect bullet-heavy, list-heavy,
or visually noisy formatting that may reward-hack without improving prose.
"""

from __future__ import annotations

import re
from typing import Any


# ---- Structural punctuation (lightweight, general heuristics) ----


def bullet_like_line_ratio(text: str) -> float:
    """Fraction of non-empty lines that look like bullet/list items (start with -, *, •, ·, or N.)."""
    lines = [L.strip() for L in text.splitlines() if L.strip()]
    if not lines:
        return 0.0
    count = 0
    for line in lines:
        if re.match(r"^[\-\*•·]\s", line) or re.match(r"^\d+[\.\)]\s", line):
            count += 1
    return count / len(lines)


def repeated_dash_ratio(text: str) -> float:
    """Fraction of characters that are part of repeated dashes (2+ consecutive - or —)."""
    if not text:
        return 0.0
    dash_runs = len(re.findall(r"-{2,}|—{2,}", text))
    # Approximate chars: each run contributes at least 2
    dash_chars = sum(len(m) for m in re.findall(r"-{2,}|—+", text))
    return min(1.0, dash_chars / max(len(text), 1))


def repeated_quote_ratio(text: str) -> float:
    """Fraction of characters that are repeated quotes ("" or '')."""
    if not text:
        return 0.0
    double = len(re.findall(r'""', text)) * 2
    single = len(re.findall(r"''", text)) * 2
    return min(1.0, (double + single) / max(len(text), 1))


def tilde_ratio(text: str) -> float:
    """Fraction of characters that are tildes."""
    if not text:
        return 0.0
    return text.count("~") / max(len(text), 1)


def list_marker_ratio(text: str) -> float:
    """Ratio of list-like markers (bullet, numbered, or colon-after-number) to non-empty lines."""
    lines = [L.strip() for L in text.splitlines() if L.strip()]
    if not lines:
        return 0.0
    count = 0
    for line in lines:
        if re.match(r"^[\-\*•·]\s", line) or re.match(r"^\d+[\.\)]\s", line) or re.match(r"^\d+\.?\s*[A-Za-z]", line):
            count += 1
    return count / len(lines)


def line_start_symbol_ratio(text: str) -> float:
    """Fraction of non-empty lines that start with a non-letter (symbol or digit)."""
    lines = [L.strip() for L in text.splitlines() if L.strip()]
    if not lines:
        return 0.0
    count = sum(1 for L in lines if L and not L[0].isalpha())
    return count / len(lines)


def abnormal_punctuation_density(text: str) -> float:
    """Combined score for structurally noisy punctuation (smooth, 0..1). Mild usage is low."""
    lines = [L.strip() for L in text.splitlines() if L.strip()]
    n_lines = max(len(lines), 1)
    bullet = bullet_like_line_ratio(text)
    dash = repeated_dash_ratio(text)
    quote = repeated_quote_ratio(text)
    tilde = min(1.0, tilde_ratio(text) * 20)  # scale so rare tildes don't dominate
    line_sym = line_start_symbol_ratio(text)
    # Weight: bullets and line-start symbols are strong signals; dashes/quotes/tilde milder
    combined = 0.35 * bullet + 0.15 * dash + 0.10 * quote + 0.10 * tilde + 0.30 * line_sym
    return min(1.0, combined)


def compute_structural_diagnostics(text: str) -> dict[str, float]:
    """Structural punctuation metrics only. Does not block scoring."""
    return {
        "bullet_like_line_ratio": bullet_like_line_ratio(text),
        "repeated_dash_ratio": repeated_dash_ratio(text),
        "repeated_quote_ratio": repeated_quote_ratio(text),
        "tilde_ratio": tilde_ratio(text),
        "list_marker_ratio": list_marker_ratio(text),
        "line_start_symbol_ratio": line_start_symbol_ratio(text),
        "abnormal_punctuation_density": abnormal_punctuation_density(text),
    }


def repetition_ratio(text: str, n: int = 2) -> float:
    """Fraction of tokens that are part of a repeated n-gram (n tokens repeated)."""
    tokens = text.split()
    if len(tokens) < n * 2:
        return 0.0
    seen = set()
    repeat_count = 0
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i : i + n])
        if ng in seen:
            repeat_count += n
        else:
            seen.add(ng)
    return repeat_count / max(len(tokens), 1)


def repeated_token_fraction(tokens: list[str]) -> float:
    """Fraction of tokens that are exact duplicates of a previous token."""
    if not tokens:
        return 0.0
    seen = set()
    dup = 0
    for t in tokens:
        if t in seen:
            dup += 1
        else:
            seen.add(t)
    return dup / len(tokens)


def punctuation_ratio(text: str) -> float:
    """Fraction of characters that are punctuation."""
    if not text:
        return 0.0
    punct = sum(1 for c in text if c in ".,;:!?\"'()-[]{}")
    return punct / len(text)


def caps_ratio(text: str) -> float:
    """Fraction of letters that are uppercase."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def filler_loop_score(text: str, filler_phrases: list[str] | None = None) -> float:
    """Crude score for repeated filler-like phrases (e.g. 'like like like')."""
    if filler_phrases is None:
        filler_phrases = ["like", "you know", "um", "uh", "well", "so", "actually"]
    text_lower = text.lower()
    score = 0.0
    for phrase in filler_phrases:
        # Count repeated adjacent occurrences
        pattern = r"(\b" + re.escape(phrase) + r"\b\s*){2,}"
        for m in re.finditer(pattern, text_lower):
            score += min(1.0, len(m.group(0).split()) / 5.0)
    return min(1.0, score)


# ---- Semantic anti-reward-hacking (meta-instructional, off-task, prompt copy) ----

# Phrases that suggest the model is echoing instructions or giving writing advice instead of answering
INSTRUCTION_ECHO_PHRASES = [
    r"write\s+(in|with|clearly|using)",
    r"use\s+short\s+sentences",
    r"be\s+concise",
    r"avoid\s+vague",
    r"use\s+concrete\s+examples",
    r"stay\s+on\s+topic",
    r"plain\s+paragraphs",
    r"direct\s+and\s+specific",
    r"do\s+not\s+use",
    r"prefer\s+direct",
    r"output\s+format",
    r"constraints?\s*:",
    r"style\s+and\s+clarity",
]

WRITING_ADVICE_PHRASES = [
    r"you\s+should",
    r"it'?s\s+important\s+to",
    r"make\s+sure\s+to",
    r"try\s+to\s+(write|use|avoid|be)",
    r"remember\s+to",
    r"focus\s+on\s+(being|writing|using)",
    r"aim\s+to\s+(be|write|provide)",
    r"ensure\s+that\s+(your|you)",
    r"when\s+writing",
    r"in\s+your\s+(response|answer|writing|explanation)",
    r"when\s+explaining",
    r"a\s+good\s+(explanation|answer)\s+(is|should)",
    r"to\s+write\s+(clearly|well)",
    r"best\s+practice\s+is",
]

OFF_TASK_GENERIC_PHRASES = [
    r"good\s+writing",
    r"clear\s+communication",
    r"effective\s+explanation",
    r"the\s+key\s+is\s+to",
    r"best\s+approach\s+is",
    r"general\s+rule",
    r"in\s+general\s*[,.]",
    r"writing\s+tip",
    r"how\s+to\s+write",
    r"how\s+to\s+explain",
    r"tips?\s+for\s+(writing|explaining)",
]


def instruction_echo_ratio(text: str) -> float:
    """Fraction of sentences/lines that contain instruction-like or rubric-like phrasing (0..1)."""
    if not text or not text.strip():
        return 0.0
    text_lower = text.lower()
    hits = 0
    for pat in INSTRUCTION_ECHO_PHRASES:
        if re.search(pat, text_lower, re.IGNORECASE):
            hits += 1
    return min(1.0, hits / max(len(INSTRUCTION_ECHO_PHRASES) * 0.2, 1))


def writing_advice_ratio(text: str) -> float:
    """Fraction of writing-advice / rubric language detected (0..1)."""
    if not text or not text.strip():
        return 0.0
    text_lower = text.lower()
    hits = 0
    for pat in WRITING_ADVICE_PHRASES:
        if re.search(pat, text_lower, re.IGNORECASE):
            hits += 1
    return min(1.0, hits / max(len(WRITING_ADVICE_PHRASES) * 0.25, 1))


def prompt_copy_ratio(output: str, prompt: str) -> float:
    """Fraction of output token types that appear in the prompt (high = echoing prompt)."""
    if not output or not output.strip():
        return 0.0
    out_tokens = set(re.findall(r"\b\w+\b", output.lower()))
    prompt_tokens = set(re.findall(r"\b\w+\b", prompt.lower()))
    if not out_tokens:
        return 0.0
    overlap = len(out_tokens & prompt_tokens) / len(out_tokens)
    return min(1.0, overlap * 1.5)  # slight scale so moderate overlap gives <1


def off_task_generic_ratio(text: str) -> float:
    """Fraction of off-task generic guidance phrases detected (0..1)."""
    if not text or not text.strip():
        return 0.0
    text_lower = text.lower()
    hits = 0
    for pat in OFF_TASK_GENERIC_PHRASES:
        if re.search(pat, text_lower, re.IGNORECASE):
            hits += 1
    return min(1.0, hits / max(len(OFF_TASK_GENERIC_PHRASES) * 0.3, 1))


def task_relevance_score(text: str, task_keywords: list[str]) -> float:
    """Lightweight task relevance: fraction of task_keywords that appear in text (0..1). Empty keywords => 1.0."""
    if not task_keywords:
        return 1.0
    if not text or not text.strip():
        return 0.0
    text_lower = text.lower()
    found = sum(1 for kw in task_keywords if kw.strip() and kw.lower() in text_lower)
    return found / len(task_keywords)


def compute_semantic_diagnostics(
    text: str,
    prompt_text: str | None = None,
    task_keywords: list[str] | None = None,
) -> dict[str, float]:
    """Semantic anti-hacking metrics: instruction echo, writing advice, prompt copy, off-task, task relevance."""
    out = {
        "instruction_echo_ratio": instruction_echo_ratio(text),
        "writing_advice_ratio": writing_advice_ratio(text),
        "off_task_generic_ratio": off_task_generic_ratio(text),
    }
    if prompt_text is not None:
        out["prompt_copy_ratio"] = prompt_copy_ratio(text, prompt_text)
    else:
        out["prompt_copy_ratio"] = 0.0
    kw = task_keywords if task_keywords is not None else []
    out["task_relevance_score"] = task_relevance_score(text, kw)
    # Combined semantic "bad" score (high = more meta/off-task)
    meta = (out["instruction_echo_ratio"] + out["writing_advice_ratio"] + out["off_task_generic_ratio"]) / 3.0
    out["semantic_meta_score"] = min(1.0, meta + 0.3 * out["prompt_copy_ratio"])
    out["semantic_off_task_score"] = min(1.0, (1.0 - out["task_relevance_score"]) + 0.5 * out["off_task_generic_ratio"])
    return out


def compute_diagnostics(
    text: str,
    token_count: int | None = None,
    filler_phrases: list[str] | None = None,
) -> dict[str, Any]:
    """Compute all diagnostics for a single text. Does not block scoring."""
    tokens = text.split()
    n_tok = token_count if token_count is not None else len(tokens)
    structural = compute_structural_diagnostics(text)
    return {
        "length_chars": len(text),
        "length_tokens": n_tok,
        "very_short": n_tok < 3,
        "repetition_ratio": repetition_ratio(text, n=2),
        "repeated_token_fraction": repeated_token_fraction(tokens),
        "punctuation_ratio": punctuation_ratio(text),
        "caps_ratio": caps_ratio(text),
        "filler_loop_score": filler_loop_score(text, filler_phrases),
        **structural,
    }
