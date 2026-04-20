"""Source-pool loading and pair-building helpers for the tournament."""

from __future__ import annotations

import json
import random
import re
import urllib.request
from pathlib import Path
from typing import Any

from hill_climb.slop_gen import RuleSloppifier
from hill_climb.tournament.io import read_csv_dicts

DEFAN_URL = (
    "https://raw.githubusercontent.com/ashikiut/DefAn/main/"
    "DefAn-Public%20Combined/DefAn_public_combined.json"
)


def normalize_text(text: Any) -> str:
    """Normalize potentially nested text fields to a single clean string."""

    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        text = " ".join(str(x) for x in text if x is not None)
    elif isinstance(text, dict):
        text = json.dumps(text, ensure_ascii=False)
    else:
        text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def trim_word_window(text: str, min_words: int, max_words: int) -> str:
    """Keep texts in a consistent length window."""

    words = text.split()
    if len(words) < min_words:
        return ""
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text


def download_defan(path: str | Path) -> Path:
    """Download the public DefAn JSON if needed."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(DEFAN_URL, path)
    return path


def load_kaggle_essays(path: str | Path, min_words: int, max_words: int) -> list[dict[str, Any]]:
    """Load Kaggle essays from the fixed CSV path."""

    path = Path(path)
    if not path.exists():
        return []
    rows = read_csv_dicts(path)
    if not rows:
        return []
    preferred_cols = [
        "text",
        "essay",
        "content",
        "Essay",
        "essay_text",
        "full_text",
    ]
    text_col = next((col for col in preferred_cols if col in rows[0]), None)
    if text_col is None:
        text_col = next(iter(rows[0]))
    out = []
    for idx, row in enumerate(rows):
        text = trim_word_window(normalize_text(row.get(text_col)), min_words, max_words)
        if text:
            out.append(
                {
                    "id": f"kaggle-{idx}",
                    "domain": "essay",
                    "source_dataset": "aeon_essays",
                    "text": text,
                    "word_count": len(text.split()),
                }
            )
    return out


def load_defan_records(path: str | Path, min_words: int, max_words: int) -> list[dict[str, Any]]:
    """Load DefAn and render it into QA text suitable for the source pool."""

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for idx, row in enumerate(raw):
        q = normalize_text(row.get("questions") or row.get("question"))
        a = normalize_text(row.get("answer"))
        rendered = trim_word_window(f"Question: {q}\nAnswer: {a}", min_words, max_words)
        if rendered:
            out.append(
                {
                    "id": f"defan-{idx}",
                    "domain": "qa",
                    "source_dataset": "defan_public",
                    "text": rendered,
                    "word_count": len(rendered.split()),
                }
            )
    return out


def sloppify_candidates(
    text: str,
    seed: int,
    include_prompt_mode: bool = False,
) -> list[tuple[str, str]]:
    """Generate named sloppified candidates from a clean target."""

    easy = RuleSloppifier.from_difficulty("easy", seed=seed).sloppify(text)
    medium = RuleSloppifier.from_difficulty("medium", seed=seed + 1).sloppify(text)
    hard = RuleSloppifier.from_difficulty("hard", seed=seed + 2).sloppify(text)
    easy_then_hard = RuleSloppifier.from_difficulty("hard", seed=seed + 3).sloppify(easy)
    easy_then_medium = RuleSloppifier.from_difficulty("medium", seed=seed + 4).sloppify(easy)
    full_stack = RuleSloppifier.from_difficulty("hard", seed=seed + 5).sloppify(
        RuleSloppifier.from_difficulty("medium", seed=seed + 6).sloppify(
            RuleSloppifier.from_difficulty("easy", seed=seed + 7).sloppify(text)
        )
    )
    modes = [
        ("rule_easy", easy),
        ("rule_medium", medium),
        ("rule_hard", hard),
        ("rule_easy_then_hard", easy_then_hard),
        ("rule_easy_then_medium", easy_then_medium),
        ("rule_full_stack", full_stack),
    ]
    if include_prompt_mode:
        modes.append(("prompt_stub", text))
    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    original = normalize_text(text)
    for mode_name, candidate in modes:
        candidate = normalize_text(candidate)
        if not candidate or candidate == original or candidate in seen:
            continue
        seen.add(candidate)
        deduped.append((mode_name, candidate))
    return deduped


def choose_split_counts(total: int, train: int, val: int, test: int) -> tuple[int, int, int]:
    """Clamp split sizes when a source underfills its target count."""

    desired = [train, val, test]
    if total >= sum(desired):
        return tuple(desired)
    train_count = min(train, total)
    remaining = max(total - train_count, 0)
    val_count = min(val, remaining)
    remaining = max(remaining - val_count, 0)
    test_count = min(test, remaining)
    return train_count, val_count, test_count


def assign_splits(rows: list[dict[str, Any]], train: int, val: int, test: int, seed: int) -> list[dict[str, Any]]:
    """Shuffle rows and stamp a split column."""

    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    train_count, val_count, test_count = choose_split_counts(len(rows), train, val, test)
    out = []
    for idx, row in enumerate(rows):
        stamped = dict(row)
        if idx < train_count:
            stamped["split"] = "train"
        elif idx < train_count + val_count:
            stamped["split"] = "val"
        elif idx < train_count + val_count + test_count:
            stamped["split"] = "test"
        else:
            continue
        out.append(stamped)
    return out
