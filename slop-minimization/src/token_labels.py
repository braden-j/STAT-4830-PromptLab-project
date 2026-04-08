"""Token-level labeling for slop detection from (human_text, slop_text) pairs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

DEFAULT_SLOP_PHRASES = [
    "you know",
    "kind of",
    "basically",
    "i mean",
    "like",
    "um",
    "uh",
    "sort of",
    "you see",
    "as it were",
]


def detect_sloppy_spans(
    text: str,
    phrase_list: list[str] | None = None,
    use_repetition: bool = True,
    use_distinct2: bool = True,
    distinct2_window: int = 10,
    distinct2_threshold: float = 0.5,
) -> list[tuple[int, int]]:
    """Detect character-level sloppy spans in text.

    Uses three heuristics: generic phrase list, repetition spans, and
    low distinct-2 windows. Returns union of all spans as (start, end) char offsets.
    """
    phrase_list = phrase_list or DEFAULT_SLOP_PHRASES
    spans: set[tuple[int, int]] = set()
    text_lower = text.lower()

    # Phrase list: case-insensitive matches
    for phrase in phrase_list:
        start = 0
        while True:
            idx = text_lower.find(phrase.lower(), start)
            if idx < 0:
                break
            spans.add((idx, idx + len(phrase)))
            start = idx + 1

    # Repetition: consecutive repeated words
    words = text.split()
    if use_repetition and len(words) >= 2:
        i = 0
        while i < len(words):
            w = words[i]
            j = i + 1
            while j < len(words) and words[j].lower() == w.lower():
                j += 1
            if j > i + 1:
                # Found run of repeated words; get char span (start inclusive, end exclusive)
                start_char = len(" ".join(words[:i]))
                end_char = len(" ".join(words[:j]))
                end_char = min(end_char, len(text))
                spans.add((start_char, end_char))
            i = j

    # Distinct-2: sliding window where distinct bigrams / max_bigrams < threshold
    if use_distinct2 and len(words) >= distinct2_window:
        bigrams = list(zip(words[:-1], words[1:]))
        max_bigrams = distinct2_window - 1 if distinct2_window > 1 else 1
        for start in range(max(0, len(bigrams) - distinct2_window + 2)):
            window_bigrams = bigrams[start : start + distinct2_window - 1]
            distinct = len(set((a.lower(), b.lower()) for a, b in window_bigrams))
            ratio = distinct / max(max_bigrams, 1)
            if ratio < distinct2_threshold:
                start_char = len(" ".join(words[: start + 1]))
                end_word_idx = min(start + distinct2_window, len(words))
                end_char = len(" ".join(words[:end_word_idx]))
                end_char = min(end_char, len(text))
                spans.add((start_char, end_char))

    return sorted(spans, key=lambda s: s[0])


def spans_to_token_labels(
    offset_mapping: list[tuple[int, int]],
    special_tokens_mask: list[int],
    spans: list[tuple[int, int]],
    label_pad_token_id: int = -100,
) -> list[int]:
    """Map character spans to per-token labels. Special tokens get -100."""
    labels: list[int] = []
    for i, (start, end) in enumerate(offset_mapping):
        if special_tokens_mask[i]:
            labels.append(label_pad_token_id)
            continue
        token_mid = (start + end) // 2 if end > start else start
        in_span = any(sstart <= token_mid < send for sstart, send in spans)
        labels.append(1 if in_span else 0)
    return labels


def _chunk_sequence(
    input_ids: list[int],
    token_labels: list[int],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    stride: int,
    label_pad_token_id: int = -100,
) -> list[dict[str, Any]]:
    """Chunk token sequence with stride; pad last chunk; return list of dicts."""
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    chunks: list[dict[str, Any]] = []
    pos = 0
    n = len(input_ids)
    while pos < n:
        chunk_ids = input_ids[pos : pos + max_length]
        chunk_labels = token_labels[pos : pos + max_length]
        attn = [1] * len(chunk_ids)
        if len(chunk_ids) < max_length:
            pad_len = max_length - len(chunk_ids)
            chunk_ids = chunk_ids + [pad_id] * pad_len
            chunk_labels = chunk_labels + [label_pad_token_id] * pad_len
            attn = attn + [0] * pad_len
        chunks.append({
            "input_ids": chunk_ids,
            "attention_mask": attn,
            "token_labels": chunk_labels,
        })
        if pos + max_length >= n:
            break
        pos += stride
    return chunks


def _tokenize_with_offsets(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> tuple[list[int], list[tuple[int, int]], list[int]]:
    """Tokenize text and return (input_ids, offset_mapping, special_tokens_mask)."""
    enc = tokenizer(
        text,
        add_special_tokens=True,
        return_attention_mask=False,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=False,
    )
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    special = enc["special_tokens_mask"]
    return ids, offsets, special


def build_token_label_examples(
    pairs: list[tuple[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    stride: int | None = None,
    label_mode: Literal["document", "span_heuristic"] = "document",
    phrase_list: list[str] | None = None,
    distinct2_window: int = 10,
    distinct2_threshold: float = 0.5,
    use_repetition: bool = True,
    use_distinct2: bool = True,
    label_pad_token_id: int = -100,
) -> list[dict[str, Any]]:
    """Build (input_ids, attention_mask, token_labels) from (human_text, slop_text) pairs.

    label_mode:
        - "document": all tokens in human_text = 0, all in slop_text = 1.
        - "span_heuristic": within slop_text only, detect sloppy spans and label those 1, others 0.

    Chunking uses max_length and stride (default stride = max_length for no overlap,
    or max_length // 2 for 50% overlap). Long texts are chunked; special/pad tokens get -100.
    """
    stride = stride if stride is not None else max_length
    phrase_list = phrase_list or DEFAULT_SLOP_PHRASES
    results: list[dict[str, Any]] = []

    for human_text, slop_text in pairs:
        if not human_text.strip() and not slop_text.strip():
            continue

        # Option 1: document-level
        if label_mode == "document":
            for text, label_val in [(human_text, 0), (slop_text, 1)]:
                if not text.strip():
                    continue
                ids, offsets, special = _tokenize_with_offsets(tokenizer, text)
                token_labels = []
                for i, (start, end) in enumerate(offsets):
                    if special[i]:
                        token_labels.append(label_pad_token_id)
                    else:
                        token_labels.append(label_val)
                chunks = _chunk_sequence(ids, token_labels, tokenizer, max_length, stride, label_pad_token_id)
                results.extend(chunks)
            continue

        # Option 2: span_heuristic (only slop_text; human_text still all 0)
        if human_text.strip():
            ids_h, offsets_h, special_h = _tokenize_with_offsets(tokenizer, human_text)
            token_labels_h = [
                label_pad_token_id if special_h[i] else 0
                for i in range(len(ids_h))
            ]
            chunks_h = _chunk_sequence(ids_h, token_labels_h, tokenizer, max_length, stride, label_pad_token_id)
            results.extend(chunks_h)

        if not slop_text.strip():
            continue
        spans = detect_sloppy_spans(
            slop_text,
            phrase_list=phrase_list,
            use_repetition=use_repetition,
            use_distinct2=use_distinct2,
            distinct2_window=distinct2_window,
            distinct2_threshold=distinct2_threshold,
        )
        ids_s, offsets_s, special_s = _tokenize_with_offsets(tokenizer, slop_text)
        token_labels_s = spans_to_token_labels(offsets_s, special_s, spans, label_pad_token_id)
        chunks_s = _chunk_sequence(ids_s, token_labels_s, tokenizer, max_length, stride, label_pad_token_id)
        results.extend(chunks_s)

    return results
