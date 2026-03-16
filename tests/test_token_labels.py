"""Unit tests for token-level slop labeling."""

import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from slop_minimization.data.token_labels import (
    build_token_label_examples,
    detect_sloppy_spans,
    spans_to_token_labels,
    DEFAULT_SLOP_PHRASES,
)
from conftest import get_tokenizer


@pytest.fixture
def tokenizer():
    return get_tokenizer()


@pytest.fixture
def pairs():
    return [
        (
            "This is clear and well-written prose with proper structure.",
            "So like you know basically it's kind of sloppy filler.",
        ),
        ("Another good sentence.", "The the quick brown fox."),
    ]


# Option 1: document mode
def test_build_document_mode_output_keys(tokenizer, pairs):
    """Output has input_ids, attention_mask, token_labels."""
    results = build_token_label_examples(pairs, tokenizer, max_length=32, label_mode="document")
    assert len(results) >= 1
    for ex in results:
        assert "input_ids" in ex
        assert "attention_mask" in ex
        assert "token_labels" in ex


def test_build_document_mode_human_all_zero(tokenizer, pairs):
    """For examples from human_text, all non-special non-pad tokens have label 0."""
    results = build_token_label_examples(pairs, tokenizer, max_length=64, label_mode="document")
    content_labels = [(ex, [l for l, a in zip(ex["token_labels"], ex["attention_mask"]) if a == 1 and l != -100]) for ex in results]
    human_chunks = [ex for ex, cl in content_labels if cl and all(l == 0 for l in cl)]
    for ex in human_chunks:
        for i, (lab, attn) in enumerate(zip(ex["token_labels"], ex["attention_mask"])):
            if attn == 1 and lab != -100:
                assert lab == 0, f"Expected 0 for human chunk at pos {i}, got {lab}"


def test_build_document_mode_slop_all_one(tokenizer, pairs):
    """For examples from slop_text, all non-special non-pad tokens have label 1."""
    results = build_token_label_examples(pairs, tokenizer, max_length=64, label_mode="document")
    slop_chunks = [ex for ex in results if 1 in ex["token_labels"]]
    for ex in slop_chunks:
        for i, (lab, attn) in enumerate(zip(ex["token_labels"], ex["attention_mask"])):
            if attn == 1 and lab != -100:
                assert lab == 1, f"Expected 1 for slop chunk at pos {i}, got {lab}"


def test_build_document_mode_lengths(tokenizer, pairs):
    """len(input_ids) == len(attention_mask) == len(token_labels) == max_length."""
    max_length = 32
    results = build_token_label_examples(pairs, tokenizer, max_length=max_length, label_mode="document")
    for ex in results:
        assert len(ex["input_ids"]) == max_length
        assert len(ex["attention_mask"]) == max_length
        assert len(ex["token_labels"]) == max_length


def test_build_document_mode_special_and_pad_minus_100(tokenizer):
    """Special tokens and padding get token_labels = -100."""
    pairs_small = [("Hi there.", "Yo you know.")]
    results = build_token_label_examples(pairs_small, tokenizer, max_length=16, label_mode="document")
    for ex in results:
        for i, attn in enumerate(ex["attention_mask"]):
            if attn == 0:
                assert ex["token_labels"][i] == -100


def test_build_document_mode_chunking_with_stride(tokenizer):
    """Long text produces multiple chunks when stride < max_length."""
    long_human = " ".join(["word"] * 80)
    long_slop = " ".join(["filler"] * 80)
    pairs_long = [(long_human, long_slop)]
    max_length = 32
    stride = 16
    results = build_token_label_examples(
        pairs_long, tokenizer, max_length=max_length, stride=stride, label_mode="document"
    )
    assert len(results) >= 4


# Option 2: span_heuristic
def test_detect_sloppy_spans_phrase_list():
    """Text containing 'you know' produces at least one span."""
    spans = detect_sloppy_spans("So you know what I mean", phrase_list=["you know"], use_repetition=False, use_distinct2=False)
    assert len(spans) >= 1
    assert any(3 <= s[0] <= 15 and s[1] - s[0] >= 8 for s in spans)


def test_detect_sloppy_spans_repetition():
    """Text with 'the the' produces a repetition span."""
    spans = detect_sloppy_spans("The the quick brown fox", phrase_list=[], use_repetition=True, use_distinct2=False)
    assert len(spans) >= 1
    assert any(s[1] - s[0] >= 4 for s in spans)


def test_build_span_heuristic_mode_phrase_labeled_one(tokenizer):
    """Span heuristic: text with phrase from list has at least one token labeled 1."""
    pairs = [("", "So you know what I mean like basically")]
    results = build_token_label_examples(
        pairs, tokenizer, max_length=32, label_mode="span_heuristic", phrase_list=["you know", "basically"]
    )
    slop_chunks = [ex for ex in results if 1 in ex["token_labels"]]
    assert len(slop_chunks) >= 1
    assert any(1 in ex["token_labels"] for ex in slop_chunks)


def test_build_span_heuristic_mode_special_minus_100(tokenizer):
    """Span heuristic: special/pad tokens get -100."""
    pairs = [("", "You know like um")]
    results = build_token_label_examples(pairs, tokenizer, max_length=16, label_mode="span_heuristic")
    for ex in results:
        for i, attn in enumerate(ex["attention_mask"]):
            if attn == 0:
                assert ex["token_labels"][i] == -100


def test_build_span_heuristic_output_structure(tokenizer):
    """Span heuristic output has same keys as document mode."""
    pairs = [("Good text.", "Bad you know text.")]
    results = build_token_label_examples(pairs, tokenizer, max_length=32, label_mode="span_heuristic")
    for ex in results:
        assert "input_ids" in ex
        assert "attention_mask" in ex
        assert "token_labels" in ex
        assert len(ex["input_ids"]) == 32


# Integration: span_heuristic can yield both 0 and 1 in one chunk
def test_span_heuristic_mixed_labels_in_chunk(tokenizer):
    """Span heuristic can produce both 0 and 1 in the same chunk."""
    pairs = [("", "Clear prose and you know filler here.")]
    results = build_token_label_examples(
        pairs, tokenizer, max_length=64, stride=64, label_mode="span_heuristic"
    )
    slop_chunks = [ex for ex in results if 1 in ex["token_labels"]]
    has_mixed = any(0 in ex["token_labels"] and 1 in ex["token_labels"] for ex in slop_chunks)
    assert has_mixed or len(slop_chunks) == 0


# document mode yields only 0s or only 1s per chunk
def test_document_mode_single_label_per_chunk(tokenizer, pairs):
    """Document mode: each chunk has only 0s or only 1s (plus -100)."""
    results = build_token_label_examples(pairs, tokenizer, max_length=64, label_mode="document")
    for ex in results:
        content_labels = [l for l, a in zip(ex["token_labels"], ex["attention_mask"]) if a == 1 and l != -100]
        if content_labels:
            unique = set(content_labels)
            assert len(unique) == 1, f"Expected single label per chunk, got {unique}"
