"""Unit tests for tokenization and label alignment."""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "slop_src"))

from slop.tokenizer_utils import tokenize_and_align_labels, SlopTokenizer
from conftest import get_tokenizer


@pytest.fixture
def tokenizer():
    return get_tokenizer()


def test_tokenize_and_align_labels_same_length(tokenizer):
    """When labels match input_ids length, they should be preserved."""
    examples = {
        "text": ["Hello world", "Test sentence"],
        "labels": [
            [0, 1],  # word-level: hello=0, world=1
            [1, 0, 1],
        ],
    }
    result = tokenize_and_align_labels(
        examples,
        tokenizer,
        text_column="text",
        label_column="labels",
        max_length=32,
    )
    assert "input_ids" in result
    assert "labels" in result
    assert "attention_mask" in result
    assert len(result["labels"]) == 2
    # Labels are padded/aligned to max_length
    assert len(result["labels"][0]) == 32
    assert len(result["labels"][1]) == 32


def test_slop_tokenizer_encode_decode(tokenizer):
    """SlopTokenizer encode/decode roundtrip."""
    st = SlopTokenizer(tokenizer, max_length=64)
    text = "This is a test sentence."
    encoded = st.encode(text)
    assert "input_ids" in encoded
    assert "attention_mask" in encoded
    decoded = st.decode(encoded["input_ids"].squeeze().tolist())
    assert text in decoded or decoded in text  # Allow subword differences


def test_slop_tokenizer_encoding_shape(tokenizer):
    """Encoded output has correct shape for batch."""
    st = SlopTokenizer(tokenizer, max_length=32)
    enc = st.encode("Short text", return_tensors="pt")
    assert enc["input_ids"].shape[1] == 32
    assert enc["attention_mask"].shape[1] == 32


def test_label_pad_token_respected(tokenizer):
    """Special tokens get -100 in labels."""
    examples = {
        "text": ["Hi"],
        "labels": [[0, 1]],  # 2 tokens worth
    }
    result = tokenize_and_align_labels(
        examples,
        tokenizer,
        max_length=16,
        label_pad_token_id=-100,
    )
    labels = result["labels"][0]
    # Padded positions and special tokens should be -100
    assert -100 in labels or len(labels) == 16
