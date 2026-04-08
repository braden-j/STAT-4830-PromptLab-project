"""Data loading and tokenization for slop minimization.

Lives in src/hill_climb/data/ so Colab can use it after clone (no %%writefile needed).
"""

from .dataset import load_jsonl, SlopDataset
from .tokenizer import tokenize_and_align_labels, SlopTokenizer
from .token_labels import (
    build_token_label_examples,
    detect_sloppy_spans,
    spans_to_token_labels,
    DEFAULT_SLOP_PHRASES,
)

__all__ = [
    "load_jsonl",
    "SlopDataset",
    "tokenize_and_align_labels",
    "SlopTokenizer",
    "build_token_label_examples",
    "detect_sloppy_spans",
    "spans_to_token_labels",
    "DEFAULT_SLOP_PHRASES",
]
