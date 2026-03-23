"""Tokenization and label alignment for per-token slop classification."""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


def tokenize_and_align_labels(
    examples: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizerBase,
    text_column: str = "text",
    label_column: str = "labels",
    max_length: int = 512,
    label_pad_token_id: int = -100,
    truncation: bool = True,
) -> dict[str, Any]:
    """Tokenize text and align per-token labels with subword tokenization.

    Expects `labels` as list of int per example (0=not slop, 1=slop),
    aligned to character/word positions. We use word-level alignment:
    duplicate label for each subword within a word.

    For simple char-level: each char gets a label; we map to tokens.
    Assumes labels are per-token already if len(labels) matches token count.
    """
    tokenized = tokenizer(
        examples[text_column],
        truncation=truncation,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
        return_attention_mask=True,
        return_special_tokens_mask=True,
    )

    aligned_labels = []
    for i, labels in enumerate(examples[label_column]):
        input_ids = tokenized["input_ids"][i]
        special_tokens_mask = tokenized.get("special_tokens_mask", [0] * len(input_ids))
        if isinstance(special_tokens_mask[0], list):
            special_tokens_mask = special_tokens_mask[i]

        # labels can be: (a) list of int same len as input_ids, or
        # (b) list of int for words; we replicate per subword
        if len(labels) == len(input_ids):
            label_ids = list(labels)
        else:
            # Word-level: assume labels per word; replicate per token
            # Simplified: pad labels to seq length, -100 for special
            label_ids = [label_pad_token_id] * len(input_ids)
            num_labels = min(len(labels), len(input_ids))
            for j in range(num_labels):
                if not special_tokens_mask[j]:
                    label_ids[j] = labels[j] if j < len(labels) else 0

        for j, mask in enumerate(special_tokens_mask):
            if mask:
                label_ids[j] = label_pad_token_id

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


class SlopTokenizer:
    """Wrapper around HF tokenizer for slop tokenization + scoring interface."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        text_column: str = "text",
        label_column: str = "labels",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column

    def encode(
        self,
        text: str,
        truncation: bool = True,
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        """Encode text for classification."""
        out = self.tokenizer(
            text,
            truncation=truncation,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=return_tensors or "pt",
            return_attention_mask=True,
        )
        return out

    def decode(self, ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token ids to string."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
