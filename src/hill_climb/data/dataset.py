"""Dataset classes for slop minimization.

Part of src/hill_climb/data/; provides load_jsonl and SlopDataset for scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset

from .tokenizer import tokenize_and_align_labels, SlopTokenizer


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL file."""
    path = Path(path)
    if not path.exists():
        return []
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


class SlopDataset(TorchDataset):
    """PyTorch Dataset for token-level slop classification."""

    def __init__(
        self,
        data: list[dict[str, Any]] | Dataset | str | Path,
        tokenizer: SlopTokenizer,
        text_column: str = "text",
        label_column: str = "labels",
        max_length: int = 512,
    ):
        if isinstance(data, (str, Path)):
            data = load_jsonl(data)
        if isinstance(data, list):
            self.data = Dataset.from_list(data)
        else:
            self.data = data

        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length

        self._tokenized: dict[str, Any] | None = None

    def _ensure_tokenized(self) -> None:
        if self._tokenized is not None:
            return
        self._tokenized = tokenize_and_align_labels(
            self.data.to_dict(),
            self.tokenizer.tokenizer,
            text_column=self.text_column,
            label_column=self.label_column,
            max_length=self.max_length,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        self._ensure_tokenized()
        return {
            "input_ids": self._tokenized["input_ids"][idx],
            "attention_mask": self._tokenized["attention_mask"][idx],
            "labels": self._tokenized["labels"][idx],
        }
