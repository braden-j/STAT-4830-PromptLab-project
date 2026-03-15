"""Frozen causal LM wrapper for black-box text generation from prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class GeneratorConfig:
    """Configuration for FrozenGenerator."""

    model_name: str = "gpt2"
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 256
    do_sample: bool = True
    device: str | None = None
    batch_size: int = 4  # for generate_batch
    pad_token_id: int | None = None  # set from tokenizer if None


class FrozenGenerator:
    """Black-box text generator: prompt -> text. No gradients."""

    def __init__(self, config: GeneratorConfig | dict | None = None, **kwargs: Any):
        if config is None:
            config = GeneratorConfig(**kwargs)
        elif isinstance(config, dict):
            config = GeneratorConfig(**{k: v for k, v in config.items() if k in GeneratorConfig.__dataclass_fields__})
        self.config = config
        self._model = None
        self._tokenizer = None
        self._device = None

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = self.config.model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        if self.config.pad_token_id is None and self._tokenizer.pad_token_id is not None:
            self.config.pad_token_id = self._tokenizer.pad_token_id
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._device = torch.device(self.config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model.to(self._device)
        self._model.eval()

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    @property
    def device(self):
        if self._device is None:
            self.load()
        return self._device

    def generate_one(
        self,
        prompt: str,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        do_sample: bool | None = None,
    ) -> str:
        """Generate one completion from the prompt. Returns only the new text (no prompt)."""
        self.load()
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        sample = do_sample if do_sample is not None else self.config.do_sample
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tok,
                temperature=temp if sample else 1.0,
                top_p=self.config.top_p if sample else 1.0,
                do_sample=sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        # Decode only the new tokens
        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if full.startswith(prompt.strip()):
            return full[len(prompt.strip()):].strip()
        return full.strip()

    def generate_batch(
        self,
        prompts: list[str],
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        do_sample: bool | None = None,
    ) -> list[str]:
        """Generate one completion per prompt. Returns list of new text only."""
        if not prompts:
            return []
        results = []
        batch_size = self.config.batch_size
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            for p in batch:
                results.append(self.generate_one(p, temperature=temperature, max_new_tokens=max_new_tokens, do_sample=do_sample))
        return results
