"""Frozen causal LM wrapper for black-box text generation from prompts.

Supports GPT-2, LLaMA-style, and other HuggingFace causal LMs. Handles tokenizer
padding and EOS defaults so the same interface works across small instruction-following
and base models.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    repetition_penalty: float = 1.0  # >1 reduces repetition
    no_repeat_ngram_size: int = 0  # 0 = disabled; 2/3/4 block repeating n-grams
    eos_token_id: int | None = None  # set from tokenizer if None; stop at EOS when set
    max_prompt_length: int = 1024  # tokenizer truncation for long prompts


class FrozenGenerator:
    """Black-box text generator: prompt -> text. No gradients. Works across small causal LMs."""

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

        # Pad token: many causal LMs have no pad_token; use EOS for generation/batch padding
        if self.config.pad_token_id is not None:
            self._tokenizer.pad_token_id = self.config.pad_token_id
        elif getattr(self._tokenizer, "pad_token_id", None) is not None:
            self.config.pad_token_id = self._tokenizer.pad_token_id
        else:
            eid = self._tokenizer.eos_token_id
            if eid is not None:
                self._tokenizer.pad_token_id = eid
                self.config.pad_token_id = eid
            if getattr(self._tokenizer, "pad_token", None) is None and getattr(self._tokenizer, "eos_token", None) is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

        # EOS for stopping: normalize to single int (some tokenizers use a list)
        if self.config.eos_token_id is not None:
            pass  # use config
        else:
            eid = getattr(self._tokenizer, "eos_token_id", None)
            if eid is not None:
                self.config.eos_token_id = eid if isinstance(eid, int) else (eid[0] if eid else None)

        # Left padding for causal LM batch generation so rightmost positions are prompt + generation
        self._tokenizer.padding_side = "left"

        self._device = torch.device(
            self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
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

    def _get_pad_token_id(self) -> int:
        """Return pad_token_id for generate(); never None so all models work."""
        pid = self.tokenizer.pad_token_id
        if pid is not None:
            return pid
        eid = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eid, list):
            eid = eid[0] if eid else None
        return int(eid) if eid is not None else 0

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
        max_len = getattr(self.config, "max_prompt_length", 1024) or 1024
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        ).to(self.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tok,
            "temperature": temp if sample else 1.0,
            "top_p": self.config.top_p if sample else 1.0,
            "do_sample": sample,
            "pad_token_id": self._get_pad_token_id(),
            "repetition_penalty": self.config.repetition_penalty,
        }
        if self.config.no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = self.config.no_repeat_ngram_size
        if self.config.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = self.config.eos_token_id
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        prompt_stripped = prompt.strip()
        if full.startswith(prompt_stripped):
            return full[len(prompt_stripped) :].strip()
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
