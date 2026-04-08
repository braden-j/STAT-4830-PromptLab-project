"""Reward model: wrap trained token-level slop classifier for downstream optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import yaml

from .aggregation import aggregate_token_scores
from .diagnostics import compute_diagnostics


@dataclass
class RewardConfig:
    """Configuration for SlopRewardModel."""

    checkpoint_path: str = "outputs/classifier"
    config_path: str | None = None  # YAML with model section; if None, use checkpoint_dir/config.yaml
    aggregation_mode: Literal["mean", "max", "topk"] = "mean"
    topk_fraction: float = 0.1
    chunk_size: int = 256
    stride: int | None = None  # default chunk_size // 2
    # Penalty weights (added to doc_slop_score before negating for reward)
    lambda_rep: float = 0.0
    lambda_generic: float = 0.0
    lambda_len: float = 0.0
    min_target_length: int = 5
    max_target_length: int = 512
    generic_phrase_list: list[str] = field(default_factory=lambda: ["like", "you know", "um", "uh", "basically"])
    device: str | None = None
    batch_size: int = 16


class SlopRewardModel:
    """Reusable reward model: load classifier from checkpoint, score text with optional penalties."""

    def __init__(self, config: RewardConfig | dict | None = None, **kwargs: Any):
        if config is None:
            config = RewardConfig(**kwargs)
        elif isinstance(config, dict):
            fields = set(RewardConfig.__dataclass_fields__.keys())
            config = RewardConfig(**{k: v for k, v in config.items() if k in fields})
        else:
            config = config
        self.config = config
        self._model = None
        self._tokenizer = None
        self._device = None

    def load(self) -> None:
        """Load model and tokenizer from checkpoint. Idempotent."""
        if self._model is not None:
            return
        checkpoint_path = Path(self.config.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        config_path = self.config.config_path
        if config_path is None:
            config_path = str(checkpoint_path / "config.yaml")
        if not Path(config_path).exists():
            # Fallback: use default model config for distilbert + LoRA
            from hill_climb.config import ModelConfig
            model_config = ModelConfig(
                backbone_name="distilbert-base-uncased",
                backbone_type="encoder",
                max_length=self.config.chunk_size,
                use_lora=True,
                lora_target_modules=["q_lin", "k_lin", "v_lin"],
            )
        else:
            from hill_climb.config import Config
            cfg = Config.from_yaml(config_path)
            model_config = cfg.model
            if getattr(model_config, "max_length", None) is None:
                model_config.max_length = self.config.chunk_size

        from hill_climb.models import create_classifier_and_tokenizer
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
        self._model, _ = create_classifier_and_tokenizer(model_config)
        state_path = checkpoint_path / "pytorch_model.bin"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(state, strict=False)
        self._model.eval()
        self._tokenizer = tokenizer
        self._device = torch.device(self.config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model.to(self._device)

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

    def _tokenize(
        self,
        texts: list[str],
        max_length: int | None = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> dict[str, torch.Tensor]:
        max_len = max_length or self.config.chunk_size
        out = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length" if padding else False,
            truncation=truncation,
            max_length=max_len,
            return_attention_mask=True,
        )
        return {k: v.to(self.device) for k, v in out.items()}

    def _chunk_text(self, text: str) -> list[str]:
        """Split long text into overlapping chunks."""
        tokens = text.split()
        size = self.config.chunk_size
        stride = self.config.stride or (size // 2)
        if len(tokens) <= size:
            return [text] if text.strip() else []
        chunks = []
        for start in range(0, len(tokens), stride):
            chunk = tokens[start : start + size]
            if not chunk:
                break
            chunks.append(" ".join(chunk))
            if start + size >= len(tokens):
                break
        return chunks

    def score_batch(
        self,
        texts: list[str],
        return_token_scores: bool = False,
        return_diagnostics: bool = False,
        long_text_aggregation: str = "max",  # how to aggregate over chunks: max, mean
    ) -> dict[str, Any]:
        """Score a batch of texts. Long texts are chunked and aggregated."""
        if not texts:
            return {"doc_slop_score": [], "reward": [], "token_scores": [], "diagnostics": []}

        self.load()
        all_token_scores = []
        all_doc_scores = []
        all_diagnostics = [] if return_diagnostics else None

        for i, text in enumerate(texts):
            chunks = self._chunk_text(text)
            if not chunks:
                all_doc_scores.append(0.0)
                if return_token_scores:
                    all_token_scores.append([])
                if return_diagnostics:
                    all_diagnostics.append(compute_diagnostics(text, token_count=0))
                continue

            # Process chunks in batches of batch_size
            batch_size = self.config.batch_size
            all_chunk_scores = []
            token_scores_first_chunk = None
            for start in range(0, len(chunks), batch_size):
                batch_chunks = chunks[start : start + batch_size]
                inputs = self._tokenize(batch_chunks, max_length=self.config.chunk_size)
                with torch.no_grad():
                    slop_probs = self.model.score_tokens(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                    )
                reduction = self.config.aggregation_mode
                topk_frac = self.config.topk_fraction if reduction == "topk" else None
                chunk_scores = aggregate_token_scores(
                    slop_probs,
                    attention_mask=inputs["attention_mask"],
                    reduction=reduction,
                    topk_fraction=topk_frac,
                )
                all_chunk_scores.extend(chunk_scores.cpu().float().tolist())
                if return_token_scores and token_scores_first_chunk is None and len(batch_chunks) > 0:
                    token_scores_first_chunk = slop_probs[0].cpu().tolist()

            if long_text_aggregation == "max":
                doc_score = max(all_chunk_scores)
            else:
                doc_score = sum(all_chunk_scores) / len(all_chunk_scores)

            # Penalties (always compute for reward; optionally return full diagnostics)
            tokens_text = text.split()
            n_tok = len(tokens_text)
            diag = compute_diagnostics(text, token_count=n_tok) if return_diagnostics else None
            if return_diagnostics and all_diagnostics is not None:
                all_diagnostics.append(diag)

            rep_pen = diag["repetition_ratio"] if diag else repetition_ratio_single(text)
            gen_pen = generic_phrase_ratio_single(text, self.config.generic_phrase_list)
            len_pen = length_penalty_single(n_tok, self.config.min_target_length, self.config.max_target_length)
            doc_score += (
                self.config.lambda_rep * rep_pen
                + self.config.lambda_generic * gen_pen
                + self.config.lambda_len * len_pen
            )
            all_doc_scores.append(float(doc_score))

            if return_token_scores:
                if token_scores_first_chunk is not None:
                    all_token_scores.append(token_scores_first_chunk)
                else:
                    all_token_scores.append([])

        rewards = [-s for s in all_doc_scores]
        result = {"doc_slop_score": all_doc_scores, "reward": rewards}
        if return_token_scores:
            result["token_scores"] = all_token_scores
        if return_diagnostics and all_diagnostics is not None:
            result["diagnostics"] = all_diagnostics
        return result

    def score(
        self,
        text: str,
        return_token_scores: bool = False,
        return_diagnostics: bool = False,
    ) -> dict[str, Any]:
        """Score a single text. Returns dict with doc_slop_score, reward, and optionally token_scores, diagnostics."""
        batch = self.score_batch(
            [text],
            return_token_scores=return_token_scores,
            return_diagnostics=return_diagnostics,
        )
        out = {
            "doc_slop_score": batch["doc_slop_score"][0],
            "reward": batch["reward"][0],
        }
        if return_token_scores and batch.get("token_scores"):
            out["token_scores"] = batch["token_scores"][0]
        if return_diagnostics and batch.get("diagnostics"):
            out["diagnostics"] = batch["diagnostics"][0]
        return out


def repetition_ratio_single(text: str, n: int = 2) -> float:
    from .diagnostics import repetition_ratio
    return repetition_ratio(text, n=n)


def generic_phrase_ratio_single(text: str, phrase_list: list[str]) -> float:
    """Fraction of tokens that are part of a generic phrase (phrase counted once per occurrence)."""
    text_lower = text.lower()
    count = 0
    for phrase in phrase_list:
        count += text_lower.count(phrase.lower())
    tokens = text.split()
    return min(1.0, count / max(len(tokens), 1))


def length_penalty_single(num_tokens: int, min_len: int, max_len: int) -> float:
    """Penalty when outside [min_len, max_len]. Linear ramp outside range."""
    if num_tokens < min_len:
        return (min_len - num_tokens) / max(min_len, 1)
    if num_tokens > max_len:
        return (num_tokens - max_len) / max(max_len, 1)
    return 0.0
