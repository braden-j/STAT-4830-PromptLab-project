"""Configuration helpers for the tournament workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SourcePoolConfig:
    """Source-pool and pack-building defaults."""

    train_size: int = 3000
    val_size: int = 500
    test_size: int = 500
    pangram_share: float = 0.80
    kaggle_share: float = 0.15
    defan_share: float = 0.05
    min_words: int = 80
    max_words: int = 260
    seed: int = 42
    kaggle_path: str = "tournament/data/raw/kaggle/aeon_essays.csv"
    defan_path: str = "tournament/data/raw/defan/DefAn_public_combined.json"
    output_path: str = "tournament/data/source_pool.jsonl"


@dataclass
class GenerationConfig:
    """Prompt/decode controls shared by training and eval."""

    max_seq_len: int = 768
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False


@dataclass
class TrainingConfig:
    """Training configuration used by SFT and DPO entrypoints."""

    experiment_id: str = "m2_full_lora"
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    training_mode: str = "sft_lora_full"
    data_pack_path: str = "tournament/data/packs/p0_pairs.jsonl"
    eval_path: str = "tournament/data/source_pool.jsonl"
    output_dir: str = "tournament/outputs/runs/m2_full_lora"
    max_seq_len: int = 768
    token_budget: int = 2_000_000
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_epochs_max: int = 2
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "linear"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    seed: int = 42
    use_wandb: bool = True
    wandb_entity: str = "jgold23-university-of-pennsylvania-model-united-nations-"
    wandb_project: str = "PromptLab_STAT4830"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=list)
    reinforce_steps: int = 200
    reinforce_batch_size: int = 4
    reward_similarity_weight: float = 0.25
    reward_length_penalty: float = 0.10
    reward_baseline_momentum: float = 0.9
    save_every_steps: int = 50
    generation: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass
class EvalConfig:
    """Evaluation and leaderboard configuration."""

    run_dir: str = "tournament/outputs/runs/m2_full_lora"
    eval_path: str = "tournament/data/source_pool.jsonl"
    output_path: str = "tournament/outputs/leaderboards/m2_eval.json"
    eval_split: str = "test"
    score_model: str = "pangram/editlens_roberta-large"
    similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    mean_similarity_gate: float = 0.88
    min_length_ratio: float = 0.80
    max_length_ratio: float = 1.25
    max_invalid_rate: float = 0.02
    generation: GenerationConfig = field(default_factory=GenerationConfig)


def _coerce_nested(dc_cls: type[Any], payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    """Promote nested dicts to dataclasses when present."""

    data = dict(payload)
    if isinstance(data.get(field_name), dict):
        data[field_name] = dc_cls(**data[field_name])
    return data


def load_yaml_config(path: str | Path, kind: str) -> SourcePoolConfig | TrainingConfig | EvalConfig:
    """Load a config file into one of the tournament config dataclasses."""

    with open(path) as f:
        payload = yaml.safe_load(f) or {}

    if kind == "source":
        return SourcePoolConfig(**payload)
    if kind == "train":
        payload = _coerce_nested(GenerationConfig, payload, "generation")
        return TrainingConfig(**payload)
    if kind == "eval":
        payload = _coerce_nested(GenerationConfig, payload, "generation")
        return EvalConfig(**payload)
    raise ValueError(f"Unsupported config kind: {kind}")
