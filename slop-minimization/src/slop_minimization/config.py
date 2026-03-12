"""Configuration handling for slop-minimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class ModelConfig:
    """Model backbone and head configuration."""

    backbone_name: str = "distilbert-base-uncased"
    backbone_type: str = "encoder"  # "encoder" | "causal"
    max_length: int = 512
    num_labels: int = 2  # slop vs not-slop
    dropout: float = 0.1
    use_unsloth: bool = False
    use_lora: bool = True
    freeze_base: bool = True  # freeze all base weights except LoRA + classifier head
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    fp16: bool = True
    use_wandb: bool = False
    run_name: str | None = None
    output_dir: str = "outputs"
    seed: int = 42
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10


@dataclass
class DataConfig:
    """Data loading and preprocessing."""

    train_path: str = "data/train.jsonl"
    val_path: str = "data/val.jsonl"
    test_path: str = "data/test.jsonl"
    text_column: str = "text"
    label_column: str = "labels"  # per-token labels
    max_samples: int | None = None


@dataclass
class PromptSearchConfig:
    """Black-box prompt optimization config."""

    num_iterations: int = 100
    population_size: int = 20
    mutation_rate: float = 0.1
    top_k: int = 5
    prompt_length: int = 32
    seed_prompts: list[str] = field(default_factory=list)


@dataclass
class Config:
    """Master configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    prompt_search: PromptSearchConfig = field(default_factory=PromptSearchConfig)
    stage: Literal["classifier", "generator", "prompt_search", "eval"] = "classifier"

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load config from YAML file."""
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        """Build Config from nested dict."""
        return cls(
            model=ModelConfig(**d.get("model", {})),
            training=TrainingConfig(**d.get("training", {})),
            data=DataConfig(**d.get("data", {})),
            prompt_search=PromptSearchConfig(**d.get("prompt_search", {})),
            stage=d.get("stage", "classifier"),
        )
