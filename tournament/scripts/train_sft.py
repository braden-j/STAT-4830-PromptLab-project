#!/usr/bin/env python3
"""Train a decoder-only rewrite model for SFT tournament entrants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hill_climb.tournament.config import TrainingConfig, load_yaml_config
from hill_climb.tournament.io import file_sha256, read_jsonl, write_json
from hill_climb.tournament.training import build_sft_record, default_lora_targets, trainable_param_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an SFT tournament entrant")
    p.add_argument("--config", required=True)
    return p.parse_args()


def _mask_prompt_labels(tokenizer, prompt: str, completion: str, max_length: int) -> dict[str, list[int]]:
    full_text = prompt + completion
    tokenized = tokenizer(full_text, truncation=True, max_length=max_length)
    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
    labels = list(tokenized["input_ids"])
    mask_len = min(len(prompt_ids), len(labels))
    labels[:mask_len] = [-100] * mask_len
    tokenized["labels"] = labels
    return tokenized


def _apply_training_mode(model, cfg: TrainingConfig):
    from peft import LoraConfig, TaskType, get_peft_model

    mode = cfg.training_mode
    if mode in {"m1", "m2", "m5", "m6", "sft_lora_attn", "sft_lora_full"}:
        targets = cfg.lora_target_modules or default_lora_targets(mode)
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=targets,
            bias="none",
        )
        return get_peft_model(model, peft_cfg)

    if mode == "m3":
        for param in model.parameters():
            param.requires_grad_(False)
        layers = list(model.model.layers)
        keep_from = max(0, int(len(layers) * 0.75))
        for layer in layers[keep_from:]:
            for param in layer.parameters():
                param.requires_grad_(True)
        for param in model.model.norm.parameters():
            param.requires_grad_(True)
        for param in model.lm_head.parameters():
            param.requires_grad_(True)
        return model

    if mode == "m4":
        for param in model.parameters():
            param.requires_grad_(True)
        return model

    raise ValueError(f"Unsupported training mode: {mode}")


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config, "train")
    assert isinstance(cfg, TrainingConfig)

    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Missing tournament dependencies. Install with `pip install .[tournament]`.") from exc

    raw_rows = [build_sft_record(row) for row in read_jsonl(cfg.data_pack_path)]
    if not raw_rows:
        raise FileNotFoundError(f"No SFT data found at {cfg.data_pack_path}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model = _apply_training_mode(model, cfg)

    train_rows = [row for row in raw_rows if row["split"] == "train"]
    eval_rows = [row for row in raw_rows if row["split"] == "val"]

    def tokenize_batch(batch):
        encoded = [
            _mask_prompt_labels(tokenizer, prompt, completion, cfg.max_seq_len)
            for prompt, completion in zip(batch["prompt"], batch["completion"])
        ]
        keys = encoded[0].keys()
        return {key: [row[key] for row in encoded] for key in keys}

    train_ds = Dataset.from_list(train_rows).map(tokenize_batch, batched=True, remove_columns=Dataset.from_list(train_rows).column_names)
    eval_ds = Dataset.from_list(eval_rows).map(tokenize_batch, batched=True, remove_columns=Dataset.from_list(eval_rows).column_names) if eval_rows else None

    report_to = ["wandb"] if cfg.use_wandb else []
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs_max,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=10,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        report_to=report_to,
        fp16=torch.cuda.is_available(),
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    metadata = {
        "config_path": args.config,
        "dataset_hash": file_sha256(cfg.data_pack_path),
        "trainable_summary": trainable_param_summary(model),
    }
    write_json(Path(cfg.output_dir) / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
