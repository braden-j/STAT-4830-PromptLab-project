#!/usr/bin/env python3
"""Train the DPO tournament entrant from the P3 preference pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hill_climb.tournament.config import TrainingConfig, load_yaml_config
from hill_climb.tournament.io import file_sha256, read_jsonl, write_json
from hill_climb.tournament.training import build_dpo_record, default_lora_targets, trainable_param_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the DPO tournament entrant")
    p.add_argument("--config", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config, "train")
    assert isinstance(cfg, TrainingConfig)

    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError("Missing tournament dependencies. Install with `pip install .[tournament]`.") from exc

    raw_rows = [build_dpo_record(row) for row in read_jsonl(cfg.data_pack_path)]
    if not raw_rows:
        raise FileNotFoundError(f"No preference data found at {cfg.data_pack_path}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules or default_lora_targets("m7"),
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_rows = [row for row in raw_rows if row["split"] == "train"]
    eval_rows = [row for row in raw_rows if row["split"] == "val"]
    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows) if eval_rows else None

    dpo_args = DPOConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs_max,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=10,
        eval_steps=50 if eval_ds is not None else None,
        save_steps=50,
        report_to=["wandb"] if cfg.use_wandb else [],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    metadata = {
        "config_path": args.config,
        "dataset_hash": file_sha256(cfg.data_pack_path),
        "trainable_summary": trainable_param_summary(trainer.model),
    }
    write_json(Path(cfg.output_dir) / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
