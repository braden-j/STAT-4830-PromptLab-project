"""T5-based seq2seq rewriter: train on (human -> slop) pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Trainer
from datasets import Dataset


def load_slop_pairs(path: str | Path) -> list[dict[str, str]]:
    """Load {human, slop} pairs from JSONL."""
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


def train_rewriter(
    train_path: str | Path,
    val_path: str | Path | None = None,
    output_dir: str | Path = "outputs/slop_rewriter",
    model_name: str = "t5-small",
    max_source_length: int = 256,
    max_target_length: int = 256,
    batch_size: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.1,
    fp16: bool = True,
    use_wandb: bool = False,
    seed: int = 42,
) -> None:
    """Train T5 to rewrite human text -> slop text."""
    import torch
    from transformers import set_seed
    set_seed(seed)

    train_pairs = load_slop_pairs(train_path)
    if not train_pairs:
        raise FileNotFoundError(f"No training pairs at {train_path}. Generate data/slop_pairs.jsonl first.")
    val_pairs = load_slop_pairs(val_path) if val_path else None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def _tokenize(examples: dict[str, list]) -> dict[str, Any]:
        inputs = tokenizer(
            examples["human"],
            max_length=max_source_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        labels = tokenizer(
            examples["slop"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        label_ids = [[x if x != tokenizer.pad_token_id else -100 for x in seq] for seq in labels["input_ids"]]
        inputs["labels"] = label_ids
        return inputs

    train_ds = Dataset.from_list(train_pairs)
    train_ds = train_ds.map(
        _tokenize,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    eval_ds = None
    if val_pairs:
        eval_ds = Dataset.from_list(val_pairs)
        eval_ds = eval_ds.map(
            _tokenize,
            batched=True,
            remove_columns=eval_ds.column_names,
        )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=fp16 and torch.cuda.is_available(),
        logging_steps=20,
        save_steps=500,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=200,
        report_to="wandb" if use_wandb else "none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
