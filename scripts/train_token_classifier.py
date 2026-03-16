#!/usr/bin/env python3
"""Train token-level slop classifier (encoder backbone + linear head) from YAML config."""

from __future__ import annotations

import argparse
import hashlib
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets import Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from slop_minimization.config import Config
from slop_minimization.data.dataset import load_jsonl
from slop_minimization.data.tokenizer import tokenize_and_align_labels
from slop_minimization.models import create_classifier_and_tokenizer
from slop_minimization.metrics import (
    token_level_f1,
    token_level_auroc,
    doc_level_auroc,
    doc_labels_from_token_labels,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train token-level slop classifier from YAML config")
    p.add_argument("--config", type=str, default="configs/classifier_encoder.yaml")
    p.add_argument("--output-dir", type=str, default=None, help="Override config output_dir")
    p.add_argument("--use-wandb", action="store_true", help="Override to enable wandb")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def first_batch_difficulty_counts(
    train_difficulties: list[str],
    weights: list[float],
    batch_size: int,
    seed_offset: int = 0,
) -> dict[str, int]:
    """Sample first batch indices using same distribution as WeightedRandomSampler; return difficulty counts."""
    probs = torch.tensor(weights, dtype=torch.float64)
    probs = probs / probs.sum()
    gen = torch.Generator()
    gen.manual_seed(42 + seed_offset)
    n_sample = min(batch_size, len(weights))
    idx = torch.multinomial(probs, num_samples=n_sample, replacement=True, generator=gen)
    first_indices = idx.tolist()
    diff_collection = [train_difficulties[i] for i in first_indices]
    return dict(Counter(diff_collection))


def checksum_state_dict(state_dict: dict) -> str:
    """Return a short SHA-256 checksum of the saved state (param names + shapes + first bytes of data)."""
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        t = state_dict[k]
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        # Include a small amount of data so different weights produce different hashes
        if t.numel() > 0:
            flat = t.flatten()
            h.update(flat[: min(32, flat.numel())].cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def checksum_saved_file(path: Path) -> str:
    """Return SHA-256 hex digest of file (first 16 chars)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            blk = f.read(65536)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()[:16]


def collate_fn(batch):
    """Stack batch and convert to tensors."""
    input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    attention_mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    max_grad_norm,
    accumulation_steps,
):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            out = model(**batch)
            loss = out.loss / accumulation_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        total_loss += loss.item() * accumulation_steps
        if (step + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, compute_token_auroc=True):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_doc_scores = []
    all_doc_labels = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        probs = torch.softmax(logits, dim=-1)
        slop_probs = probs[..., 1]
        preds = logits.argmax(dim=-1)
        all_preds.append(preds)
        all_labels.append(labels)
        all_probs.append(slop_probs)
        doc_scores = model.doc_slop_score(input_ids, attention_mask)
        all_doc_scores.append(doc_scores)
        doc_lab = doc_labels_from_token_labels(labels, attention_mask, strategy="any")
        all_doc_labels.append(doc_lab)
    if not all_preds:
        return {"token_f1": 0.0, "token_auroc": 0.0, "doc_auroc": 0.0}
    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    probs = torch.cat(all_probs, dim=0)
    doc_scores = torch.cat(all_doc_scores, dim=0)
    doc_labels = torch.cat(all_doc_labels, dim=0)
    token_f1 = token_level_f1(preds, labels)
    token_auroc = token_level_auroc(probs, labels) if compute_token_auroc else 0.0
    doc_auroc = doc_level_auroc(doc_scores, doc_labels)
    return {"token_f1": token_f1, "token_auroc": token_auroc, "doc_auroc": doc_auroc}


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    cfg_model = config.model
    cfg_train = config.training
    cfg_data = config.data

    if args.output_dir:
        cfg_train.output_dir = args.output_dir
    if args.use_wandb:
        cfg_train.use_wandb = True

    set_seed(int(cfg_train.seed))
    out_dir = Path(cfg_train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # [DEBUG] output_dir and overwrite risk
    print(f"[DEBUG] output_dir = {out_dir.resolve()}")
    existing_ckpt = out_dir / "pytorch_model.bin"
    if existing_ckpt.exists():
        print(f"[DEBUG] WARNING: {existing_ckpt} already exists; training will overwrite it. Use a different --output-dir for curriculum vs baseline.")

    model, tokenizer = create_classifier_and_tokenizer(cfg_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # [DEBUG] trainable parameter count
    n_trainable = count_trainable_parameters(model)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[DEBUG] trainable_parameters = {n_trainable} / {n_total} (total)")

    # [DEBUG] initial weights checksum (should differ from checksum after training if params update)
    initial_checksum = checksum_state_dict(model.state_dict())
    print(f"[DEBUG] initial state_dict_checksum = {initial_checksum}")

    train_data = load_jsonl(cfg_data.train_path)
    val_data = load_jsonl(cfg_data.val_path)
    if not train_data:
        raise FileNotFoundError(f"No training data at {cfg_data.train_path}. Run build_data.py first.")

    if cfg_data.max_samples:
        train_data = train_data[: cfg_data.max_samples]
        if val_data:
            val_data = val_data[: max(1, cfg_data.max_samples // 10)]

    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data) if val_data else None

    def tokenize_fn(examples):
        return tokenize_and_align_labels(
            examples,
            tokenizer,
            text_column=cfg_data.text_column,
            label_column=cfg_data.label_column,
            max_length=int(cfg_model.max_length),
        )

    tokenized_train = train_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    tokenized_val = val_ds.map(tokenize_fn, batched=True, remove_columns=val_ds.column_names) if val_ds else None
    if tokenized_val is not None and len(tokenized_val) == 0:
        tokenized_val = None

    train_loader = DataLoader(
        tokenized_train,
        batch_size=int(cfg_train.batch_size),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    if getattr(cfg_data, "curriculum_enabled", False) and getattr(cfg_data, "difficulty_column", ""):
        diff_col = getattr(cfg_data, "difficulty_column", "difficulty")
        train_difficulties = [ex.get(diff_col, "easy") for ex in train_data]
    else:
        train_difficulties = None
    val_loader = DataLoader(
        tokenized_val,
        batch_size=int(cfg_train.batch_size),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    ) if tokenized_val else None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_train.learning_rate),
        weight_decay=float(cfg_train.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler() if cfg_train.fp16 and device.type == "cuda" else None
    accumulation_steps = int(cfg_train.gradient_accumulation_steps)
    patience = int(cfg_train.early_stopping_patience)
    best_metric = -1.0
    best_epoch = -1
    patience_counter = 0

    n_epochs = int(cfg_train.num_epochs)
    early_ratio = getattr(cfg_data, "curriculum_early_epoch_ratio", 0.5)
    n_early = max(1, int(n_epochs * early_ratio)) if train_difficulties else 0

    for epoch in range(n_epochs):
        epoch_loader = train_loader
        curriculum_sampler_enabled = False
        if train_difficulties is not None:
            curriculum_sampler_enabled = True
            if epoch < n_early:
                w_e, w_m, w_h = getattr(cfg_data, "curriculum_early_easy", 0.8), getattr(cfg_data, "curriculum_early_medium", 0.2), getattr(cfg_data, "curriculum_early_hard", 0.0)
            else:
                w_e, w_m, w_h = getattr(cfg_data, "curriculum_late_easy", 0.2), getattr(cfg_data, "curriculum_late_medium", 0.4), getattr(cfg_data, "curriculum_late_hard", 0.4)
            wmap = {"easy": w_e, "medium": w_m, "hard": w_h}
            weights = [wmap.get(d, 1.0) for d in train_difficulties]
            epoch_loader = DataLoader(tokenized_train, batch_size=int(cfg_train.batch_size), shuffle=False, sampler=WeightedRandomSampler(weights, num_samples=len(weights)), collate_fn=collate_fn, num_workers=0)

        # [DEBUG] curriculum sampler enabled this epoch
        print(f"[DEBUG] epoch {epoch + 1}/{n_epochs} curriculum_sampler_enabled = {curriculum_sampler_enabled}")

        # [DEBUG] first-batch difficulty distribution for epoch 1 and a later epoch
        if train_difficulties is not None and (epoch == 0 or epoch == n_early):
            first_batch_diff = first_batch_difficulty_counts(train_difficulties, weights, int(cfg_train.batch_size), seed_offset=epoch * 999)
            print(f"[DEBUG] epoch {epoch + 1} first_batch_difficulty_distribution = {first_batch_diff}")

        train_loss = train_one_epoch(
            model,
            epoch_loader,
            optimizer,
            scaler,
            device,
            float(cfg_train.max_grad_norm),
            accumulation_steps,
        )
        log_msg = f"Epoch {epoch + 1} train_loss={train_loss:.4f}"
        if val_loader:
            metrics = evaluate(model, val_loader, device, compute_token_auroc=True)
            log_msg += f" val_token_f1={metrics['token_f1']:.4f} val_doc_auroc={metrics['doc_auroc']:.4f}"
            if metrics["token_auroc"]:
                log_msg += f" val_token_auroc={metrics['token_auroc']:.4f}"
            print(log_msg)
            if cfg_train.use_wandb:
                try:
                    import wandb
                    wandb.log({"train_loss": train_loss, "epoch": epoch + 1, **{f"val_{k}": v for k, v in metrics.items()}})
                except Exception:
                    pass
            metric_for_best = metrics["doc_auroc"]
            if metric_for_best > best_metric:
                best_metric = metric_for_best
                best_epoch = epoch + 1
                patience_counter = 0
                save_path = out_dir / "pytorch_model.bin"
                torch.save(model.state_dict(), save_path)
                tokenizer.save_pretrained(out_dir)
                ckpt_checksum = checksum_saved_file(save_path)
                state_checksum = checksum_state_dict(model.state_dict())
                print(f"[DEBUG] saved checkpoint: file_checksum = {ckpt_checksum} state_dict_checksum = {state_checksum}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1} (best epoch {best_epoch}, doc_auroc={best_metric:.4f})")
                    break
        else:
            print(log_msg)
            save_path = out_dir / "pytorch_model.bin"
            torch.save(model.state_dict(), save_path)
            tokenizer.save_pretrained(out_dir)
            ckpt_checksum = checksum_saved_file(save_path)
            state_checksum = checksum_state_dict(model.state_dict())
            print(f"[DEBUG] saved checkpoint (no val): file_checksum = {ckpt_checksum} state_dict_checksum = {state_checksum}")

    if val_loader and patience_counter < patience:
        print(f"Best epoch {best_epoch} doc_auroc={best_metric:.4f}")
    print(f"Model and tokenizer saved to {out_dir}")
    final_ckpt = out_dir / "pytorch_model.bin"
    if final_ckpt.exists():
        print(f"[DEBUG] final saved checkpoint file_checksum = {checksum_saved_file(final_ckpt)}")
    if train_difficulties is not None:
        print("[DEBUG] Curriculum was enabled. To compare baseline vs curriculum, run baseline with --output-dir outputs/classifier and curriculum with --output-dir outputs/classifier_curriculum (different dirs).")


if __name__ == "__main__":
    main()
