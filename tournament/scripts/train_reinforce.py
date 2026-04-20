#!/usr/bin/env python3
"""Train a rewrite model with a lightweight REINFORCE loop using Pangram as reward."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hill_climb.tournament.config import TrainingConfig, load_yaml_config
from hill_climb.tournament.io import file_sha256, read_jsonl, write_json
from hill_climb.tournament.prompts import format_rewrite_prompt
from hill_climb.tournament.scoring import length_ratio, load_canonical_pangram_module
from hill_climb.tournament.training import default_lora_targets, trainable_param_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a tournament entrant with REINFORCE")
    p.add_argument("--config", required=True)
    return p.parse_args()


def _apply_lora(model, cfg: TrainingConfig):
    from peft import LoraConfig, TaskType, get_peft_model

    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules or default_lora_targets("m7"),
        bias="none",
    )
    return get_peft_model(model, peft_cfg)


def _build_labels(full_ids, prompt_len: int):
    import torch

    labels = full_ids.clone()
    labels[:, :prompt_len] = -100
    return labels


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config, "train")
    assert isinstance(cfg, TrainingConfig)

    try:
        import torch
        from sentence_transformers import SentenceTransformer, util
        from torch.optim import AdamW
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Missing tournament dependencies. Install with `pip install .[tournament]`.") from exc

    rows = read_jsonl(cfg.data_pack_path)
    train_rows = [row for row in rows if row.get("split") == "train"]
    if not train_rows:
        raise FileNotFoundError(f"No REINFORCE train rows found at {cfg.data_pack_path}")

    if cfg.use_wandb:
        os.environ.setdefault("WANDB_ENTITY", cfg.wandb_entity)
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
        import wandb

        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=os.environ.get("WANDB_NAME", cfg.experiment_id),
            group=os.environ.get("WANDB_RUN_GROUP", "tournament_reinforce"),
            config={
                "config_path": args.config,
                "experiment_id": cfg.experiment_id,
                "learning_rate": cfg.learning_rate,
                "reinforce_steps": cfg.reinforce_steps,
                "reinforce_batch_size": cfg.reinforce_batch_size,
            },
        )
    else:
        wandb = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model = _apply_lora(model, cfg).to(device)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_epsilon,
    )

    pangram = load_canonical_pangram_module()
    scorer_tok, scorer_mdl = pangram.load_editlens(device)
    sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    reward_baseline = 0.0
    os.makedirs(cfg.output_dir, exist_ok=True)

    for step in range(1, cfg.reinforce_steps + 1):
        batch = rng.sample(train_rows, k=min(cfg.reinforce_batch_size, len(train_rows)))
        loss_terms = []
        rewards = []
        pangram_scores = []
        similarities = []
        ratios = []

        model.eval()
        for row in batch:
            prompt = format_rewrite_prompt(row["input_text"])
            prompt_inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.max_seq_len,
            ).to(device)
            with torch.no_grad():
                generated = model.generate(
                    **prompt_inputs,
                    max_new_tokens=cfg.generation.max_new_tokens,
                    do_sample=True,
                    temperature=max(cfg.generation.temperature, 0.7),
                    top_p=min(cfg.generation.top_p, 0.95),
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = prompt_inputs["input_ids"].shape[1]
            completion_ids = generated[:, prompt_len:]
            output_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True).strip()
            if not output_text:
                continue

            pangram_score = pangram.editlens_score(scorer_tok, scorer_mdl, output_text, device)
            ref_emb = sim_model.encode(row["target_text"], convert_to_tensor=True)
            out_emb = sim_model.encode(output_text, convert_to_tensor=True)
            similarity = float(util.cos_sim(ref_emb, out_emb).item())
            ratio = length_ratio(row["target_text"], output_text)
            reward = (
                (1.0 - pangram_score)
                + cfg.reward_similarity_weight * similarity
                - cfg.reward_length_penalty * abs(1.0 - ratio)
            )
            centered_reward = reward - reward_baseline

            model.train()
            full_ids = generated.to(device)
            attention_mask = torch.ones_like(full_ids, device=device)
            labels = _build_labels(full_ids, prompt_len)
            outputs = model(input_ids=full_ids, attention_mask=attention_mask, labels=labels)
            n_valid = max(int((labels != -100).sum().item()), 1)
            seq_logprob = -outputs.loss * n_valid
            loss_terms.append(-seq_logprob * centered_reward)
            model.eval()

            rewards.append(reward)
            pangram_scores.append(pangram_score)
            similarities.append(similarity)
            ratios.append(ratio)

        if not loss_terms:
            continue

        reward_baseline = (
            cfg.reward_baseline_momentum * reward_baseline
            + (1.0 - cfg.reward_baseline_momentum) * (sum(rewards) / len(rewards))
        )
        loss = torch.stack(loss_terms).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        log_payload = {
            "step": step,
            "loss": float(loss.item()),
            "mean_reward": sum(rewards) / len(rewards),
            "mean_pangram_score": sum(pangram_scores) / len(pangram_scores),
            "mean_similarity": sum(similarities) / len(similarities),
            "mean_length_ratio": sum(ratios) / len(ratios),
            "reward_baseline": reward_baseline,
        }
        print(json.dumps(log_payload), flush=True)
        if wandb is not None:
            wandb.log(log_payload, step=step)

        if step % cfg.save_every_steps == 0:
            checkpoint_dir = Path(cfg.output_dir) / f"step_{step:04d}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    metadata = {
        "config_path": args.config,
        "dataset_hash": file_sha256(cfg.data_pack_path),
        "trainable_summary": trainable_param_summary(model),
        "reinforce_steps": cfg.reinforce_steps,
    }
    write_json(Path(cfg.output_dir) / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))

    if wandb is not None:
        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
