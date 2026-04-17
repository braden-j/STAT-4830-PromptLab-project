"""Training helpers for tournament SFT and DPO scripts."""

from __future__ import annotations

from typing import Any

from hill_climb.tournament.prompts import format_rewrite_prompt


def default_lora_targets(training_mode: str) -> list[str]:
    """Return the default target modules for a given training regime."""

    if training_mode in {"m1", "sft_lora_attn"}:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if training_mode in {"m2", "sft_lora_full", "m5", "m6", "m7"}:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return []


def build_sft_record(row: dict[str, Any]) -> dict[str, str]:
    """Convert a pack row into prompt/completion text."""

    source_text = row["input_text"]
    target_text = row["target_text"]
    prompt = format_rewrite_prompt(source_text)
    return {
        "prompt": prompt,
        "completion": target_text.strip(),
        "source_id": row.get("source_id", row.get("id", "")),
        "split": row.get("split", "train"),
    }


def build_dpo_record(row: dict[str, Any]) -> dict[str, str]:
    """Convert a preference row into DPO prompt/chosen/rejected format."""

    prompt = format_rewrite_prompt(row["input_text"])
    return {
        "prompt": prompt,
        "chosen": row["chosen_text"].strip(),
        "rejected": row["rejected_text"].strip(),
        "source_id": row.get("source_id", row.get("id", "")),
        "split": row.get("split", "train"),
    }


def trainable_param_summary(model) -> dict[str, int | float]:
    """Summarize trainable parameter counts for logging."""

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = (100.0 * trainable / total) if total else 0.0
    return {"trainable": trainable, "total": total, "pct": pct}
