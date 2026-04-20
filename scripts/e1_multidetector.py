"""
scripts/e1_multidetector.py — Experiment E1: Multi-detector evaluation

Load step_500 LoRA checkpoint and score 20 ai_generated essays with two detectors:
  - EditLens    (pangram/editlens_roberta-large): 0=human, 1=AI
  - Perplexity  (meta-llama/Llama-3.2-3B, frozen): higher = more human-like

For each essay: generate a rewrite, then score both original and rewrite.
Print a table and save results.

Usage:
    python scripts/e1_multidetector.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from fluency_reward_validation import editlens_score, load_editlens, load_fluency_model
from c2_reinforce import build_prompt, POLICY_MODEL_ID

CHECKPOINT_PATH = os.path.join(_REPO_ROOT, "outputs", "c2_checkpoints", "step_500")
RESULTS_PATH    = os.path.join(_REPO_ROOT, "outputs", "e1_multidetector.jsonl")
SUMMARY_PATH    = os.path.join(_REPO_ROOT, "outputs", "e1_summary.txt")

N_ESSAYS       = 20
MAX_NEW_TOKENS = 512
MIN_WORDS      = 80
MAX_WORDS      = 250


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[device] CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        dev = torch.device("cpu")
        print("[device] CPU", flush=True)
    return dev


def load_policy(device: torch.device):
    print(f"[policy] Loading {POLICY_MODEL_ID} + LoRA from {CHECKPOINT_PATH} ...", flush=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(POLICY_MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        POLICY_MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base, CHECKPOINT_PATH, is_trainable=False)
    model = model.to(device).eval()
    print("[policy] Loaded.\n", flush=True)
    return tok, model


def load_essays() -> list[str]:
    print(f"[data] Loading {N_ESSAYS} ai_generated essays from test split ...", flush=True)
    ds = load_dataset("pangram/editlens_iclr", split="test", streaming=True)
    essays: list[str] = []
    for row in ds:
        if row.get("text_type") != "ai_generated":
            continue
        text = row["text"].strip()
        words = text.split()
        if len(words) < MIN_WORDS:
            continue
        if len(words) > MAX_WORDS:
            text = " ".join(words[:MAX_WORDS])
        essays.append(text)
        if len(essays) >= N_ESSAYS:
            break
    print(f"[data] Collected {len(essays)} essays.\n", flush=True)
    return essays


def generate_rewrite(essay: str, tok, model, device: torch.device) -> str:
    prompt_text = build_prompt(essay, tok)
    enc = tok(
        prompt_text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    prompt_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    gen_ids = out[0, prompt_len:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


def perplexity_score(text: str, fl_tok, fl_mdl, device: torch.device) -> float:
    """
    Mean per-token cross-entropy perplexity under the frozen Llama-3.2-3B base.
    Returns exp(mean NLL). Higher = more human-like (less predictable to the LM).
    """
    inputs = fl_tok(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        loss = fl_mdl(**inputs, labels=inputs["input_ids"]).loss.item()
    return math.exp(loss)


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    os.makedirs(os.path.join(_REPO_ROOT, "outputs"), exist_ok=True)

    print("=" * 64, flush=True)
    print("  E1: Multi-Detector Evaluation", flush=True)
    print("=" * 64 + "\n", flush=True)

    device = get_device()
    el_tok, el_mdl = load_editlens(device)
    fl_tok, fl_mdl, _ = load_fluency_model(device)
    policy_tok, policy_mdl = load_policy(device)
    essays = load_essays()

    records: list[dict] = []

    print(
        f"{'#':>3}  {'EL-orig':>8}  {'EL-rew':>8}  {'PPL-orig':>9}  {'PPL-rew':>9}",
        flush=True,
    )
    print("-" * 50, flush=True)

    for i, essay in enumerate(essays, start=1):
        print(f"\n[essay {i}/{N_ESSAYS}] Generating rewrite ...", flush=True)
        rewrite = generate_rewrite(essay, policy_tok, policy_mdl, device)

        el_orig  = editlens_score(el_tok, el_mdl, essay,   device)
        el_rew   = editlens_score(el_tok, el_mdl, rewrite, device)
        ppl_orig = perplexity_score(essay,   fl_tok, fl_mdl, device)
        ppl_rew  = perplexity_score(rewrite, fl_tok, fl_mdl, device)

        record = {
            "essay_num":    i,
            "essay":        essay,
            "rewrite":      rewrite,
            "el_original":  el_orig,
            "el_rewrite":   el_rew,
            "ppl_original": ppl_orig,
            "ppl_rewrite":  ppl_rew,
        }
        records.append(record)

        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(
            f"{i:>3}  {el_orig:>8.4f}  {el_rew:>8.4f}  {ppl_orig:>9.2f}  {ppl_rew:>9.2f}",
            flush=True,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    mean_el_orig  = mean([r["el_original"]  for r in records])
    mean_el_rew   = mean([r["el_rewrite"]   for r in records])
    mean_ppl_orig = mean([r["ppl_original"] for r in records])
    mean_ppl_rew  = mean([r["ppl_rewrite"]  for r in records])

    header = f"{'#':>3}  {'EL-orig':>8}  {'EL-rew':>8}  {'PPL-orig':>9}  {'PPL-rew':>9}"
    sep    = "-" * 50

    lines = [
        "=" * 64,
        "  E1: Multi-Detector Results",
        "  (EditLens: 0=human, 1=AI | Perplexity: higher=more human-like)",
        "=" * 64,
        "",
        header,
        sep,
    ]
    for r in records:
        lines.append(
            f"{r['essay_num']:>3}  {r['el_original']:>8.4f}  {r['el_rewrite']:>8.4f}  "
            f"{r['ppl_original']:>9.2f}  {r['ppl_rewrite']:>9.2f}"
        )
    lines += [
        sep,
        f"{'mean':>3}  {mean_el_orig:>8.4f}  {mean_el_rew:>8.4f}  "
        f"{mean_ppl_orig:>9.2f}  {mean_ppl_rew:>9.2f}",
        "",
        f"Results saved to: {RESULTS_PATH}",
    ]

    summary = "\n".join(lines)
    print("\n" + summary, flush=True)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\nSummary saved to: {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
