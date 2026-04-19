"""
scripts/e1_multidetector.py — Experiment E1: Multi-detector evaluation

Load step_500 LoRA checkpoint and score 20 ai_generated essays with two detectors:
  - EditLens    (pangram/editlens_roberta-large): 0=human, 1=AI
  - Binoculars  (binoculars-ai): lower score = AI-generated

For each essay: generate a rewrite, then score both original and rewrite.
Print a table and save results.

Usage:
    python scripts/e1_multidetector.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import torch
from binoculars import Binoculars
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from fluency_reward_validation import editlens_score, load_editlens
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


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    os.makedirs(os.path.join(_REPO_ROOT, "outputs"), exist_ok=True)

    print("=" * 64, flush=True)
    print("  E1: Multi-Detector Evaluation", flush=True)
    print("=" * 64 + "\n", flush=True)

    device = get_device()
    el_tok, el_mdl = load_editlens(device)
    print("[binoculars] Initializing Binoculars ...", flush=True)
    bino = Binoculars()
    print("[binoculars] Ready.\n", flush=True)
    policy_tok, policy_mdl = load_policy(device)
    essays = load_essays()

    records: list[dict] = []

    print(
        f"{'#':>3}  {'EL-orig':>8}  {'EL-rew':>8}  {'BINO-orig':>9}  {'BINO-rew':>9}",
        flush=True,
    )
    print("-" * 50, flush=True)

    for i, essay in enumerate(essays, start=1):
        print(f"\n[essay {i}/{N_ESSAYS}] Generating rewrite ...", flush=True)
        rewrite = generate_rewrite(essay, policy_tok, policy_mdl, device)

        el_orig   = editlens_score(el_tok, el_mdl, essay,   device)
        el_rew    = editlens_score(el_tok, el_mdl, rewrite, device)
        bino_orig = float(bino.compute_score(essay))
        bino_rew  = float(bino.compute_score(rewrite))

        record = {
            "essay_num":    i,
            "essay":        essay,
            "rewrite":      rewrite,
            "el_original":  el_orig,
            "el_rewrite":   el_rew,
            "bino_original": bino_orig,
            "bino_rewrite":  bino_rew,
        }
        records.append(record)

        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(
            f"{i:>3}  {el_orig:>8.4f}  {el_rew:>8.4f}  {bino_orig:>9.4f}  {bino_rew:>9.4f}",
            flush=True,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    mean_el_orig   = mean([r["el_original"]   for r in records])
    mean_el_rew    = mean([r["el_rewrite"]    for r in records])
    mean_bino_orig = mean([r["bino_original"] for r in records])
    mean_bino_rew  = mean([r["bino_rewrite"]  for r in records])

    header = f"{'#':>3}  {'EL-orig':>8}  {'EL-rew':>8}  {'BINO-orig':>9}  {'BINO-rew':>9}"
    sep    = "-" * 50

    lines = [
        "=" * 64,
        "  E1: Multi-Detector Results",
        "  (EditLens: 0=human, 1=AI | Binoculars: lower=AI)",
        "=" * 64,
        "",
        header,
        sep,
    ]
    for r in records:
        lines.append(
            f"{r['essay_num']:>3}  {r['el_original']:>8.4f}  {r['el_rewrite']:>8.4f}  "
            f"{r['bino_original']:>9.4f}  {r['bino_rewrite']:>9.4f}"
        )
    lines += [
        sep,
        f"{'mean':>3}  {mean_el_orig:>8.4f}  {mean_el_rew:>8.4f}  "
        f"{mean_bino_orig:>9.4f}  {mean_bino_rew:>9.4f}",
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
