"""
scripts/e1_multidetector.py — Experiment E1: Multi-detector evaluation

Load step_500 LoRA checkpoint and score 20 ai_generated essays with two detectors:
  - EditLens (pangram/editlens_roberta-large)
  - GPTZero  (/v2/predict/text, completely_generated_prob field)

For each essay: generate a rewrite, then score both original and rewrite.
Print a table and save results.

Usage:
    python scripts/e1_multidetector.py --gptzero-key <KEY>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import requests
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from fluency_reward_validation import editlens_score, load_editlens
from c2_reinforce import build_prompt, POLICY_MODEL_ID

CHECKPOINT_PATH = os.path.join(_REPO_ROOT, "outputs", "c2_checkpoints", "step_500")
RESULTS_PATH    = os.path.join(_REPO_ROOT, "outputs", "e1_multidetector.jsonl")
SUMMARY_PATH    = os.path.join(_REPO_ROOT, "outputs", "e1_summary.txt")

N_ESSAYS      = 20
MAX_NEW_TOKENS = 512
MIN_WORDS      = 80
MAX_WORDS      = 250

GPTZERO_URL = "https://api.gptzero.me/v2/predict/text"


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


def gptzero_score(text: str, api_key: str, retries: int = 3) -> float | None:
    """Call GPTZero /v2/predict/text and return completely_generated_prob."""
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-key": api_key,
    }
    payload = {"document": text}
    for attempt in range(retries):
        try:
            resp = requests.post(GPTZERO_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return float(data["documents"][0]["completely_generated_prob"])
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [gptzero] attempt {attempt+1} failed: {e} — retrying in {wait}s", flush=True)
            time.sleep(wait)
    return None


def fmt_or_na(val: float | None, decimals: int = 4) -> str:
    return f"{val:.{decimals}f}" if val is not None else "N/A"


def main() -> None:
    parser = argparse.ArgumentParser(description="E1: Multi-detector evaluation")
    parser.add_argument("--gptzero-key", required=True, help="GPTZero API key")
    args = parser.parse_args()

    os.makedirs(os.path.join(_REPO_ROOT, "outputs"), exist_ok=True)

    print("=" * 64, flush=True)
    print("  E1: Multi-Detector Evaluation", flush=True)
    print("=" * 64 + "\n", flush=True)

    device = get_device()
    el_tok, el_mdl = load_editlens(device)
    policy_tok, policy_mdl = load_policy(device)
    essays = load_essays()

    records: list[dict] = []

    print(f"{'#':>3}  {'EL-orig':>8}  {'EL-rew':>8}  {'GZ-orig':>8}  {'GZ-rew':>8}", flush=True)
    print("-" * 46, flush=True)

    for i, essay in enumerate(essays, start=1):
        print(f"\n[essay {i}/{N_ESSAYS}] Generating rewrite ...", flush=True)
        rewrite = generate_rewrite(essay, policy_tok, policy_mdl, device)

        el_orig = editlens_score(el_tok, el_mdl, essay,   device)
        el_rew  = editlens_score(el_tok, el_mdl, rewrite, device)

        print(f"  [essay {i}] Scoring with GPTZero (original) ...", flush=True)
        gz_orig = gptzero_score(essay,   args.gptzero_key)
        time.sleep(1)  # be polite to the API
        print(f"  [essay {i}] Scoring with GPTZero (rewrite) ...", flush=True)
        gz_rew  = gptzero_score(rewrite, args.gptzero_key)
        time.sleep(1)

        record = {
            "essay_num":    i,
            "essay":        essay,
            "rewrite":      rewrite,
            "el_original":  el_orig,
            "el_rewrite":   el_rew,
            "gz_original":  gz_orig,
            "gz_rewrite":   gz_rew,
        }
        records.append(record)

        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(
            f"{i:>3}  {el_orig:>8.4f}  {el_rew:>8.4f}  "
            f"{fmt_or_na(gz_orig):>8}  {fmt_or_na(gz_rew):>8}",
            flush=True,
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    valid = [r for r in records if r["gz_original"] is not None and r["gz_rewrite"] is not None]

    def mean(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    mean_el_orig = mean([r["el_original"] for r in records])
    mean_el_rew  = mean([r["el_rewrite"]  for r in records])
    mean_gz_orig = mean([r["gz_original"] for r in valid])
    mean_gz_rew  = mean([r["gz_rewrite"]  for r in valid])

    header = f"{'#':>3}  {'EL-orig':>8}  {'EL-rew':>8}  {'GZ-orig':>8}  {'GZ-rew':>8}"
    sep    = "-" * 46

    lines = [
        "=" * 64,
        "  E1: Multi-Detector Results",
        "  (EditLens: 0=human, 1=AI | GPTZero completely_generated_prob)",
        "=" * 64,
        "",
        header,
        sep,
    ]
    for r in records:
        lines.append(
            f"{r['essay_num']:>3}  {r['el_original']:>8.4f}  {r['el_rewrite']:>8.4f}  "
            f"{fmt_or_na(r['gz_original']):>8}  {fmt_or_na(r['gz_rewrite']):>8}"
        )
    lines += [
        sep,
        f"{'mean':>3}  {mean_el_orig:>8.4f}  {mean_el_rew:>8.4f}  "
        f"{fmt_or_na(mean_gz_orig):>8}  {fmt_or_na(mean_gz_rew):>8}",
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
