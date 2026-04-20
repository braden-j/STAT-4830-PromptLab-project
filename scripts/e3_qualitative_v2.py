"""
scripts/e3_qualitative_v2.py — Experiment E3: Qualitative evaluation of step_500 checkpoint

Loads the step_500 LoRA checkpoint on top of meta-llama/Llama-3.2-3B-Instruct,
generates rewrites for 10 ai_generated essays from the pangram/editlens_iclr
test split using greedy decoding, and compares original vs. rewritten EditLens scores.

Output: outputs/e3_qualitative_v2.txt
"""

from __future__ import annotations

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

from c2_reinforce import build_prompt, POLICY_MODEL_ID
from fluency_reward_validation import editlens_score, load_editlens

CHECKPOINT_PATH = os.path.join(_REPO_ROOT, "outputs", "c2_checkpoints", "step_500")
OUT_PATH        = os.path.join(_REPO_ROOT, "outputs", "e3_qualitative_v2.txt")
N_ESSAYS        = 20
MAX_NEW_TOKENS  = 512
MIN_WORDS       = 80
MAX_WORDS       = 250


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
        POLICY_MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, CHECKPOINT_PATH, is_trainable=False)
    model = model.to(device).eval()
    print("[policy] Loaded.\n", flush=True)
    return tok, model


def load_test_essays() -> list[str]:
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
    prompt = build_prompt(essay, tok)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    prompt_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tok.eos_token_id,
        )
    gen_ids = out[0, prompt_len:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


def format_pair(
    idx: int,
    original: str,
    rewrite: str,
    orig_score: float,
    rew_score: float,
) -> str:
    sep = "=" * 72
    thin = "-" * 72
    delta = rew_score - orig_score
    sign = "+" if delta >= 0 else ""
    lines = [
        sep,
        f"Essay {idx:02d}",
        sep,
        "",
        "ORIGINAL TEXT:",
        thin,
        original,
        "",
        f"EditLens score (original): {orig_score:.4f}  (0=human, 1=AI)",
        "",
        "REWRITTEN TEXT:",
        thin,
        rewrite,
        "",
        f"EditLens score (rewrite):  {rew_score:.4f}  (Δ = {sign}{delta:.4f})",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    print("=" * 64, flush=True)
    print("  E3 Qualitative Evaluation — step_500 checkpoint", flush=True)
    print("=" * 64 + "\n", flush=True)

    device = get_device()
    el_tok, el_mdl = load_editlens(device)
    policy_tok, policy_mdl = load_policy(device)
    essays = load_test_essays()

    blocks: list[str] = []
    summary_rows: list[tuple[int, float, float]] = []

    for i, essay in enumerate(essays, start=1):
        print(f"[{i:02d}/{N_ESSAYS}] Generating rewrite ...", flush=True)
        rewrite = generate_rewrite(essay, policy_tok, policy_mdl, device)

        print(f"[{i:02d}/{N_ESSAYS}] Scoring ...", flush=True)
        orig_score = editlens_score(el_tok, el_mdl, essay,   device)
        rew_score  = editlens_score(el_tok, el_mdl, rewrite, device)

        print(
            f"[{i:02d}/{N_ESSAYS}] orig={orig_score:.4f}  "
            f"rewrite={rew_score:.4f}  Δ={rew_score - orig_score:+.4f}\n",
            flush=True,
        )

        blocks.append(format_pair(i, essay, rewrite, orig_score, rew_score))
        summary_rows.append((i, orig_score, rew_score))

    # Build summary table
    sep = "=" * 72
    summary_lines = [
        sep,
        "SUMMARY",
        sep,
        f"{'Essay':>7}  {'Orig EditLens':>13}  {'Rew EditLens':>12}  {'Delta':>8}",
        "-" * 48,
    ]
    for idx, orig, rew in summary_rows:
        delta = rew - orig
        sign  = "+" if delta >= 0 else ""
        summary_lines.append(
            f"{idx:>7}  {orig:>13.4f}  {rew:>12.4f}  {sign}{delta:>7.4f}"
        )
    mean_orig = sum(r[1] for r in summary_rows) / len(summary_rows)
    mean_rew  = sum(r[2] for r in summary_rows) / len(summary_rows)
    mean_d    = mean_rew - mean_orig
    sign      = "+" if mean_d >= 0 else ""
    summary_lines += [
        "-" * 48,
        f"{'MEAN':>7}  {mean_orig:>13.4f}  {mean_rew:>12.4f}  {sign}{mean_d:>7.4f}",
        sep,
    ]

    output = "\n".join(blocks) + "\n" + "\n".join(summary_lines) + "\n"

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"\nSaved {N_ESSAYS} pairs → {OUT_PATH}", flush=True)
    print(f"Mean EditLens: {mean_orig:.4f} → {mean_rew:.4f}  (Δ {sign}{mean_d:.4f})", flush=True)


if __name__ == "__main__":
    main()
