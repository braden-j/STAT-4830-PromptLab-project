"""
probe_editlens.py

Loads pangram/editlens_roberta-large and scores 5 real inputs pulled from
pangram/editlens_iclr (the EditLens training corpus):
  - 2 human_written essays
  - 2 ai_generated essays (GPT-4.1 / Gemini / Claude Sonnet 4)
  - 1 ai_edited essay (AI-lightly-edited human text: best proxy available
    for a deslopifier / hill-climb output in this environment)

Uses the canonical EditLens inference formula from the official repo:
    probs       = softmax(logits)
    bucket_pred = argmax(probs)
    score_pred  = (probs @ [0,1,2,3]) / (n_buckets - 1)  -> float in [0, 1]

score_pred: 0.0 = fully human  |  1.0 = fully AI-generated

NOTE: Live generation from Qwen/TinyLlama is blocked in this environment
because transformers 5.5.4 loads torch._dynamo which pulls in onnxruntime,
which crashes against NumPy 2.x. Pulling from the EditLens corpus is cleaner
for demonstrating the full score range anyway (GPT-4.1/Claude/Gemini vs human).
"""

from __future__ import annotations

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EDITLENS_MODEL_ID = "pangram/editlens_roberta-large"
EDITLENS_DATASET  = "pangram/editlens_iclr"
N_BUCKETS = 4
BUCKET_LABELS = np.array([0, 1, 2, 3], dtype=np.float32)
BUCKET_NAMES = ["fully human", "lightly AI-edited", "heavily AI-edited", "fully AI-generated"]

MIN_WORDS = 80   # skip very short excerpts
MAX_WORDS = 300  # keep inputs manageable


# ---------------------------------------------------------------------------
# Step 1: Pull samples from pangram/editlens_iclr
# ---------------------------------------------------------------------------
def fetch_samples(
    n_human: int = 2,
    n_ai_gen: int = 2,
    n_ai_edited: int = 1,
) -> list[tuple[str, str]]:
    """Stream pangram/editlens_iclr train split and collect requested samples.

    Returns list of (label, text) pairs.
    """
    targets = {
        "human_written": n_human,
        "ai_generated":  n_ai_gen,
        "ai_edited":     n_ai_edited,
    }
    collected: dict[str, list] = {k: [] for k in targets}
    needed = dict(targets)

    print(f"[data] Streaming {EDITLENS_DATASET} train split...")
    ds = load_dataset(EDITLENS_DATASET, split="train", streaming=True)

    for row in ds:
        tt = row.get("text_type", "")
        if tt not in needed or needed[tt] <= 0:
            continue
        text = row["text"].strip()
        words = text.split()
        if len(words) < MIN_WORDS:
            continue
        # Truncate very long texts to MAX_WORDS words (first chunk)
        if len(words) > MAX_WORDS:
            text = " ".join(words[:MAX_WORDS])
        model_tag = row.get("model", "human")
        source_tag = row.get("source", "?")
        label = f"{tt.upper()} [{model_tag} / {source_tag}]"
        collected[tt].append((label, text))
        needed[tt] -= 1
        if all(v <= 0 for v in needed.values()):
            break

    samples: list[tuple[str, str]] = []
    for tt in ["human_written", "ai_generated", "ai_edited"]:
        samples.extend(collected[tt])

    print(f"[data] Collected: "
          f"{len(collected['human_written'])} human, "
          f"{len(collected['ai_generated'])} ai_generated, "
          f"{len(collected['ai_edited'])} ai_edited\n")
    return samples


# ---------------------------------------------------------------------------
# Step 2: Load EditLens
# ---------------------------------------------------------------------------
def load_editlens():
    print(f"[editlens] Loading: {EDITLENS_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(EDITLENS_MODEL_ID, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(EDITLENS_MODEL_ID)
    model.eval()
    print(f"[editlens] id2label: {model.config.id2label}\n")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Step 3: Score with canonical EditLens inference pipeline
# ---------------------------------------------------------------------------
def score(tokenizer, model, text: str) -> dict:
    """Run the canonical EditLens inference formula."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits          # shape (1, 4), raw logits

    # Avoid .numpy() — broken in this env (NumPy 2.x / torch NumPy bridge issue).
    # Do all arithmetic in torch, then extract scalars with .item() / .tolist().
    probs_t = torch.softmax(logits, dim=-1)[0]         # shape (4,)
    probs   = probs_t.tolist()                          # plain Python list of floats
    bucket_pred = int(probs_t.argmax().item())
    # Canonical formula: weighted sum of bucket indices, normalized to [0,1]
    score_pred  = sum(p * i for i, p in enumerate(probs)) / (N_BUCKETS - 1)

    return {
        "logits":      logits[0].tolist(),
        "probs":       probs,        # already a Python list from .tolist()
        "bucket_pred": bucket_pred,
        "score_pred":  score_pred,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def print_result(label: str, text: str, r: dict) -> None:
    sep = "=" * 72
    print(sep)
    print(f"  {label}")
    print(sep)
    word_count = len(text.split())
    print(f"  Text ({word_count} words): {text[:140].replace(chr(10), ' ')}...")
    print()
    print(f"  Raw logits : {[f'{v:+.3f}' for v in r['logits']]}")
    print(f"  Probs      : {[f'{v:.4f}' for v in r['probs']]}")
    print()
    print(f"  Bucket distribution:")
    for i, p in enumerate(r["probs"]):
        bar = "#" * int(round(p * 30)) + "." * (30 - int(round(p * 30)))
        marker = " <-- predicted" if i == r["bucket_pred"] else ""
        print(f"    LABEL_{i} ({BUCKET_NAMES[i]:22s}): {p:.4f}  |{bar}|{marker}")
    print()
    print(f"  bucket_pred : {r['bucket_pred']}  ({BUCKET_NAMES[r['bucket_pred']]})")
    print(f"  score_pred  : {r['score_pred']:.4f}  (0=human ... 1=AI)")
    print()


def main():
    samples = fetch_samples(n_human=2, n_ai_gen=2, n_ai_edited=1)

    tokenizer, model = load_editlens()

    print("\n" + "=" * 72)
    print("  EDITLENS SCORES  (softmax -> weighted avg -> normalize)")
    print("=" * 72 + "\n")

    results = []
    for label, text in samples:
        r = score(tokenizer, model, text)
        print_result(label, text, r)
        results.append((label, r))

    # Summary table
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  {'Source':<50} {'bucket':>6}  {'score_pred':>10}")
    print(f"  {'-'*50} {'-'*6}  {'-'*10}")
    for label, r in results:
        short = label[:50]
        print(f"  {short:<50} {r['bucket_pred']:>6}  {r['score_pred']:>10.4f}")
    print()
    print("  score_pred: 0.0 = fully human  |  1.0 = fully AI-generated")
    print("  Spec success criteria: human < 0.3, raw AI slop > 0.7")


if __name__ == "__main__":
    main()
