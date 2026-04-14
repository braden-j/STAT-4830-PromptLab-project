"""
fluency_reward_validation.py  —  Experiment A2

Implements and validates the fluency reward from the spec (Section 4, A2):

  1. Load frozen meta-llama/Llama-3.2-3B base (no fine-tuning) for fluency.
  2. Load pangram/editlens_roberta-large for EditLens detection scores.
  3. Pull 50 essays from pangram/editlens_iclr — balanced across
     human_written / ai_generated / ai_edited.
  4. Compute:
       - EditLens score_pred  (0=human, 1=AI) using the canonical formula
       - R_fluency = mean per-token log-probability under the frozen base model
  5. Scatter plot: EditLens score_pred (x) vs R_fluency (y), colored by text_type.
  6. Print Pearson and Spearman correlations between the two signals.

Fluency reward formula (spec §4 A2):
    outputs = model_frozen(**inputs, labels=input_ids)
    R_fluency = -outputs.loss.item()   # negate NLL → higher = more fluent

EditLens formula (spec §2.4, confirmed by probe_editlens.py):
    probs      = softmax(logits)[0]
    score_pred = sum(p*i for i,p in enumerate(probs.tolist())) / 3.0

Usage:
    python scripts/fluency_reward_validation.py

Output:
    scripts/fluency_vs_editlens.png   — scatter plot
    Correlations printed to stdout.
"""

from __future__ import annotations

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
from scipy import stats
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Fluency model preference order:
#   1. meta-llama/Llama-3.2-3B  (spec target; gated — needs Meta license)
#   2. Qwen/Qwen2.5-3B          (same 3B scale; open; needs ~12 GB CPU RAM)
#   3. Qwen/Qwen2.5-0.5B        (local fallback; fits in ~2 GB CPU RAM)
# On H100 (80 GB), use option 1 or 2.  On a laptop, option 3 is used automatically.
FLUENCY_CANDIDATES = [
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-0.5B",
]
EDITLENS_MODEL_ID = "pangram/editlens_roberta-large"
EDITLENS_DATASET  = "pangram/editlens_iclr"
N_BUCKETS         = 4

# How many essays per text_type (total ≈ 50)
N_HUMAN    = 17
N_AI_GEN   = 17
N_AI_EDIT  = 16

MIN_WORDS  = 80
MAX_WORDS  = 300

# Output path for scatter plot (relative to repo root, written from this script's dir)
import os
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH    = os.path.join(_SCRIPT_DIR, "fluency_vs_editlens.png")


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[device] CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[device] No GPU - running on CPU (Llama 3B will be slow)")
    return dev


# ---------------------------------------------------------------------------
# Step 1: Pull 50 balanced essays from the EditLens corpus
# ---------------------------------------------------------------------------
def fetch_samples(
    n_human: int = N_HUMAN,
    n_ai_gen: int = N_AI_GEN,
    n_ai_edited: int = N_AI_EDIT,
) -> list[dict]:
    """
    Stream pangram/editlens_iclr train split and return a balanced list of dicts:
        {"text": str, "text_type": str, "model": str, "source": str}
    """
    targets = {
        "human_written": n_human,
        "ai_generated":  n_ai_gen,
        "ai_edited":     n_ai_edited,
    }
    collected: dict[str, list] = {k: [] for k in targets}
    needed = dict(targets)

    n_total = n_human + n_ai_gen + n_ai_edited
    print(f"\n[data] Streaming {EDITLENS_DATASET} - collecting {n_total} essays "
          f"({n_human} human / {n_ai_gen} ai_generated / {n_ai_edited} ai_edited)...")

    ds = load_dataset(EDITLENS_DATASET, split="train", streaming=True)

    for row in ds:
        tt = row.get("text_type", "")
        if tt not in needed or needed[tt] <= 0:
            continue
        text = row["text"].strip()
        words = text.split()
        if len(words) < MIN_WORDS:
            continue
        if len(words) > MAX_WORDS:
            text = " ".join(words[:MAX_WORDS])
        collected[tt].append({
            "text":      text,
            "text_type": tt,
            "model":     row.get("model", "human"),
            "source":    row.get("source", "?"),
        })
        needed[tt] -= 1
        if all(v <= 0 for v in needed.values()):
            break

    samples = []
    for tt in ["human_written", "ai_generated", "ai_edited"]:
        samples.extend(collected[tt])

    counts = {tt: len(collected[tt]) for tt in targets}
    print(f"[data] Collected: {counts}\n")
    return samples


# ---------------------------------------------------------------------------
# Step 2: Load models
# ---------------------------------------------------------------------------
def load_editlens(device: torch.device):
    """Load pangram/editlens_roberta-large for sequence classification."""
    print(f"[editlens] Loading {EDITLENS_MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(EDITLENS_MODEL_ID, use_fast=False)
    mdl = AutoModelForSequenceClassification.from_pretrained(EDITLENS_MODEL_ID)
    mdl = mdl.to(device).eval()
    print(f"[editlens] Loaded ({sum(p.numel() for p in mdl.parameters()):,} params)\n")
    return tok, mdl


def _available_ram_gb() -> float:
    """Return available system RAM in GB (best-effort)."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return 8.0   # conservative assumption


def load_fluency_model(device: torch.device):
    """
    Load frozen causal LM for log-prob fluency.

    Tries FLUENCY_CANDIDATES in order.  Skips models that need more RAM
    than is currently available (fp32 estimate: n_params * 4 bytes).
    On H100 (80 GB VRAM + bfloat16): Llama 3.2 3B loads fine.
    On CPU with limited RAM: falls back to Qwen 0.5B.
    """
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    bytes_per_param = 2 if dtype == torch.bfloat16 else 4
    available_gb = _available_ram_gb() if device.type == "cpu" else 999.0

    # Rough param counts (billions) to pre-filter by RAM
    approx_params_b = {
        "meta-llama/Llama-3.2-3B": 3.21,
        "Qwen/Qwen2.5-3B":         3.09,
        "Qwen/Qwen2.5-0.5B":       0.49,
    }

    for model_id in FLUENCY_CANDIDATES:
        est_gb = approx_params_b.get(model_id, 3.0) * 1e9 * bytes_per_param / (1024**3)
        if device.type == "cpu" and est_gb > available_gb * 0.75:
            print(f"[fluency] Skipping {model_id} "
                  f"(~{est_gb:.1f} GB needed, {available_gb:.1f} GB available)")
            continue

        try:
            print(f"[fluency] Trying {model_id} ...")
            tok = AutoTokenizer.from_pretrained(model_id)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            mdl = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            mdl = mdl.to(device).eval()
            for p in mdl.parameters():
                p.requires_grad_(False)
            n_params = sum(p.numel() for p in mdl.parameters())
            spec_model = "meta-llama/Llama-3.2-3B"
            if model_id != spec_model:
                print(
                    f"\n[fluency] NOTE: Using {model_id} as local fallback.\n"
                    f"[fluency] On H100, run with {spec_model} (spec sec.4 A2).\n"
                    f"[fluency] Fluency-signal semantics are identical - both measure\n"
                    f"[fluency] mean per-token NLL under a frozen LM.\n"
                )
            print(f"[fluency] Loaded {model_id} ({n_params/1e9:.3f}B params, dtype={dtype})\n")
            return tok, mdl, model_id
        except Exception as e:
            print(f"[fluency] {model_id} failed: {type(e).__name__}: {str(e)[:120]}")
            continue

    raise RuntimeError(
        "Could not load any fluency model.  "
        "On H100: accept the Meta Llama license at "
        "https://huggingface.co/meta-llama/Llama-3.2-3B .  "
        "Locally: ensure psutil is installed and Qwen/Qwen2.5-0.5B is reachable."
    )


# ---------------------------------------------------------------------------
# Step 3: Scoring functions
# ---------------------------------------------------------------------------
def editlens_score(tok, mdl, text: str, device: torch.device) -> float:
    """
    Canonical EditLens inference formula (spec §2.4).
    Returns score_pred in [0, 1]:  0.0 = fully human, 1.0 = fully AI.
    """
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = mdl(**inputs).logits        # shape (1, 4)
    probs_t = torch.softmax(logits, dim=-1)[0]   # shape (4,)
    probs = probs_t.tolist()                      # avoid NumPy 2.x bridge issues
    score = sum(p * i for i, p in enumerate(probs)) / (N_BUCKETS - 1)
    return float(score)


def fluency_reward(tok, mdl, text: str, device: torch.device) -> float:
    """
    Mean per-token log-probability under the frozen Llama 3.2 3B base model.
    Formula from spec §4 A2:
        R_fluency = -outputs.loss.item()
    Higher = more fluent / natural.  Returns a negative float.
    """
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = mdl(**inputs, labels=input_ids)
    # outputs.loss = mean negative log-likelihood per token
    return float(-outputs.loss.item())


# ---------------------------------------------------------------------------
# Step 4: Score all essays
# ---------------------------------------------------------------------------
def score_all(
    samples: list[dict],
    el_tok, el_mdl,
    fl_tok, fl_mdl,
    device: torch.device,
) -> list[dict]:
    results = []
    n = len(samples)
    for i, s in enumerate(samples):
        text = s["text"]
        tt   = s["text_type"]
        sp   = editlens_score(el_tok, el_mdl, text, device)
        rf   = fluency_reward(fl_tok, fl_mdl, text, device)
        results.append({
            "text_type":  tt,
            "model":      s["model"],
            "source":     s["source"],
            "score_pred": sp,
            "R_fluency":  rf,
            "n_words":    len(text.split()),
        })
        bar = "#" * int(round((i + 1) / n * 30))
        print(
            f"  [{i+1:2d}/{n}] {tt:<16}  "
            f"EditLens={sp:.4f}  R_fluency={rf:.4f}  "
            f"[{bar:<30}]",
            flush=True,
        )
    return results


# ---------------------------------------------------------------------------
# Step 5: Correlations
# ---------------------------------------------------------------------------
def compute_correlations(results: list[dict]) -> None:
    xs = [r["score_pred"] for r in results]
    ys = [r["R_fluency"]  for r in results]

    pearson_r, pearson_p   = stats.pearsonr(xs, ys)
    spearman_r, spearman_p = stats.spearmanr(xs, ys)

    print("\n" + "=" * 60)
    print("  CORRELATIONS:  EditLens score_pred  vs  R_fluency")
    print("=" * 60)
    print(f"  Pearson  r = {pearson_r:+.4f}   (p = {pearson_p:.4e})")
    print(f"  Spearman r = {spearman_r:+.4f}   (p = {spearman_p:.4e})")
    print()
    if abs(pearson_r) > 0.8:
        print("  Interpretation: HIGH correlation - signals are near-redundant.")
        print("  Spec guidance: reduce beta to 0.2.")
    elif pearson_r < -0.1:
        print("  Interpretation: ANTI-correlated - genuine tension between")
        print("  detection evasion and fluency.  Keep beta=0.5, watch KL ablation.")
    else:
        print("  Interpretation: MODERATE/LOW correlation - signals are complementary.")
        print("  Spec guidance: keep beta=0.5.")
    print("=" * 60)

    return pearson_r, spearman_r


# ---------------------------------------------------------------------------
# Step 6: Scatter plot
# ---------------------------------------------------------------------------
COLOR_MAP = {
    "human_written": "#2196F3",   # blue
    "ai_generated":  "#F44336",   # red
    "ai_edited":     "#FF9800",   # orange
}
LABEL_MAP = {
    "human_written": "human_written",
    "ai_generated":  "ai_generated",
    "ai_edited":     "ai_edited",
}


def scatter_plot(results: list[dict], pearson_r: float, spearman_r: float,
                 fluency_model_id: str = "meta-llama/Llama-3.2-3B") -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    groups: dict[str, dict[str, list]] = {
        tt: {"x": [], "y": []} for tt in COLOR_MAP
    }
    for r in results:
        tt = r["text_type"]
        groups[tt]["x"].append(r["score_pred"])
        groups[tt]["y"].append(r["R_fluency"])

    for tt, data in groups.items():
        ax.scatter(
            data["x"], data["y"],
            c=COLOR_MAP[tt],
            label=LABEL_MAP[tt],
            alpha=0.75,
            edgecolors="white",
            linewidths=0.5,
            s=70,
        )

    ax.set_xlabel("EditLens score_pred  (0 = human, 1 = AI)", fontsize=12)
    ax.set_ylabel(f"R_fluency  (mean per-token log-prob, frozen {fluency_model_id})",
                  fontsize=10)
    ax.set_title(
        "EditLens Detection Score vs. Fluency Reward\n"
        f"(Pearson r = {pearson_r:+.3f},  Spearman ρ = {spearman_r:+.3f})",
        fontsize=13,
    )
    ax.legend(title="text_type", fontsize=10, title_fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"\n[plot] Saved -> {PLOT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Experiment A2 - Fluency Reward Validation")
    print("  (spec sec.4, STAT 4830 PromptLab Deslopifier)")
    print("=" * 60)

    device = get_device()

    # 1. Data
    samples = fetch_samples()

    # 2. Load models
    el_tok, el_mdl = load_editlens(device)
    fl_tok, fl_mdl, fluency_model_id = load_fluency_model(device)

    # 3. Score all essays
    print(f"\n[scoring] Scoring {len(samples)} essays ...\n")
    results = score_all(samples, el_tok, el_mdl, fl_tok, fl_mdl, device)

    # 4. Correlations
    pearson_r, spearman_r = compute_correlations(results)

    # 5. Scatter plot
    scatter_plot(results, pearson_r, spearman_r, fluency_model_id=fluency_model_id)

    # 6. Summary table
    print("\n" + "=" * 72)
    print("  SUMMARY TABLE")
    print("=" * 72)
    print(f"  {'text_type':<16} {'model':<20} {'score_pred':>10}  {'R_fluency':>10}")
    print(f"  {'-'*16} {'-'*20} {'-'*10}  {'-'*10}")
    for r in sorted(results, key=lambda x: x["score_pred"]):
        print(
            f"  {r['text_type']:<16} {r['model'][:20]:<20} "
            f"{r['score_pred']:>10.4f}  {r['R_fluency']:>10.4f}"
        )
    print()
    print("  score_pred: 0.0=fully human  |  1.0=fully AI")
    print("  R_fluency : higher is more fluent (less negative log-prob)")


if __name__ == "__main__":
    main()
