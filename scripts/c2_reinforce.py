"""
scripts/c2_reinforce.py — Experiment C2: 500-step REINFORCE training run

  Policy      : meta-llama/Llama-3.2-3B-Instruct + LoRA (r=16, α=32)
  Reward model: pangram/editlens_Llama-3.2-3B
  Fluency ref : meta-llama/Llama-3.2-3B base (frozen)
  Algorithm   : REINFORCE, 500 steps, 48 rollouts/step

  Eval every 50 steps on 100 held-out test essays (50 ai_generated, 50 ai_edited):
    - mean EditLens score
    - mean R_fluency
    - KL divergence from reference (token-level, on raw eval texts)
    - label distribution (LABEL_0/1/2/3 by EditLens argmax on rewrites)

  Checkpoints every 100 steps → outputs/c2_checkpoints/step_NNN/
  Training log → outputs/c2_log.jsonl
  Eval log     → outputs/c2_eval.jsonl

Run:
    python scripts/c2_reinforce.py
    python scripts/c2_reinforce.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")

# ── Make scripts/ importable ──────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from fluency_reward_validation import editlens_score, fluency_reward, load_editlens

# ── Hyperparameters ───────────────────────────────────────────────────────────
POLICY_MODEL_ID  = "meta-llama/Llama-3.2-3B-Instruct"
FLUENCY_MODEL_ID = "meta-llama/Llama-3.2-3B"

LORA_R       = 16
LORA_ALPHA   = 32
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_DROPOUT = 0.05

N_STEPS           = 500
N_ROLLOUTS        = 48
MAX_NEW_TOKENS    = 512
MIN_OUTPUT_TOKENS = 50
MIN_OUTPUT_RATIO  = 0.40

LR             = 1e-5
WEIGHT_DECAY   = 0.01
GRAD_CLIP_NORM = 1.0

# Defined for completeness; reward formula is unchanged from C1
KL_PENALTY = 0.1

# Reward weights (locked in spec §4 A3)
ALPHA = 1.0
BETA  = 0.5
GAMMA = 0.1

SEED = 42

EVAL_INTERVAL       = 50
EVAL_N_AI_GEN       = 50
EVAL_N_AI_EDIT      = 50
CHECKPOINT_INTERVAL = 100

# Output paths
CHECKPOINT_DIR = os.path.join(_REPO_ROOT, "outputs", "c2_checkpoints")
LOG_PATH       = os.path.join(_REPO_ROOT, "outputs", "c2_log.jsonl")
EVAL_LOG_PATH  = os.path.join(_REPO_ROOT, "outputs", "c2_eval.jsonl")

INSTRUCTION = (
    "Rewrite the following AI-generated essay to sound more natural and human. "
    "Preserve the same ideas but change the writing style so that it reads as "
    "if a person wrote it.\n\n"
    "Original:\n{essay}"
)

SYSTEM_PROMPT = (
    "You are a skilled human writing editor. Rewrite the following text to sound "
    "more like it was written by a thoughtful human. Make targeted, precise edits. "
    "Preserve the core meaning and approximate length."
)


# ── Prompt building (Llama-3.2-3B-Instruct chat format) ──────────────────────
def build_prompt(essay: str, tokenizer) -> str:
    """Format essay as a system+user turn using the Instruct chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": essay},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── Device ────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[device] CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        dev = torch.device("cpu")
        print("[device] CPU  (no GPU found — training will be slow)", flush=True)
    return dev


# ── Data ──────────────────────────────────────────────────────────────────────
def load_train_prompts(n: int = N_ROLLOUTS, min_words: int = 80, max_words: int = 250) -> list[str]:
    """Load N ai_generated essays from pangram/editlens_iclr train split."""
    print(f"[data]  Loading {n} ai_generated essays from train split ...", flush=True)
    ds = load_dataset("pangram/editlens_iclr", split="train", streaming=True)
    essays: list[str] = []
    for row in ds:
        if row.get("text_type") != "ai_generated":
            continue
        text = row["text"].strip()
        words = text.split()
        if len(words) < min_words:
            continue
        if len(words) > max_words:
            text = " ".join(words[:max_words])
        essays.append(text)
        if len(essays) >= n:
            break
    print(f"[data]  Collected {len(essays)} train prompts.\n", flush=True)
    return essays


def load_eval_essays(min_words: int = 80, max_words: int = 250) -> list[str]:
    """Load 50 ai_generated + 50 ai_edited from the test split for periodic eval."""
    total = EVAL_N_AI_GEN + EVAL_N_AI_EDIT
    print(f"[eval-data] Loading {EVAL_N_AI_GEN} ai_generated + {EVAL_N_AI_EDIT} ai_edited from test split ...", flush=True)
    ds = load_dataset("pangram/editlens_iclr", split="test", streaming=True)
    counts  = {"ai_generated": 0, "ai_edited": 0}
    targets = {"ai_generated": EVAL_N_AI_GEN, "ai_edited": EVAL_N_AI_EDIT}
    essays: list[str] = []
    for row in ds:
        tt = row.get("text_type", "")
        if tt not in targets or counts[tt] >= targets[tt]:
            continue
        text = row["text"].strip()
        words = text.split()
        if len(words) < min_words:
            continue
        if len(words) > max_words:
            text = " ".join(words[:max_words])
        essays.append(text)
        counts[tt] += 1
        if all(counts[k] >= targets[k] for k in targets):
            break
    print(f"[eval-data] Collected {counts}  (target: {total} essays)\n", flush=True)
    return essays


# ── Policy — Llama-3.2-3B-Instruct + LoRA ────────────────────────────────────
def load_policy(device: torch.device, checkpoint_path: str | None = None):
    """Load Llama-3.2-3B-Instruct with LoRA adapters. Optionally restore from checkpoint."""
    print(f"[policy] Loading {POLICY_MODEL_ID} ...", flush=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(POLICY_MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        POLICY_MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    if checkpoint_path is not None:
        print(f"[policy] Restoring LoRA from {checkpoint_path} ...", flush=True)
        model = PeftModel.from_pretrained(base, checkpoint_path, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGETS,
            lora_dropout=LORA_DROPOUT,
            bias="none",
        )
        model = get_peft_model(base, lora_cfg)

    model = model.to(device)
    model.train()

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"[policy] LoRA r={LORA_R} α={LORA_ALPHA} on {LORA_TARGETS}\n"
        f"[policy] Trainable: {n_train:,} / {n_total:,} params ({100*n_train/n_total:.3f}%)\n",
        flush=True,
    )
    return tok, model


# ── Fluency reference — frozen Llama-3.2-3B base ─────────────────────────────
def load_fluency_ref(device: torch.device):
    """Load meta-llama/Llama-3.2-3B base (frozen) for fluency scoring and KL eval."""
    print(f"[fluency] Loading {FLUENCY_MODEL_ID} (frozen) ...", flush=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(FLUENCY_MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        FLUENCY_MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    mdl = mdl.to(device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)

    n = sum(p.numel() for p in mdl.parameters())
    print(f"[fluency] Loaded {FLUENCY_MODEL_ID} ({n/1e9:.3f}B params, dtype={dtype}, frozen)\n", flush=True)
    return tok, mdl


# ── Combined reward (unchanged from C1) ──────────────────────────────────────
def combined_reward(
    editlens_val: float,
    fluency_normalized: float,
    output_length: int,
    input_length: int,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> float:
    """
    R = alpha * (1 - editlens_val)
      + beta  * fluency_normalized
      + gamma * length_reward
    Weights locked per spec §4 A3.
    """
    r_detection = 1.0 - editlens_val

    r_fluency = fluency_normalized

    ratio = output_length / max(input_length, 1)
    if 0.40 <= ratio <= 1.60:
        r_length = 1.0
    else:
        r_length = max(0.0, 1.0 - abs(ratio - 1.0))

    return alpha * r_detection + beta * r_fluency + gamma * r_length


# ── Minimum-length filter (unchanged from C1) ────────────────────────────────
def passes_min_length(n_gen_tokens: int, essay_token_len: int) -> bool:
    if n_gen_tokens < MIN_OUTPUT_TOKENS:
        return False
    if n_gen_tokens < MIN_OUTPUT_RATIO * essay_token_len:
        return False
    return True


# ── Eval helpers ──────────────────────────────────────────────────────────────
def editlens_argmax_label(tok, mdl, text: str, device: torch.device) -> int:
    """Return argmax class (0–3) from EditLens."""
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = mdl(**inputs).logits
    return int(logits.argmax(dim=-1).item())


def compute_kl_from_ref(
    policy_mdl,
    ref_mdl,
    policy_tok,
    texts: list[str],
    device: torch.device,
    max_len: int = 256,
) -> float:
    """
    Mean token-level KL(π_θ || π_ref) over texts.
    Encodes with policy_tok (shared Llama vocab); runs both models.
    """
    policy_mdl.eval()
    total_kl = 0.0
    n = 0
    with torch.no_grad():
        for text in texts:
            enc = policy_tok(
                text, return_tensors="pt", truncation=True, max_length=max_len
            ).to(device)
            if enc["input_ids"].shape[1] < 2:
                continue
            policy_logits = policy_mdl(**enc).logits[0, :-1]   # (T-1, V)
            ref_logits    = ref_mdl(**enc).logits[0, :-1]       # (T-1, V)
            v = min(policy_logits.shape[-1], ref_logits.shape[-1])
            policy_lp = F.log_softmax(policy_logits[..., :v], dim=-1)
            ref_lp    = F.log_softmax(ref_logits[..., :v], dim=-1)
            kl = (policy_lp.exp() * (policy_lp - ref_lp)).sum(dim=-1).mean().item()
            total_kl += kl
            n += 1
    policy_mdl.train()
    return total_kl / max(n, 1)


def run_eval(
    step: int,
    eval_essays: list[str],
    policy_tok,
    policy_mdl,
    el_tok, el_mdl,
    fl_tok, fl_mdl,
    device: torch.device,
) -> dict:
    """
    Generate rewrites for all eval_essays and compute eval metrics.
    EditLens score, R_fluency, and label distribution are measured on rewrites.
    KL divergence is measured on raw eval texts (policy vs frozen base).
    """
    n = len(eval_essays)
    print(f"\n[eval] Step {step}: scoring {n} held-out essays ...", flush=True)
    policy_mdl.eval()

    el_vals:   list[float] = []
    fl_vals:   list[float] = []
    label_dist = [0, 0, 0, 0]

    with torch.no_grad():
        for i, essay in enumerate(eval_essays):
            prompt_text = build_prompt(essay, policy_tok)
            enc = policy_tok(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)
            prompt_len = enc["input_ids"].shape[1]

            out = policy_mdl.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=policy_tok.eos_token_id,
            )
            gen_ids  = out[0, prompt_len:]
            gen_text = policy_tok.decode(gen_ids, skip_special_tokens=True).strip()

            if not gen_text:
                continue

            el    = editlens_score(el_tok, el_mdl, gen_text, device)
            fl    = fluency_reward(fl_tok, fl_mdl, gen_text, device)
            label = editlens_argmax_label(el_tok, el_mdl, gen_text, device)

            el_vals.append(el)
            fl_vals.append(fl)
            label_dist[label] += 1

            if (i + 1) % 10 == 0:
                print(f"  [{i+1:3d}/{n}] EditLens={el:.4f}  R_fluency={fl:.4f}", flush=True)

    mean_el = sum(el_vals) / max(len(el_vals), 1)
    mean_fl = sum(fl_vals) / max(len(fl_vals), 1)

    kl_div = compute_kl_from_ref(
        policy_mdl, fl_mdl, policy_tok, eval_essays, device
    )

    n_scored = max(sum(label_dist), 1)
    label_pct = [100.0 * c / n_scored for c in label_dist]

    print(
        f"[eval] step={step}  mean_editlens={mean_el:.4f}  "
        f"mean_fluency={mean_fl:.4f}  kl_div={kl_div:.4f}",
        flush=True,
    )
    print(
        f"[eval] label_dist: "
        f"LABEL_0={label_pct[0]:.1f}%  LABEL_1={label_pct[1]:.1f}%  "
        f"LABEL_2={label_pct[2]:.1f}%  LABEL_3={label_pct[3]:.1f}%\n",
        flush=True,
    )

    policy_mdl.train()

    return {
        "step":          step,
        "mean_editlens": mean_el,
        "mean_fluency":  mean_fl,
        "kl_div":        kl_div,
        "label_dist": {
            "LABEL_0": label_pct[0],
            "LABEL_1": label_pct[1],
            "LABEL_2": label_pct[2],
            "LABEL_3": label_pct[3],
        },
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def save_checkpoint(step: int, policy_mdl, optimizer, checkpoint_dir: str) -> None:
    path = os.path.join(checkpoint_dir, f"step_{step:03d}")
    os.makedirs(path, exist_ok=True)
    policy_mdl.save_pretrained(path)
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
    with open(os.path.join(path, "step.json"), "w") as f:
        json.dump({"step": step}, f)
    print(f"[ckpt] Saved checkpoint → {path}", flush=True)


def find_latest_checkpoint(checkpoint_dir: str) -> tuple[str | None, int]:
    """Return (path, step) of the highest-numbered step_NNN directory, or (None, 0)."""
    if not os.path.isdir(checkpoint_dir):
        return None, 0
    best_step, best_path = 0, None
    for name in os.listdir(checkpoint_dir):
        if not name.startswith("step_"):
            continue
        try:
            step = int(name[5:])
        except ValueError:
            continue
        if step > best_step:
            best_step = step
            best_path = os.path.join(checkpoint_dir, name)
    return best_path, best_step


# ── Log helper ────────────────────────────────────────────────────────────────
def append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── One REINFORCE step (unchanged from C1) ───────────────────────────────────
def reinforce_step(
    step_idx: int,
    essays: list[str],
    policy_tok,
    policy_mdl,
    el_tok, el_mdl,
    fl_tok, fl_mdl,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[float, float, int]:
    """
    Execute one REINFORCE step over the essay prompts.
    Returns (mean_reward, grad_norm, n_rejected).
    Logic identical to C1 except prompt uses Instruct chat template.
    """
    # ── Phase 1: generate rollouts (no gradient) ─────────────────────────
    policy_mdl.eval()

    accepted_inputs: list[tuple[torch.Tensor, torch.Tensor, int, int]] = []
    el_scores: list[float] = []
    fl_raw:    list[float] = []
    kl_scores: list[float] = []
    n_rejected = 0

    with torch.no_grad():
        # Pre-compute per-essay token lengths for the min-length ratio check
        essay_token_lens = [
            len(policy_tok(essay, add_special_tokens=False)["input_ids"])
            for essay in essays
        ]

        # Tokenize all prompts as a left-padded batch and generate in one call
        prompt_texts = [build_prompt(essay, policy_tok) for essay in essays]
        policy_tok.padding_side = "left"
        enc = policy_tok(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        prompt_len = enc["input_ids"].shape[1]  # padded length, same for every row

        out = policy_mdl.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            pad_token_id=policy_tok.eos_token_id,
        )
        # out: (B, prompt_len + max_new_tokens); generated tokens start at prompt_len

        eos_id = policy_tok.eos_token_id
        for i, (essay, essay_token_len) in enumerate(zip(essays, essay_token_lens)):
            # Slice this row's generated tokens and trim trailing EOS/pad
            gen_ids = out[i, prompt_len:]
            eos_pos = (gen_ids == eos_id).nonzero(as_tuple=False)
            if len(eos_pos) > 0:
                gen_ids = gen_ids[: eos_pos[0, 0].item()]
            n_gen = gen_ids.shape[0]

            if not passes_min_length(n_gen, essay_token_len):
                n_rejected += 1
                continue

            gen_text = policy_tok.decode(gen_ids, skip_special_tokens=True).strip()

            el = editlens_score(el_tok, el_mdl, gen_text, device)
            fl = fluency_reward(fl_tok, fl_mdl, gen_text, device)

            # Per-sequence KL(π_θ || π_ref) on generated text
            enc_kl = policy_tok(
                gen_text, return_tensors="pt", truncation=True, max_length=256
            ).to(device)
            if enc_kl["input_ids"].shape[1] >= 2:
                policy_logits_kl = policy_mdl(**enc_kl).logits[0, :-1]
                ref_logits_kl    = fl_mdl(**enc_kl).logits[0, :-1]
                v = min(policy_logits_kl.shape[-1], ref_logits_kl.shape[-1])
                policy_lp_kl = F.log_softmax(policy_logits_kl[..., :v], dim=-1)
                ref_lp_kl    = F.log_softmax(ref_logits_kl[..., :v], dim=-1)
                kl_val = (policy_lp_kl.exp() * (policy_lp_kl - ref_lp_kl)).sum(dim=-1).mean().item()
            else:
                kl_val = 0.0

            el_scores.append(el)
            fl_raw.append(fl)
            kl_scores.append(kl_val)
            accepted_inputs.append(
                (enc["input_ids"][i].unsqueeze(0).cpu(), gen_ids.cpu(), prompt_len, essay_token_len)
            )

    if not accepted_inputs:
        return 0.0, 0.0, n_rejected

    n_accepted = len(accepted_inputs)

    # ── Phase 2: normalize fluency and compute combined rewards ───────────
    fl_t    = torch.tensor(fl_raw, dtype=torch.float32)
    fl_mean = fl_t.mean().item()
    fl_std  = fl_t.std().item() if n_accepted > 1 else 1.0
    fl_norm = [(f - fl_mean) / (fl_std + 1e-8) for f in fl_raw]

    rewards = [
        combined_reward(
            editlens_val=el,
            fluency_normalized=fn,
            output_length=gi.shape[0],
            input_length=il,
        ) - KL_PENALTY * kl_val
        for el, fn, kl_val, (_, gi, _, il) in zip(el_scores, fl_norm, kl_scores, accepted_inputs)
    ]
    mean_reward = sum(rewards) / len(rewards)

    # Batch advantage normalization: zero-mean, unit-variance across the batch.
    # This works from step 1 — no warmup required.
    rewards_t  = torch.tensor(rewards, dtype=torch.float32)
    adv_mean   = rewards_t.mean().item()
    adv_std    = rewards_t.std().item() if n_accepted > 1 else 1.0
    advantages = [(r - adv_mean) / (adv_std + 1e-8) for r in rewards]

    # ── Phase 3: recompute log-probs WITH grad (REINFORCE policy gradient) ─
    # Process one rollout at a time and call .backward() immediately to avoid
    # accumulating all activation tensors in VRAM simultaneously.
    # Dividing each loss by n_accepted before .backward() is mathematically
    # equivalent to computing the mean and calling .backward() once.
    policy_mdl.train()
    optimizer.zero_grad()

    for (prompt_ids, gen_ids, p_len, _), advantage in zip(accepted_inputs, advantages):
        prompt_ids = prompt_ids.to(device)
        gen_ids    = gen_ids.to(device)

        full_ids      = torch.cat([prompt_ids, gen_ids.unsqueeze(0)], dim=1)
        logits        = policy_mdl(full_ids).logits
        log_probs_all = F.log_softmax(logits, dim=-1)

        G = gen_ids.shape[0]
        gen_log_probs = log_probs_all[0, p_len - 1 : p_len - 1 + G, :]
        per_tok_lp    = gen_log_probs.gather(1, gen_ids.unsqueeze(1)).squeeze(1)
        seq_lp        = per_tok_lp.mean()

        loss = -seq_lp * advantage / n_accepted
        loss.backward()

        del full_ids, logits, log_probs_all, gen_log_probs, per_tok_lp, loss
        torch.cuda.empty_cache()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in policy_mdl.parameters() if p.requires_grad],
        GRAD_CLIP_NORM,
    ).item()

    optimizer.step()

    return mean_reward, grad_norm, n_rejected


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="C2: 500-step REINFORCE training run")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in outputs/c2_checkpoints/",
    )
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=N_ROLLOUTS,
        help=f"Rollouts per training step (default: {N_ROLLOUTS})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help=f"Max new tokens per generation (default: {MAX_NEW_TOKENS})",
    )
    args = parser.parse_args()

    n_rollouts     = args.n_rollouts
    max_new_tokens = args.max_new_tokens

    os.makedirs(os.path.join(_REPO_ROOT, "outputs"), exist_ok=True)
    torch.manual_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("=" * 64, flush=True)
    print("  C2 REINFORCE — 500-step Training Run", flush=True)
    print("  STAT 4830 PromptLab Deslopifier", flush=True)
    print("=" * 64 + "\n", flush=True)

    device = get_device()

    # Load data
    essays_pool = load_train_prompts(n=max(1000, n_rollouts))
    eval_essays = load_eval_essays()

    # Determine resume state
    start_step = 0
    ckpt_path  = None
    if args.resume:
        ckpt_path, start_step = find_latest_checkpoint(CHECKPOINT_DIR)
        if ckpt_path is None:
            print("[resume] No checkpoint found; starting from step 0.\n", flush=True)
        else:
            print(f"[resume] Resuming from step {start_step}: {ckpt_path}\n", flush=True)

    # Load models
    el_tok, el_mdl              = load_editlens(device)
    fl_tok, fl_mdl              = load_fluency_ref(device)
    policy_tok, policy_mdl      = load_policy(device, checkpoint_path=ckpt_path)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in policy_mdl.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    if ckpt_path is not None:
        opt_path = os.path.join(ckpt_path, "optimizer.pt")
        if os.path.isfile(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            print(f"[resume] Loaded optimizer state from {opt_path}\n", flush=True)

    # Header
    print(
        f"Config: {N_STEPS} steps × {n_rollouts} rollouts  |  "
        f"min_tokens={MIN_OUTPUT_TOKENS}  min_ratio={MIN_OUTPUT_RATIO}  "
        f"lr={LR}  clip={GRAD_CLIP_NORM}  kl_penalty={KL_PENALTY}\n",
        flush=True,
    )
    if start_step > 0:
        print(f"Resuming from step {start_step}\n", flush=True)
    print(f"{'step':>5}  {'mean_reward':>11}  {'grad_norm':>9}  {'rejected':>24}", flush=True)
    print("-" * 56, flush=True)

    for step in range(start_step, N_STEPS):
        mean_r, gnorm, n_rej = reinforce_step(
            step_idx=step,
            essays=random.sample(essays_pool, n_rollouts),
            policy_tok=policy_tok,
            policy_mdl=policy_mdl,
            el_tok=el_tok,
            el_mdl=el_mdl,
            fl_tok=fl_tok,
            fl_mdl=fl_mdl,
            optimizer=optimizer,
            device=device,
            max_new_tokens=max_new_tokens,
        )

        append_jsonl(LOG_PATH, {
            "step":        step + 1,
            "mean_reward": mean_r,
            "grad_norm":   gnorm,
            "n_rejected":  n_rej,
        })

        rej_str = (
            f"{n_rej}/{n_rollouts} YES (filter fired)"
            if n_rej > 0
            else f"0/{n_rollouts} none"
        )
        print(f"{step+1:>5}  {mean_r:>11.4f}  {gnorm:>9.4f}  {rej_str}", flush=True)

        # Eval every EVAL_INTERVAL steps (1-indexed: step 50, 100, …, 500)
        if (step + 1) % EVAL_INTERVAL == 0:
            eval_record = run_eval(
                step=step + 1,
                eval_essays=eval_essays,
                policy_tok=policy_tok,
                policy_mdl=policy_mdl,
                el_tok=el_tok,
                el_mdl=el_mdl,
                fl_tok=fl_tok,
                fl_mdl=fl_mdl,
                device=device,
            )
            append_jsonl(EVAL_LOG_PATH, eval_record)

        # Save checkpoint every CHECKPOINT_INTERVAL steps (step 100, 200, …, 500)
        if (step + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(step + 1, policy_mdl, optimizer, CHECKPOINT_DIR)

    print("\n" + "=" * 64, flush=True)
    print("  C2 REINFORCE TRAINING COMPLETE", flush=True)
    print(f"  Training log → {LOG_PATH}", flush=True)
    print(f"  Eval log     → {EVAL_LOG_PATH}", flush=True)
    print(f"  Checkpoints  → {CHECKPOINT_DIR}", flush=True)
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
