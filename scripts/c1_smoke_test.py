"""
slop_scripts/c1_smoke_test.py  —  Experiment C1: Smoke Test

Validates end-to-end RL infrastructure before committing to C2 (500-step run).

  Policy      : Qwen/Qwen2.5-0.5B + LoRA (r=16, α=32, target q_proj/k_proj/v_proj/o_proj)
  Reward model: pangram/editlens_roberta-large  — editlens_score() from probe_editlens.py
  Fluency ref : meta-llama/Llama-3.2-3B base   — fluency_reward()  from fluency_reward_validation.py
  Algorithm   : REINFORCE, 10 steps, 5 rollouts/step

Per-step output:
    step | mean_reward | grad_norm | n_rejected

At step 10: confirm LoRA parameters changed from step-0 snapshot.

Run:
    python slop_scripts/c1_smoke_test.py

C1 success checklist (from spec §5, Experiment C1):
  [x] Policy generates a non-empty output (> 50 tokens)
  [x] EditLens returns a finite float in [0, 1]
  [x] Fluency log-probs are finite
  [x] Combined reward is finite and non-zero
  [x] REINFORCE gradient norm in [1e-4, 10]
  [x] LoRA parameters differ between step 0 and step 10
  [x] Min-length filter rejects correctly (logged per step)
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ── Make scripts/ importable ──────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import canonical scoring functions from our validated experiment scripts.
# editlens_score : runs pangram/editlens_roberta-large; returns float in [0,1]
# fluency_reward  : runs frozen Llama/Qwen log-prob; returns negative float (higher=more fluent)
# load_editlens   : loads the reward model to the given device
from fluency_reward_validation import editlens_score, fluency_reward, load_editlens

# ── Hyperparameters (mirroring spec C2 config) ────────────────────────────────
POLICY_MODEL_ID  = "Qwen/Qwen2.5-0.5B"
FLUENCY_MODEL_ID = "meta-llama/Llama-3.2-3B"

LORA_R       = 16
LORA_ALPHA   = 32
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_DROPOUT = 0.05

N_STEPS           = 10
N_PROMPTS         = 5       # ai_generated essays from pangram/editlens_iclr per step
MAX_NEW_TOKENS    = 200
MIN_OUTPUT_TOKENS = 50      # reject if output is shorter than this many tokens
MIN_OUTPUT_RATIO  = 0.40    # reject if output < 40% of input essay token count

LR             = 1e-5
WEIGHT_DECAY   = 0.01
GRAD_CLIP_NORM = 1.0        # clip before step (see spec C1 troubleshooting)

# Reward weights — locked in spec §4 A3 (β=0.5 confirmed by A2 Pearson r=0.303)
ALPHA = 1.0   # EditLens detection-evasion
BETA  = 0.5   # fluency (log-prob under frozen Llama)
GAMMA = 0.1   # length-match penalty

SEED = 42

# Instruction prompt for the deslopifier task.
# Qwen 0.5B base has no chat template, so we use a simple completion format.
PROMPT_TEMPLATE = (
    "Rewrite the following AI-generated essay to sound more natural and human. "
    "Preserve the same ideas but change the writing style so that it reads as "
    "if a person wrote it.\n\n"
    "Original:\n{essay}\n\n"
    "Rewritten:\n"
)


# ── Device ───────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[device] CUDA: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[device] CPU  (no GPU found — smoke test will be slow)")
    return dev


# ── Data ─────────────────────────────────────────────────────────────────────
def load_prompts(n: int = N_PROMPTS, min_words: int = 80, max_words: int = 250) -> list[str]:
    """Stream N ai_generated essays from pangram/editlens_iclr (train split)."""
    print(f"[data]  Loading {n} ai_generated essays from pangram/editlens_iclr ...")
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
    print(f"[data]  Collected {len(essays)} prompts.\n")
    return essays


# ── Policy — Qwen 0.5B + LoRA ────────────────────────────────────────────────
def load_policy(device: torch.device):
    """Load Qwen/Qwen2.5-0.5B and attach LoRA adapters (r=16, α=32, q/k/v/o)."""
    print(f"[policy] Loading {POLICY_MODEL_ID} ...")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(POLICY_MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        POLICY_MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

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
        f"[policy] Trainable: {n_train:,} / {n_total:,} params ({100*n_train/n_total:.3f}%)\n"
    )
    return tok, model


# ── Fluency reference — frozen Llama-3.2-3B base ─────────────────────────────
def load_fluency_ref(device: torch.device):
    """
    Load meta-llama/Llama-3.2-3B base as a frozen fluency reference.
    Requires the Meta Llama license accepted at huggingface.co/meta-llama/Llama-3.2-3B.
    """
    print(f"[fluency] Loading {FLUENCY_MODEL_ID} (frozen) ...")
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
    print(f"[fluency] Loaded {FLUENCY_MODEL_ID} ({n/1e9:.3f}B params, dtype={dtype}, frozen)\n")
    return tok, mdl


# ── Combined reward (locked in spec §4 A3) ───────────────────────────────────
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
    R = alpha * (1 - editlens_val)   ← detection evasion: 1 when human, 0 when AI
      + beta  * fluency_normalized    ← fluency z-score (normalized across batch)
      + gamma * length_reward         ← soft length-match penalty

    All weights locked per spec §4 A3 (β=0.5 confirmed by A2, Pearson r=+0.303).
    """
    r_detection = 1.0 - editlens_val

    r_fluency = fluency_normalized   # caller z-scores across batch before calling

    ratio = output_length / max(input_length, 1)
    if 0.40 <= ratio <= 1.60:
        r_length = 1.0
    else:
        r_length = max(0.0, 1.0 - abs(ratio - 1.0))

    return alpha * r_detection + beta * r_fluency + gamma * r_length


# ── Minimum-length filter (spec §4 A3) ───────────────────────────────────────
def passes_min_length(n_gen_tokens: int, essay_token_len: int) -> bool:
    """
    Reject rollouts that are too short.
    Criteria (spec §4 A3):
      - fewer than MIN_OUTPUT_TOKENS absolute tokens, OR
      - fewer than MIN_OUTPUT_RATIO × essay_token_len tokens
    Returns True if the rollout should be kept.
    """
    if n_gen_tokens < MIN_OUTPUT_TOKENS:
        return False
    if n_gen_tokens < MIN_OUTPUT_RATIO * essay_token_len:
        return False
    return True


# ── LoRA parameter snapshot helpers ──────────────────────────────────────────
def snapshot_lora(model) -> dict[str, torch.Tensor]:
    """Clone all LoRA weight tensors for later comparison."""
    return {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if "lora_" in name and param.requires_grad
    }


def lora_changed(initial: dict[str, torch.Tensor], model) -> tuple[bool, float]:
    """
    Check whether any LoRA tensor differs from the initial snapshot.
    Returns (changed: bool, max_abs_delta: float).
    """
    max_delta = 0.0
    changed = False
    for name, param in model.named_parameters():
        if name not in initial:
            continue
        delta = (param.data - initial[name]).abs().max().item()
        if delta > 0.0:
            changed = True
        max_delta = max(max_delta, delta)
    return changed, max_delta


# ── One REINFORCE step ────────────────────────────────────────────────────────
def reinforce_step(
    step_idx: int,
    essays: list[str],
    policy_tok,
    policy_mdl,
    el_tok, el_mdl,
    fl_tok, fl_mdl,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, int]:
    """
    Execute one REINFORCE step over the essay prompts.

    Algorithm:
      1. Generate rollouts from current policy (no_grad — discrete sampling).
      2. Apply minimum-length filter; log rejections.
      3. Compute EditLens and fluency scores for accepted rollouts.
      4. Z-score fluency across the accepted batch.
      5. Compute combined_reward for each rollout.
      6. Re-run forward pass WITH grad to get per-sequence log-probs.
      7. REINFORCE loss: L = -mean(log π(o|x) * R)
      8. Backward + gradient clip + optimizer step.

    Returns:
      mean_reward : float   — mean combined reward over accepted rollouts
      grad_norm   : float   — gradient norm (after clipping)
      n_rejected  : int     — number of rollouts rejected by min-length filter
    """
    # ── Phase 1: generate rollouts (no gradient) ──────────────────────────
    policy_mdl.eval()

    accepted_inputs:  list[tuple[torch.Tensor, torch.Tensor, int, int]] = []
    # Each entry: (prompt_ids, gen_ids, prompt_len, essay_token_len)

    el_scores:    list[float] = []
    fl_raw:       list[float] = []
    n_rejected = 0

    with torch.no_grad():
        for essay in essays:
            # Tokenize essay alone (for length ratio)
            essay_ids = policy_tok(essay, add_special_tokens=False)["input_ids"]
            essay_token_len = len(essay_ids)

            # Build full prompt
            prompt_text = PROMPT_TEMPLATE.format(essay=essay)
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
                do_sample=True,
                temperature=0.8,
                pad_token_id=policy_tok.eos_token_id,
            )
            gen_ids = out[0, prompt_len:]  # generated-only token IDs, shape (G,)
            n_gen = gen_ids.shape[0]

            # Apply minimum-length filter
            if not passes_min_length(n_gen, essay_token_len):
                n_rejected += 1
                continue

            # Decode and score the generated text
            gen_text = policy_tok.decode(gen_ids, skip_special_tokens=True).strip()

            el = editlens_score(el_tok, el_mdl, gen_text, device)
            fl = fluency_reward(fl_tok, fl_mdl, gen_text, device)

            el_scores.append(el)
            fl_raw.append(fl)
            accepted_inputs.append(
                (enc["input_ids"].cpu(), gen_ids.cpu(), prompt_len, essay_token_len)
            )

    # If every rollout was rejected, skip this step
    if not accepted_inputs:
        return 0.0, 0.0, n_rejected

    n_accepted = len(accepted_inputs)

    # ── Phase 2: normalize fluency and compute combined rewards ───────────
    fl_t = torch.tensor(fl_raw, dtype=torch.float32)
    fl_mean = fl_t.mean().item()
    fl_std  = fl_t.std().item() if n_accepted > 1 else 1.0
    fl_norm = [(f - fl_mean) / (fl_std + 1e-8) for f in fl_raw]

    rewards = [
        combined_reward(
            editlens_val=el,
            fluency_normalized=fn,
            output_length=gi.shape[0],
            input_length=il,
        )
        for el, fn, (_, gi, _, il) in zip(el_scores, fl_norm, accepted_inputs)
    ]
    mean_reward = sum(rewards) / len(rewards)

    # ── Phase 3: recompute log-probs WITH grad (REINFORCE policy gradient) ─
    policy_mdl.train()
    optimizer.zero_grad()

    losses: list[torch.Tensor] = []

    for (prompt_ids, gen_ids, p_len, _), reward in zip(accepted_inputs, rewards):
        prompt_ids = prompt_ids.to(device)
        gen_ids    = gen_ids.to(device)

        full_ids = torch.cat([prompt_ids, gen_ids.unsqueeze(0)], dim=1)  # (1, P+G)
        logits   = policy_mdl(full_ids).logits                            # (1, P+G, V)
        log_probs_all = F.log_softmax(logits, dim=-1)                     # (1, P+G, V)

        G = gen_ids.shape[0]
        # Positions [P-1 .. P+G-2] in logits predict tokens [P .. P+G-1] in full_ids,
        # which correspond exactly to gen_ids[0 .. G-1].
        gen_log_probs = log_probs_all[0, p_len - 1 : p_len - 1 + G, :]   # (G, V)
        per_tok_lp    = gen_log_probs.gather(
            1, gen_ids.unsqueeze(1)
        ).squeeze(1)                                                       # (G,)

        # Normalize by sequence length to remove length bias, then scale by reward.
        seq_lp = per_tok_lp.mean()   # mean log-prob per token
        losses.append(-seq_lp * reward)

    loss = torch.stack(losses).mean()
    loss.backward()

    # Gradient norm (clip to GRAD_CLIP_NORM per spec C1 troubleshooting)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in policy_mdl.parameters() if p.requires_grad],
        GRAD_CLIP_NORM,
    ).item()

    optimizer.step()

    return mean_reward, grad_norm, n_rejected


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    torch.manual_seed(SEED)

    print("=" * 64)
    print("  C1 Smoke Test — REINFORCE Infrastructure Validation")
    print("  STAT 4830 PromptLab Deslopifier")
    print("=" * 64 + "\n")

    device = get_device()

    # 1. Load training prompts
    essays = load_prompts(N_PROMPTS)

    # 2. Load models
    el_tok, el_mdl = load_editlens(device)               # reward model (frozen)
    fl_tok, fl_mdl = load_fluency_ref(device)            # fluency ref (frozen)
    policy_tok, policy_mdl = load_policy(device)         # Qwen 0.5B + LoRA

    # 3. Optimizer — AdamW over LoRA parameters only
    optimizer = torch.optim.AdamW(
        [p for p in policy_mdl.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # 4. Snapshot LoRA at step 0 (for final comparison)
    initial_lora = snapshot_lora(policy_mdl)
    print(f"[init]  Snapshotted {len(initial_lora)} LoRA parameter tensors at step 0.\n")

    # 5. REINFORCE training loop
    print(f"Config: {N_STEPS} steps × {N_PROMPTS} rollouts  |  "
          f"min_tokens={MIN_OUTPUT_TOKENS}  min_ratio={MIN_OUTPUT_RATIO}  "
          f"lr={LR}  clip={GRAD_CLIP_NORM}\n")
    print(f"{'step':>5}  {'mean_reward':>11}  {'grad_norm':>9}  {'rejected':>20}")
    print("-" * 52)

    for step in range(N_STEPS):
        mean_r, gnorm, n_rej = reinforce_step(
            step_idx=step,
            essays=essays,
            policy_tok=policy_tok,
            policy_mdl=policy_mdl,
            el_tok=el_tok,
            el_mdl=el_mdl,
            fl_tok=fl_tok,
            fl_mdl=fl_mdl,
            optimizer=optimizer,
            device=device,
        )

        rej_str = (
            f"{n_rej}/{N_PROMPTS} YES (filter fired)"
            if n_rej > 0
            else f"0/{N_PROMPTS} none"
        )
        print(f"{step+1:>5}  {mean_r:>11.4f}  {gnorm:>9.4f}  {rej_str}")

    # 6. C1 final check: confirm LoRA parameters changed
    print("\n" + "=" * 64)
    changed, max_delta = lora_changed(initial_lora, policy_mdl)
    print(f"  LoRA parameters changed from step 0: {changed}")
    if changed:
        print(f"  Max absolute weight delta : {max_delta:.6e}")
    else:
        print("  WARNING: LoRA parameters did NOT change.")
        print("  → Check that LoRA layers have requires_grad=True after get_peft_model()")
        print("  → Check that optimizer was given the correct parameter group")

    # 7. C1 checklist summary
    print("\n  C1 Checklist:")
    print(f"    Policy generates non-empty output       : assumed True (filter enforces > {MIN_OUTPUT_TOKENS} tokens)")
    print(f"    EditLens returns finite float in [0,1]  : enforced by editlens_score()")
    print(f"    Fluency log-probs finite                : enforced by fluency_reward()")
    print(f"    Combined reward finite and non-zero     : check mean_reward column above")
    print(f"    Gradient norm in [1e-4, 10]             : check grad_norm column above")
    print(f"    LoRA params changed step 0 → step 10   : {changed}")
    print(f"    Min-length filter logs rejections       : check 'rejected' column above")
    print("=" * 64)
    print("  C1 SMOKE TEST COMPLETE")
    print("=" * 64)


if __name__ == "__main__":
    main()
