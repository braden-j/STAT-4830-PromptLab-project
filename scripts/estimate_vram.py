"""
estimate_vram.py

Estimates VRAM requirements for the C2 REINFORCE training setup from the
experiment spec (docs/STAT4830_Experiment_Spec_v2.md):

  Policy:  Llama 3.2 3B Instruct  (or Qwen 1.5B as fallback)
           LoRA r=16, alpha=32 on q/k/v/o projections
  Reward:  editlens_Llama-3.2-3B  (frozen, 3B)  -- or use RoBERTa-large (355M)
  Ref:     meta-llama/Llama-3.2-3B base (frozen, for KL / fluency)
  Batch:   16 rollouts × 512 tokens

No model is actually loaded. All estimates derive from architecture configs
and standard GPU memory formulas.

Methodology:
  1. Count parameters from architecture (attention + MLP + embedding layers)
  2. Model weights: n_params × bytes_per_param (bf16 = 2 bytes)
  3. LoRA trainable params: (in_dim + out_dim) × r, for each target projection
  4. Optimizer states (AdamW on LoRA only): 12 bytes/param
       = fp32 param copy (4) + fp32 momentum (4) + fp32 variance (4)
  5. Gradient buffer (fp32): 4 bytes per trainable param
  6. Activation memory (log-prob forward+backward, WITH gradient checkpointing):
       stored residuals at each checkpoint: n_layers × B × T × H × 2 bytes
       + one layer recomputed at a time: per-layer activation estimate
  7. KV cache during rollout generation (no grad):
       B × T × n_kv_heads × head_dim × 2 (K+V) × n_layers × 2 bytes
  8. Reward model weights (frozen, bf16)
  9. Reference model weights (frozen, bf16)
       (can share base weights with policy if LoRA is applied additively)

GPU target: 80 GB H100 SXM5
"""

from __future__ import annotations
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Architecture configs (from published HuggingFace model cards / configs)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str
    n_params_reported: float   # billions, for sanity-check
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attn_heads: int
    num_kv_heads: int
    vocab_size: int
    tied_embeddings: bool      # True = lm_head shares embed weights (no extra params)
    mlp_matrices: int = 3      # 3 = SwiGLU (gate+up+down, LLaMA-style)
                               # 2 = standard FFN (fc1+fc2, BERT/RoBERTa-style)
    classification_head: int = 0  # extra head on top (e.g. 4-class for EditLens)
    head_dim: int | None = None  # if None, computed as hidden_size // num_attn_heads

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attn_heads


# Published architecture specs from HuggingFace model cards
LLAMA_3B = ModelConfig(
    name="meta-llama/Llama-3.2-3B-Instruct",
    n_params_reported=3.21,
    hidden_size=3072,
    intermediate_size=8192,
    num_layers=28,
    num_attn_heads=24,
    num_kv_heads=8,
    vocab_size=128_256,
    tied_embeddings=True,   # lm_head = embed_tokens (tied)
    head_dim=128,
)

QWEN_1_5B = ModelConfig(
    name="Qwen/Qwen2.5-1.5B-Instruct",
    n_params_reported=1.54,
    hidden_size=1536,
    intermediate_size=8960,
    num_layers=28,
    num_attn_heads=12,
    num_kv_heads=2,
    vocab_size=151_936,
    tied_embeddings=True,   # lm_head = embed_tokens (tied)
    head_dim=128,
)

EDITLENS_ROBERTA = ModelConfig(
    name="pangram/editlens_roberta-large",
    n_params_reported=0.355,
    hidden_size=1024,
    intermediate_size=4096,
    num_layers=24,
    num_attn_heads=16,
    num_kv_heads=16,        # standard MHA (not GQA)
    vocab_size=50_265,
    tied_embeddings=False,
    mlp_matrices=2,         # RoBERTa uses fc1+fc2, not SwiGLU
    classification_head=4,  # EditLens 4-class head on top of pooler
    head_dim=64,
)

EDITLENS_LLAMA_3B = ModelConfig(
    name="pangram/editlens_Llama-3.2-3B",
    n_params_reported=3.21,
    hidden_size=3072,
    intermediate_size=8192,
    num_layers=28,
    num_attn_heads=24,
    num_kv_heads=8,
    vocab_size=128_256,
    tied_embeddings=True,
    head_dim=128,
)

# ---------------------------------------------------------------------------
# LoRA config
# ---------------------------------------------------------------------------
LORA_RANK = 16
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]   # per spec


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
BATCH_SIZE      = 16    # rollouts per step
MAX_SEQ_LEN     = 512   # max tokens
BYTES_BF16      = 2
BYTES_FP32      = 4
GB              = 1024 ** 3


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_params(cfg: ModelConfig) -> dict[str, int]:
    """Count parameters by component without loading a model."""
    H  = cfg.hidden_size
    I  = cfg.intermediate_size
    L  = cfg.num_layers
    Nh = cfg.num_attn_heads
    Nk = cfg.num_kv_heads
    V  = cfg.vocab_size
    D  = cfg.head_dim    # D = H / Nh (may differ from H//Nh for some models)

    kv_dim = Nk * D      # total KV projection output dim

    # Per transformer layer
    q_proj   = H * (Nh * D)          # H × H (for standard MHA; Nh*D = H)
    k_proj   = H * kv_dim
    v_proj   = H * kv_dim
    o_proj   = (Nh * D) * H          # same as q_proj
    attn     = q_proj + k_proj + v_proj + o_proj

    # MLP: SwiGLU (gate+up+down, 3 matrices) for LLaMA-family;
    #      standard FFN (fc1+fc2, 2 matrices) for BERT/RoBERTa-family
    if cfg.mlp_matrices == 3:
        mlp = 3 * H * I   # gate_proj + up_proj + down_proj
    else:
        mlp = 2 * H * I   # fc1 (H→I) + fc2 (I→H)

    # Layer norms (RMSNorm): 2 per layer, each H params
    norm_per_layer = 2 * H

    per_layer = attn + mlp + norm_per_layer
    all_layers = L * per_layer

    # Embedding
    embed_tokens = V * H
    final_norm   = H

    # LM head
    lm_head = 0 if cfg.tied_embeddings else V * H

    # Classification head (e.g. 4-class in EditLens, includes pooler dense 1024×1024)
    cls_head = cfg.classification_head * cfg.hidden_size + cfg.hidden_size ** 2

    total = all_layers + embed_tokens + final_norm + lm_head + cls_head

    return {
        "per_layer_attn":  attn,
        "per_layer_mlp":   mlp,
        "per_layer_norm":  norm_per_layer,
        "per_layer_total": per_layer,
        "all_layers":      all_layers,
        "embed_tokens":    embed_tokens,
        "lm_head":         lm_head,
        "final_norm":      final_norm,
        "total":           total,
    }


def count_lora_params(cfg: ModelConfig, rank: int, targets: list[str]) -> dict[str, int]:
    """LoRA adds A(rank×in) + B(out×rank) matrices for each target projection."""
    H  = cfg.hidden_size
    Nh = cfg.num_attn_heads
    Nk = cfg.num_kv_heads
    D  = cfg.head_dim
    kv_dim = Nk * D

    proj_dims = {
        "q_proj": (H, Nh * D),
        "k_proj": (H, kv_dim),
        "v_proj": (H, kv_dim),
        "o_proj": (Nh * D, H),
    }

    per_layer = 0
    per_proj = {}
    for t in targets:
        if t not in proj_dims:
            continue
        in_dim, out_dim = proj_dims[t]
        # LoRA: A is (in_dim × rank), B is (rank × out_dim)
        # Alternatively written as down (in→rank) and up (rank→out)
        params_t = (in_dim + out_dim) * rank
        per_proj[t] = params_t
        per_layer += params_t

    total_lora = cfg.num_layers * per_layer

    return {
        "per_proj": per_proj,
        "per_layer": per_layer,
        "total": total_lora,
    }


# ---------------------------------------------------------------------------
# VRAM estimation
# ---------------------------------------------------------------------------

def estimate_vram_gb(
    policy_cfg: ModelConfig,
    reward_cfg: ModelConfig,
    lora_rank: int,
    lora_targets: list[str],
    batch_size: int,
    seq_len: int,
    separate_reference: bool = True,
) -> dict[str, float]:
    """
    Estimates VRAM in GB for a REINFORCE training step.

    separate_reference: if True, load the reference model as a separate copy.
      If False, assume reference = frozen base weights (already in memory as
      the policy's frozen backbone), costing 0 extra VRAM.
    """
    policy_params = count_params(policy_cfg)
    lora_params   = count_lora_params(policy_cfg, lora_rank, lora_targets)
    reward_params = count_params(reward_cfg)

    B  = batch_size
    T  = seq_len
    H  = policy_cfg.hidden_size
    I  = policy_cfg.intermediate_size
    L  = policy_cfg.num_layers
    Nh = policy_cfg.num_attn_heads
    Nk = policy_cfg.num_kv_heads
    D  = policy_cfg.head_dim
    kv_dim = Nk * D

    # 1. Policy model weights (frozen base + LoRA deltas), bf16
    policy_weights_gb = policy_params["total"] * BYTES_BF16 / GB

    # 2. LoRA trainable weights, bf16
    lora_weights_gb = lora_params["total"] * BYTES_BF16 / GB

    # 3. Optimizer states for LoRA only (AdamW: fp32 param copy + momentum + variance)
    optimizer_gb = lora_params["total"] * (BYTES_FP32 * 3) / GB

    # 4. Gradient buffer for LoRA (fp32)
    grad_buffer_gb = lora_params["total"] * BYTES_FP32 / GB

    # 5. Reference model (frozen, bf16) — separate copy or shared
    if separate_reference:
        ref_model_gb = policy_params["total"] * BYTES_BF16 / GB
    else:
        # Policy base weights ARE the reference; no additional cost
        ref_model_gb = 0.0

    # 6. Reward model (frozen, bf16)
    reward_model_gb = reward_params["total"] * BYTES_BF16 / GB

    # 7. KV cache during rollout GENERATION (no grad, bf16)
    #    Layout: 2 (K and V) × B × L × T × kv_dim × 2 bytes
    kv_cache_gb = 2 * B * L * T * kv_dim * BYTES_BF16 / GB

    # 8. Activation memory during log-prob forward + backward WITH grad checkpointing
    #
    #    With per-layer gradient checkpointing (standard practice for LLM training):
    #    - Store residual hidden states at each layer boundary: L checkpoints × B×T×H
    #    - During backward, recompute one layer at a time; keep ~2 layers in memory
    #
    #    Residual checkpoints (bf16):
    checkpoint_gb = L * B * T * H * BYTES_BF16 / GB
    #
    #    One-layer recompute buffer: attention + MLP activations (simplified, with flash-attn)
    #    Flash attention avoids storing O(T²) attention scores; just store Q,K,V,output ≈ 4×B×T×H
    attn_layer_act = 4 * B * T * H * BYTES_BF16
    #    MLP: gate + up + silu intermediates ≈ 3×B×T×I × 2 bytes
    mlp_layer_act  = 3 * B * T * I * BYTES_BF16
    one_layer_gb   = (attn_layer_act + mlp_layer_act) / GB

    # Total activation: checkpoints + one-layer recompute buffer
    activation_gb = checkpoint_gb + one_layer_gb

    total_gb = (
        policy_weights_gb
        + lora_weights_gb
        + optimizer_gb
        + grad_buffer_gb
        + ref_model_gb
        + reward_model_gb
        + kv_cache_gb
        + activation_gb
    )

    return {
        "policy_weights_gb":   policy_weights_gb,
        "lora_weights_gb":     lora_weights_gb,
        "optimizer_states_gb": optimizer_gb,
        "grad_buffer_gb":      grad_buffer_gb,
        "reference_model_gb":  ref_model_gb,
        "reward_model_gb":     reward_model_gb,
        "kv_cache_rollout_gb": kv_cache_gb,
        "activation_gb":       activation_gb,
        "total_gb":            total_gb,
        # sub-breakdown
        "_activation_checkpoints_gb": checkpoint_gb,
        "_activation_one_layer_gb":   one_layer_gb,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_param_sanity(cfg: ModelConfig) -> None:
    p = count_params(cfg)
    reported_B  = cfg.n_params_reported
    computed_B  = p["total"] / 1e9
    delta_pct   = 100 * (computed_B - reported_B) / reported_B
    print(f"  {cfg.name}")
    print(f"    computed params : {computed_B:.3f}B  "
          f"(reported {reported_B:.3f}B, delta {delta_pct:+.1f}%)")
    print(f"    layers: {cfg.num_layers}, H: {cfg.hidden_size}, "
          f"I: {cfg.intermediate_size}, "
          f"Nh: {cfg.num_attn_heads}, Nkv: {cfg.num_kv_heads}, "
          f"head_dim: {cfg.head_dim}")
    print(f"    tied embeddings: {cfg.tied_embeddings}")
    p_lora = count_lora_params(cfg, LORA_RANK, LORA_TARGETS)
    print(f"    LoRA params (r={LORA_RANK}, {LORA_TARGETS}): "
          f"{p_lora['total']:,}  ({p_lora['total']/p['total']*100:.3f}% of total)")
    print()


def print_vram_table(
    label: str,
    policy_cfg: ModelConfig,
    reward_cfg: ModelConfig,
    est: dict[str, float],
    h100_gb: float = 80.0,
) -> None:
    fits = "YES" if est["total_gb"] < h100_gb else "NO (OOM)"
    headroom_gb = h100_gb - est["total_gb"]

    print(f"{'='*72}")
    print(f"  {label}")
    print(f"  Policy: {policy_cfg.name}")
    print(f"  Reward: {reward_cfg.name}")
    print(f"  LoRA: r={LORA_RANK}, alpha=32, targets={LORA_TARGETS}")
    print(f"  Batch: {BATCH_SIZE} rollouts × {MAX_SEQ_LEN} tokens")
    print(f"{'='*72}")
    rows = [
        ("Model weights (policy, bf16)",      est["policy_weights_gb"]),
        ("  of which: LoRA delta (bf16)",      est["lora_weights_gb"]),
        ("Optimizer states (AdamW, fp32)",     est["optimizer_states_gb"]),
        ("Gradient buffer (fp32)",             est["grad_buffer_gb"]),
        ("Reference model (frozen, bf16)",     est["reference_model_gb"]),
        ("Reward model (frozen, bf16)",        est["reward_model_gb"]),
        ("KV cache — rollout gen (bf16)",      est["kv_cache_rollout_gb"]),
        ("Activations + grad (w/ ckpt, bf16)", est["activation_gb"]),
        ("  of which: layer checkpoints",      est["_activation_checkpoints_gb"]),
        ("  of which: 1-layer recompute buf",  est["_activation_one_layer_gb"]),
    ]
    col_w = 44
    print(f"  {'Component':<{col_w}} {'GB':>8}")
    print(f"  {'-'*col_w} {'-'*8}")
    for name, val in rows:
        indent = "  " if name.startswith("  ") else ""
        print(f"  {name:<{col_w}} {val:>8.2f}")
    print(f"  {'-'*col_w} {'-'*8}")
    print(f"  {'TOTAL':<{col_w}} {est['total_gb']:>8.2f}")
    print(f"  {'H100 capacity':<{col_w}} {h100_gb:>8.2f}")
    print(f"  {'Fits on 80 GB H100?':<{col_w}} {'':>4}{fits}")
    if fits == "YES":
        print(f"  {'Headroom':<{col_w}} {headroom_gb:>8.2f} GB")
    print()


def print_scenario_note(label: str, note: str) -> None:
    print(f"  Note ({label}): {note}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    H100_GB = 80.0

    print("\n" + "=" * 72)
    print("  PARAMETER SANITY CHECK  (computed vs reported)")
    print("=" * 72 + "\n")
    for cfg in [LLAMA_3B, QWEN_1_5B, EDITLENS_ROBERTA, EDITLENS_LLAMA_3B]:
        print_param_sanity(cfg)

    print("=" * 72)
    print("  VRAM ESTIMATES  (training scenario from C2 spec)")
    print("=" * 72)
    print(f"  Batch={BATCH_SIZE} rollouts × {MAX_SEQ_LEN} tokens | "
          f"LoRA r={LORA_RANK} | AdamW | Grad checkpointing ON\n")

    # -----------------------------------------------------------------------
    # Scenario 1: Llama 3.2 3B policy + editlens_Llama-3.2-3B reward
    #             + separate reference model (worst case)
    # -----------------------------------------------------------------------
    est1 = estimate_vram_gb(
        policy_cfg=LLAMA_3B,
        reward_cfg=EDITLENS_LLAMA_3B,
        lora_rank=LORA_RANK,
        lora_targets=LORA_TARGETS,
        batch_size=BATCH_SIZE,
        seq_len=MAX_SEQ_LEN,
        separate_reference=True,
    )
    print_vram_table(
        "Scenario 1 — Llama 3.2 3B  +  EditLens-3B reward  +  separate ref model",
        LLAMA_3B, EDITLENS_LLAMA_3B, est1, H100_GB,
    )
    print_scenario_note("Sc1",
        "Policy + EditLens reward + reference = 3 × Llama 3.2 3B in memory. "
        "Worst case but still fits with headroom.")

    # -----------------------------------------------------------------------
    # Scenario 2: Llama 3.2 3B policy + editlens_Llama-3.2-3B reward
    #             + SHARED reference (policy base IS the reference)
    # -----------------------------------------------------------------------
    est2 = estimate_vram_gb(
        policy_cfg=LLAMA_3B,
        reward_cfg=EDITLENS_LLAMA_3B,
        lora_rank=LORA_RANK,
        lora_targets=LORA_TARGETS,
        batch_size=BATCH_SIZE,
        seq_len=MAX_SEQ_LEN,
        separate_reference=False,
    )
    print_vram_table(
        "Scenario 2 — Llama 3.2 3B  +  EditLens-3B reward  +  SHARED ref (optimal)",
        LLAMA_3B, EDITLENS_LLAMA_3B, est2, H100_GB,
    )
    print_scenario_note("Sc2",
        "With LoRA, the frozen base weights ARE the reference; no separate copy needed. "
        "Standard in LoRA-based RL (trl, OpenRLHF, etc.).")

    # -----------------------------------------------------------------------
    # Scenario 3: Llama 3.2 3B policy + editlens_roberta-large reward
    #             + SHARED reference (recommended fast-ablation setup)
    # -----------------------------------------------------------------------
    est3 = estimate_vram_gb(
        policy_cfg=LLAMA_3B,
        reward_cfg=EDITLENS_ROBERTA,
        lora_rank=LORA_RANK,
        lora_targets=LORA_TARGETS,
        batch_size=BATCH_SIZE,
        seq_len=MAX_SEQ_LEN,
        separate_reference=False,
    )
    print_vram_table(
        "Scenario 3 — Llama 3.2 3B  +  EditLens-RoBERTa reward  +  SHARED ref",
        LLAMA_3B, EDITLENS_ROBERTA, est3, H100_GB,
    )
    print_scenario_note("Sc3",
        "Swap EditLens-3B reward for RoBERTa-large (355M). Saves ~6 GB vs Sc2. "
        "Recommended for fast iteration and ablation runs.")

    # -----------------------------------------------------------------------
    # Scenario 4: Qwen 1.5B policy + editlens_roberta-large reward
    #             + SHARED reference (fallback / smallest viable setup)
    # -----------------------------------------------------------------------
    est4 = estimate_vram_gb(
        policy_cfg=QWEN_1_5B,
        reward_cfg=EDITLENS_ROBERTA,
        lora_rank=LORA_RANK,
        lora_targets=LORA_TARGETS,
        batch_size=BATCH_SIZE,
        seq_len=MAX_SEQ_LEN,
        separate_reference=False,
    )
    print_vram_table(
        "Scenario 4 — Qwen 1.5B  +  EditLens-RoBERTa reward  +  SHARED ref  (fallback)",
        QWEN_1_5B, EDITLENS_ROBERTA, est4, H100_GB,
    )
    print_scenario_note("Sc4",
        "Qwen 1.5B fallback with lightweight reward. Leaves vast headroom for "
        "larger batches (could go to batch=128+) or longer sequences.")

    # -----------------------------------------------------------------------
    # Cross-scenario summary table
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("  CROSS-SCENARIO SUMMARY")
    print("=" * 72)
    scenarios = [
        ("Sc1: Llama 3.2 3B + EL-3B + sep ref",  est1),
        ("Sc2: Llama 3.2 3B + EL-3B + shared ref", est2),
        ("Sc3: Llama 3.2 3B + EL-RoBERTa + shared ref", est3),
        ("Sc4: Qwen 1.5B + EL-RoBERTa + shared ref", est4),
    ]
    header_cols = ["Scenario", "Wts+LoRA", "Optim", "Ref+Rwd", "Activ+KV", "TOTAL", "H100 80GB"]
    fmt = "  {:<42} {:>8} {:>7} {:>8} {:>9} {:>7}  {}"
    print(fmt.format(*header_cols))
    print("  " + "-"*95)
    for name, e in scenarios:
        wts  = e["policy_weights_gb"] + e["lora_weights_gb"]
        opt  = e["optimizer_states_gb"] + e["grad_buffer_gb"]
        rwd  = e["reference_model_gb"] + e["reward_model_gb"]
        act  = e["activation_gb"] + e["kv_cache_rollout_gb"]
        tot  = e["total_gb"]
        fits = "OK  (+{:.0f} GB)".format(80 - tot) if tot < 80 else "OOM"
        print(fmt.format(
            name,
            f"{wts:.1f} GB", f"{opt:.1f} GB", f"{rwd:.1f} GB",
            f"{act:.1f} GB", f"{tot:.1f} GB",
            fits,
        ))
    print()
    print("  Columns: Wts+LoRA = frozen base + LoRA delta (bf16)")
    print("           Optim    = AdamW fp32 (LoRA only) + grad buffer")
    print("           Ref+Rwd  = reference model + reward model (both frozen, bf16)")
    print("           Activ+KV = activation checkpoints + 1-layer buffer + KV cache")
    print()
    print("  Assumptions:")
    print("  - bf16 inference and frozen weights; fp32 optimizer states (standard AdamW)")
    print("  - Flash Attention: avoids storing O(T^2) attention scores")
    print("  - Gradient checkpointing: stores L residuals + recomputes 1 layer at a time")
    print("  - KV cache only during rollout generation (torch.no_grad)")
    print("  - Shared reference = LoRA policy's frozen base IS the reference (no copy)")
    print("  - Does NOT include: CUDA kernel overhead (~0.5-1 GB), PyTorch allocator")
    print("    fragmentation (~5-10% buffer recommended in practice)")
    print()


if __name__ == "__main__":
    main()
