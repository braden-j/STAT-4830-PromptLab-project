# STAT 4830 — PromptLab Deslopifier
# Experiment Specification — FINAL (v3)

**Date:** April 14, 2026
**Final Presentation:** April 21/23, 2026
**Compute:** H100 via Prime Intellect
**Status:** All open questions resolved via probe experiments. Ready to convert to scripts.

**Changelog from v2:**
- EditLens inference formula confirmed and locked (Section 2.4)
- Real score ranges from probe on actual corpus data (Section 2.5)
- H100 memory budget confirmed; batch size upgraded 16 → 48 (Section 2.6)
- All 8 open questions resolved — Section 11 now a confirmed decisions log
- Group A partially complete: A1 done, A3 reward formula locked
- Iteration checklist updated with confirmed checkboxes

---

## 0. Framing & North Star

The goal is to train a **Deslopifier** — a model that rewrites AI-generated text to reduce its detectable AI signature while preserving fluency and coherence. This is framed as a **reinforcement learning problem**:

- **Policy** `π_θ`: Llama 3.2 3B Instruct, fine-tuned via LoRA — rewrites AI essays
- **Reward** `R`: EditLens continuous detection score + log-probability fluency signal
- **Optimization**: REINFORCE (primary) or GRPO (comparison), on H100 via Prime Intellect

The task sits on a **continuous spectrum** — not a binary threshold:

```
Fully AI-generated ─────────────────────────────────► Fully human-written
   EditLens ≈ 0.999     EditLens ≈ 0.347     EditLens ≈ 0.007–0.035
   (raw Gemini slop)    (GPT-4.1 light edit)   (human essays, confirmed)
```

These are **real measured values from probe_editlens.py on actual corpus data** (see Section 2.5). The deslopifier's job is to move essays from the left end toward the right. Even reaching the middle zone (EditLens ≈ 0.347, "lightly AI-edited") is a concrete, measurable result suitable for the final presentation.

**The professor's workflow:** Write Big Spec → Iterate → Convert to scripts → Commit to GitHub → Run on server. The spec is now in final state. Next step: convert to scripts.

---

## 1. Current State Baseline

### 1.1 Classifier — Three Generations (Pre-EditLens)

| Version | Backbone | Slop Generator | Notes |
|---|---|---|---|
| v1 | DistilGPT2 (causal) | Character corruption | Near-100% AUC — inflated by corruption artifacts |
| v1.5 | DistilGPT2 (causal) | Mirror-prompting via Qwen 0.5B | Ablation: isolates slop-quality effect from backbone |
| v2 | DistilBERT (bidirectional) + LoRA | Mirror-prompting via Qwen 0.5B | Current best; AUC ~1.0 but likely too easy |

These are superseded by EditLens as the primary reward. They remain useful as fast secondary scorers and as a record of the project's evolution.

### 1.2 Hill-Climbing Prompt Optimizer

Population-based hill climbing (`evolve.py`) with TinyLlama 1.1B as the generator and our local DistilBERT classifier as reward. Never evaluated against EditLens. The reward signal was too weak and too dataset-specific to produce generalizable results.

### 1.3 What Does Not Yet Exist

- Gradient-based RL training (REINFORCE or GRPO) of any kind
- Integration of EditLens as the live reward signal during training
- Any policy model larger than TinyLlama 1.1B
- End-to-end evaluation of deslopified outputs against any external detector

---

## 2. Open Pangram / EditLens — Full Technical Picture

On March 24, 2026, Pangram released EditLens under CC BY-NC-SA 4.0. All technical details below have been **confirmed empirically** via `probe_editlens.py` and `estimate_vram.py` run in the project environment.

### 2.1 What Was Released

**Models on HuggingFace (`huggingface.co/pangram`):**

| Model | Parameters | Base | Max Tokens | Role in this project |
|---|---|---|---|---|
| `pangram/editlens_Llama-3.2-3B` | 3B | `meta-llama/Llama-3.2-3B` | 1024 | **Primary reward model** |
| `pangram/editlens_roberta-large` | 355M | RoBERTa-Large | 512 | Fast proxy reward for ablations |

Both models require HuggingFace authentication. Access approved; token stored via `huggingface-cli login` (use `python -c "from huggingface_hub import login; login()"` if CLI not on PATH).

**Dataset: `pangram/editlens_iclr`**

| Split | Size | Classes |
|---|---|---|
| Train | 60,000 | human_written / ai_generated / ai_edited |
| Validation | 2,400 | Same |
| Test | 6,000 | Same |
| OOD — Enron emails | ~6,000 | Same |
| OOD — Llama 3.3 70B | ~6,000 | Same |

Domains: news (CNN/DailyMail), creative writing (WritingPrompts), Amazon reviews, Google reviews, education web content (FineWeb). AI text from GPT-4.1, Claude Sonnet 4, and Gemini 2.5 Flash. Also released: 1,800-text Grammarly edits dataset (9 edit types × 200 human source texts).

**Source code:** `github.com/pangramlabs/EditLens`

**Citation (required in final report):**
```
Thai, K., Emi, B., Masrour, E., & Iyyer, M. (2025). EditLens: Quantifying
the Extent of AI Editing in Text. ICLR 2026. arXiv:2510.03154.
```

### 2.2 What EditLens Measures

EditLens is an **ordinal regression model** that predicts the extent of AI editing in a text on a 4-bucket scale. It is not a binary classifier.

| Label | Meaning | Typical score_pred |
|---|---|---|
| LABEL_0 | Fully human-written | 0.00–0.10 |
| LABEL_1 | Lightly AI-edited | 0.10–0.43 |
| LABEL_2 | Heavily AI-edited | 0.43–0.76 |
| LABEL_3 | Fully AI-generated | 0.76–1.00 |

The boundaries above are approximate; the continuous `score_pred` value is what matters for RL, not the discrete bucket.

**Key design implication:** The deslopifier is by definition an AI-editing tool. EditLens was specifically trained to distinguish AI-edited text from both fully human and fully AI text. To fool EditLens, the deslopifier must learn to make edits that resemble those of a human editor — light-touch, targeted, stylistically coherent — rather than wholesale rewrites. This is consistent with the Grammarly dataset finding: small grammar-fix edits score near 0, while large additive rewrites score ~0.5.

### 2.3 License

CC BY-NC-SA 4.0 — non-commercial use only. Appropriate for this course project. Must be cited in the final report. Cannot be used in any commercial product.

### 2.4 Confirmed Inference Pipeline

**Confirmed via `probe_editlens.py`.** The model outputs raw logits (shape `(1, 4)`, dtype `float32`, range approximately `[-3, +6.5]`). The canonical inference formula from the official EditLens repo is:

```python
import torch

def editlens_score(model, tokenizer, text: str, max_length: int = 512) -> float:
    """
    Returns score_pred in [0, 1].
    0.0 = fully human-written
    1.0 = fully AI-generated
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )
    with torch.no_grad():
        logits = model(**inputs).logits          # shape (1, 4), raw logits

    probs = torch.softmax(logits, dim=-1)[0]     # shape (4,), sums to 1.0
    bucket_pred = int(probs.argmax().item())      # 0–3, discrete prediction
    score_pred = sum(
        p * i for i, p in enumerate(probs.tolist())
    ) / 3.0                                       # weighted avg, normalized to [0, 1]

    return score_pred
```

**Critical notes:**
- Do NOT use raw logits as the reward. They are not in [0,1] and their magnitude varies.
- Do NOT apply sigmoid. The correct normalization is the weighted average divided by 3.
- Avoid calling `.numpy()` directly — use `.tolist()` then standard Python arithmetic (NumPy 2.x / PyTorch bridge issue in this environment).
- `max_length=512` is appropriate for `editlens_roberta-large`; use `max_length=1024` for `editlens_Llama-3.2-3B`.

### 2.5 Confirmed Score Ranges on Real Corpus Data

**From `probe_editlens.py` run on `pangram/editlens_iclr` samples — these are the ground-truth reference values for all experiments.**

| Sample | Source | bucket_pred | score_pred |
|---|---|---|---|
| Human essay | Reddit WritingPrompts | 0 — fully human | **0.035** |
| Human essay | News article | 0 — fully human | **0.007** |
| AI-generated | Gemini 2.5 Flash, edu | 3 — fully AI | **0.9996** |
| AI-generated | Gemini 2.5 Flash, creative | 3 — fully AI | **0.9994** |
| AI-edited | GPT-4.1 light edit | 1 — lightly AI-edited | **0.347** |

**The spec's success criteria are confirmed achievable:**
- Human essays: score_pred ≈ 0.007–0.035 ✅ (well below the 0.3 target)
- Raw AI slop: score_pred ≈ 0.999 ✅ (well above the 0.7 baseline)
- The intermediate zone exists: GPT-4.1 light edit at 0.347 ✅

**Calibration note from the probe:** The model is decisive and high-confidence. Human text receives a large positive logit on LABEL_0 (~+4.2). AI-generated text receives a large positive logit on LABEL_3 (~+6.5). The probability mass rarely splits evenly — the model commits strongly to a bucket. This means the continuous score_pred will often sit near 0.0 or 1.0 rather than smoothly distributed, particularly early in training.

**Data hygiene warning confirmed by probe:** Polished, well-formed English prose — even if written by a human — may score near LABEL_3 because EditLens was calibrated on informal writing (Reddit, reviews, news) rather than literary essays. When selecting human essays for RL training inputs, prefer texts from domains matching the EditLens training distribution (news, reviews, creative writing) rather than highly polished academic or literary prose.

### 2.6 Confirmed Memory Budget

**From `estimate_vram.py`. All figures include LoRA, optimizer states, and activations for 48 rollouts × 512 tokens. Add ~2–3 GB for PyTorch allocator fragmentation.**

| Scenario | Model Weights + LoRA | Optimizer | Ref + Reward | Activations | Total | H100 80 GB |
|---|---|---|---|---|---|---|
| **Llama 3.2 3B + EL-3B + shared ref** ← **USE THIS** | 6.0 GB | 0.1 GB | 6.0 GB | 2.8 GB | **14.9 GB** | ✅ +65 GB headroom |
| Llama 3.2 3B + EL-3B + separate ref | 6.0 GB | 0.1 GB | 12.0 GB | 2.8 GB | 20.9 GB | ✅ +59 GB headroom |
| Llama 3.2 3B + EL-RoBERTa + shared ref | 6.0 GB | 0.1 GB | 0.8 GB | 2.8 GB | 9.7 GB | ✅ +70 GB headroom |
| Qwen 1.5B + EL-RoBERTa + shared ref (fallback) | 2.9 GB | 0.1 GB | 0.8 GB | 1.4 GB | 5.1 GB | ✅ +75 GB headroom |

**Confirmed choice: Scenario 2 (Llama 3.2 3B + EditLens-3B + shared reference).**

"Shared reference" means the frozen Llama 3.2 3B base checkpoint serves dual purpose: its forward pass provides log-probability fluency scores, and it is the base from which `editlens_Llama-3.2-3B` was fine-tuned. One copy on GPU, used for both functions.

**Key finding from memory estimation:** The original spec's batch size of 16 was overly conservative. With 65 GB of headroom, batch size should be **48 rollouts per step** — a 3× increase that directly reduces gradient variance in REINFORCE and speeds convergence. The optimizer remains cheap (AdamW on 9.18M LoRA parameters = 0.10 GB total).

---

## 3. Experiment Overview

| Group | Name | Priority | Compute | Status |
|---|---|---|---|---|
| A | Reward Model Setup & Validation | **Critical** | Local / T4 | 🟡 A1 done; A2–A3 remaining |
| B | Classifier Hardening via EditLens Dataset | High | T4 | ⬜ Not started |
| C | RL Training Loop | **Critical** | H100 | ⬜ Not started |
| D | Architecture & Model Size | High | H100 | ⬜ Not started |
| E | Evaluation & Analysis | **Critical** | T4 + local | ⬜ Not started |

---

## 4. Group A — Reward Model Setup & Validation

### ✅ Experiment A1: EditLens Baseline on Real Corpus Data — COMPLETE

**Status: Done.** `probe_editlens.py` confirmed the inference pipeline, label semantics, and score ranges. Results documented in Section 2.4–2.5. The script is the canonical EditLens scoring function for all subsequent experiments.

**Confirmed outputs:**
- Inference formula locked (Section 2.4)
- Score ranges on real data locked (Section 2.5)
- `probe_editlens.py` committed to repo as the canonical scoring utility

**One remaining follow-up:** Run A1 at scale — score the full 300-essay eval set (100 human, 100 AI-generated, 100 AI-edited from the EditLens test split) and produce distribution histograms. This generates the opening visualization for the final presentation. Extend `probe_editlens.py` with a `--batch` flag that takes a dataset split and outputs a histogram. Estimated time: 1 hour.

---

### Experiment A2: Fluency Reward Implementation and Validation

**What:** Implement and validate the log-probability fluency reward using frozen `meta-llama/Llama-3.2-3B` (base model, no EditLens fine-tuning) as the reference:

```python
def fluency_reward(model_frozen, tokenizer, text: str) -> float:
    """
    Mean per-token log-probability under the frozen Llama 3.2 3B base model.
    Higher = more natural/fluent from a language modeling perspective.
    Returns a negative float (log-probs are ≤ 0); normalize across batch.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model_frozen(**inputs, labels=input_ids)
    # outputs.loss is the mean negative log-likelihood per token
    return -outputs.loss.item()   # negate: higher = more fluent
```

**Why this reference model:** `meta-llama/Llama-3.2-3B` base (pre-fine-tuning) has no knowledge of AI detection patterns — it is a general language model. Its log-probabilities measure how "natural" text is from a pure language modeling standpoint, without any detection bias. Since `editlens_Llama-3.2-3B` was fine-tuned from this exact checkpoint, the two models share a vocabulary and tokenizer, preventing any tokenization mismatch in the reward computation.

**Validation task:** Score the same 300-essay corpus used in A1's extended run. Plot: EditLens score (x-axis) vs R_fluency (y-axis). Expected result: human essays cluster in the top-left (low EditLens, high fluency). Raw AI slop clusters in the bottom-right (high EditLens, lower fluency). The GPT-4.1 light edit sits somewhere between.

**What we need to confirm:**
- Are the two signals complementary (moderate positive correlation, r ≈ 0.3–0.6) or redundant (r > 0.9)?
- If redundant, the fluency term adds little and β can be set to 0.1 instead of 0.5
- If anti-correlated, there is a genuine tension between "evades detection" and "sounds natural" — which is the more interesting scientific finding

**Outputs:** Scatter plot with human/AI/edited labels. Pearson and Spearman correlation between R_fluency and EditLens score_pred. This directly informs the starting values of α and β in A3.

**Estimated time:** 1–2 hours local/T4.

---

### Experiment A3: Combined Reward Formula — Design Decision

**Status:** Partially locked. Full formula confirmed. Starting weights set below pending A2 correlation results.

**Locked reward formula:**

```python
def combined_reward(
    editlens_score: float,    # from editlens_score() above; 0=human, 1=AI
    fluency_score: float,     # from fluency_reward() above; normalized per batch
    output_length: int,
    input_length: int,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.1,
) -> float:
    """
    Combined RL reward. Higher = better (more human-like, more fluent, right length).

    R = alpha * (1 - editlens_score)       # detection evasion: 0 when AI, 1 when human
      + beta  * fluency_score_normalized    # fluency: normalized to zero-mean unit-var across batch
      + gamma * length_penalty              # length: penalize outputs < 40% or > 160% of input length
    """
    # Detection evasion: invert EditLens score so higher = more human
    r_detection = 1.0 - editlens_score

    # Fluency: passed in already normalized across the batch
    r_fluency = fluency_score

    # Length: reward 1.0 if within 40–160% of input length, else linear penalty
    length_ratio = output_length / max(input_length, 1)
    if 0.40 <= length_ratio <= 1.60:
        r_length = 1.0
    else:
        r_length = max(0.0, 1.0 - abs(length_ratio - 1.0))

    return alpha * r_detection + beta * r_fluency + gamma * r_length
```

**Starting weights:** α=1.0, β=0.5, γ=0.1. These are ablated in Experiment C5. If A2 finds the two signals are highly correlated (r > 0.8), reduce β to 0.2. If A2 finds them anti-correlated, keep β=0.5 and watch C4's KL ablation carefully.

**Minimum output length filter (separate from reward):** Reject any rollout shorter than 50 tokens OR shorter than 40% of the input length before computing reward. This prevents the degenerate case where the model learns to output very short text that EditLens scores as human simply because it is short. Log the number of rejected rollouts per training step — if > 10% are being rejected, the threshold is too aggressive.

---

## 5. Group B — Classifier Hardening via EditLens Dataset

**Purpose:** Harden the local DistilBERT v2 classifier using the EditLens dataset. Not needed for RL training (EditLens is the reward directly), but produces a fast, accurate proxy reward for cheap ablations where running the 3B reward model is too slow.

---

### Experiment B1: OOD Evaluation of v2 Classifier

**What:** Evaluate how well DistilBERT v2 generalizes to text it was never trained on. Pull samples from the EditLens test split (GPT-4.1, Claude Sonnet 4, Gemini 2.5 Flash generated) and compare:
- v2 DistilBERT AUC on these samples
- `editlens_roberta-large` AUC on the same samples

**Expected result:** v2 will underperform significantly on GPT-4.1/Claude/Gemini text because it was trained only on Qwen 0.5B mirror-prompted slop. This quantifies the gap and motivates B2.

**Outputs:** Side-by-side AUC comparison. Examples of essays that fool v2 but not EditLens-RoBERTa.

---

### Experiment B2: Fine-Tune DistilBERT on EditLens Training Data

**What:** Fine-tune our existing v2 DistilBERT on the EditLens training split. Binarize the three-class labels: `human_written` → 0, `ai_generated` → 1, discard `ai_edited` for the binary task (or use it as a third class for B3).

This is substantially simpler than the hard negative mining loop (Pangram Algorithm 1) that was scaffolded in the v2 notebook — we have 60k labeled examples already, covering 4 domains and 3 major LLM generators. Use them directly.

**Implementation:** Adapt the existing `train_token_classifier.py` script to load from `pangram/editlens_iclr` instead of our local data. The architecture (DistilBERT + LoRA + binary head) is unchanged.

**Outputs:**
- AUC after fine-tuning vs. original v2 (on EditLens test split)
- Inference time: fine-tuned DistilBERT vs `editlens_roberta-large` (in ms/essay)

**Success criteria:** Fine-tuned DistilBERT closes at least 50% of the AUC gap between v2-original and EditLens-RoBERTa. If it does, it becomes the fast proxy reward for C3/C4/C5 ablations.

**Estimated time:** 2–3 hours on T4 Colab. Zero data collection cost.

---

### Experiment B3: Ternary Classifier (Stretch Goal)

**What:** Train a three-class version of our classifier (human / AI-edited / fully AI) on the full EditLens training set including `ai_edited` examples. This mirrors EditLens's ternary classification setup exactly and enables ternary tracking during RL training (E5) without calling the full 3B reward model at every eval step.

**Dependency:** Only worth running if C2 shows essays reaching the AI-edited zone. Defer until after the first RL results are in.

---

## 6. Group C — RL Training Loop

**Purpose:** Train the deslopifier using RL. This is the core of the project.

### Confirmed Setup (applies to all Group C experiments)

**Policy:** `meta-llama/Llama-3.2-3B-Instruct` with LoRA
- LoRA: r=16, α=32, target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- LoRA parameter count: ~9.18M (0.3% of total model parameters)
- Base weights frozen; only LoRA adapters updated

**Reward model:** `pangram/editlens_Llama-3.2-3B` (frozen throughout)

**Fluency reference:** `meta-llama/Llama-3.2-3B` base, no fine-tuning (frozen; shared with reward model base weights)

**Training data:** `pangram/editlens_iclr` train split, filtered to `text_type == "ai_generated"` and `score_pred > 0.7`. This gives a clean set of high-confidence AI-generated inputs for the deslopifier to work on. Size: approximately 20,000 examples after filtering.

**Batch size: 48 rollouts per step** (upgraded from the original spec's 16 — confirmed to fit with 65 GB headroom, Section 2.6)

**Evaluation protocol (every 50 steps):** Score 100 held-out essays (50 from EditLens test `ai_generated`, 50 from `ai_edited`) on:
- Mean EditLens score_pred (primary; should decrease)
- Mean R_fluency (should stay roughly flat)
- Ternary class distribution: % in each of (fully AI / lightly edited / heavily edited / human) buckets
- KL divergence from reference model

**Training prompt:**
```
System: You are a skilled human writing editor. Rewrite the following text to
sound more like it was written by a thoughtful human. Make targeted, precise
edits. Preserve the core meaning and approximate length.
User: [input AI-generated essay]
Assistant: [rewritten essay]
```

**Maximum sequence lengths:** input 512 tokens, output 512 tokens. The EditLens-3B model supports 1024 tokens, so full essays fit for reward scoring without truncation.

---

### Experiment C1: Smoke Test — Infrastructure Validation

**What:** Run 10 RL steps with the smallest viable model (Qwen 0.5B or TinyLlama 1.1B) on 5 training examples. Confirm end-to-end pipeline integrity.

**Checklist — must all pass before C2:**
- [ ] Policy generates a non-empty essay (> 50 tokens)
- [ ] EditLens scores the output and returns a float in [0, 1] — not NaN, not a tensor, not a probability vector
- [ ] Fluency log-probs are finite (not -inf, which happens for empty outputs or padding)
- [ ] Combined reward is finite and non-zero
- [ ] REINFORCE gradient norm is in [1e-4, 10] — not zero, not exploding
- [ ] LoRA parameters change between step 0 and step 10
- [ ] The minimum-length filter fires correctly on artificially short outputs (inject a short output to test)

**What to do if it fails:**
- NaN reward → check EditLens score normalization (Section 2.4) and fluency padding mask
- Zero gradient → confirm LoRA layers have `requires_grad=True` after loading
- Exploding gradient → add gradient clipping (`max_norm=1.0` in AdamW)
- EditLens returns a tensor not a scalar → confirm `.item()` call in `editlens_score()`

**Estimated time:** 15 minutes on H100. Run this before committing to any full training run.

---

### Experiment C2: REINFORCE — Primary Training Run

**What:** The main RL experiment. Llama 3.2 3B Instruct deslopifier, trained with REINFORCE for 500 steps.

**Full hyperparameter config:**

```yaml
# c2_reinforce.yaml
policy:
  model: meta-llama/Llama-3.2-3B-Instruct
  lora_r: 16
  lora_alpha: 32
  lora_targets: [q_proj, k_proj, v_proj, o_proj]
  max_input_length: 512
  max_output_length: 512

reward:
  editlens_model: pangram/editlens_Llama-3.2-3B
  fluency_ref: meta-llama/Llama-3.2-3B        # shared base, frozen
  alpha: 1.0
  beta: 0.5
  gamma: 0.1
  min_output_tokens: 50
  min_output_ratio: 0.40                       # reject if < 40% of input length

training:
  algorithm: REINFORCE
  batch_size: 48                               # confirmed fits in 14.9 GB
  optimizer: AdamW
  lr: 1e-5
  weight_decay: 0.01
  gradient_clip: 1.0
  kl_penalty: 0.1                             # ablated in C4
  num_steps: 500
  eval_every: 50
  seed: 42

data:
  dataset: pangram/editlens_iclr
  split: train
  filter: text_type == "ai_generated" and score_pred > 0.7
  eval_n: 100
```

**Expected training behavior:**
- Steps 0–50: reward should increase slightly as the model learns the rewriting format
- Steps 50–200: EditLens score should begin to decrease; fluency should stay roughly flat
- Steps 200–500: continued EditLens decrease, potentially entering the AI-edited zone (score_pred < 0.7)
- Watch for: KL divergence growing rapidly (increase `kl_penalty`), reward stalling at a plateau (try GRPO in C3), fluency collapsing (increase `beta`)

**Outputs per eval checkpoint:**
1. Mean EditLens score_pred on 100 held-out essays
2. Mean R_fluency score
3. KL divergence from reference
4. Ternary distribution: (% fully AI, % lightly edited, % heavily edited, % human)
5. 5 qualitative examples at step 0 vs step 500: input text and output text side by side

**Success criteria:**
- Primary: EditLens mean score_pred drops from ~0.999 to below 0.700 (at least some essays reaching AI-edited zone)
- Secondary: R_fluency does not drop by more than 1 standard deviation from its step-0 value
- Stretch: Any essays reach score_pred < 0.347 (the GPT-4.1 light-edit benchmark from A1)

**Estimated H100 time:** 4–5 hours.

---

### Experiment C3: GRPO vs REINFORCE

**What:** Identical setup to C2 but using GRPO (Group Relative Policy Optimization). GRPO generates K=8 outputs per input prompt, normalizes rewards within the group (subtract group mean, divide by group std), and uses normalized rewards as advantages. This eliminates the need for a learned value function while reducing gradient variance.

**With 48-rollout batch size:** GRPO uses 6 prompt groups × K=8 = 48 total rollouts per step, matching C2's batch size exactly. The cost difference vs C2 is: 8× more policy forward passes per unique prompt, but same total rollouts and same EditLens/fluency scoring overhead. Net cost is ~20–30% higher per step (generator dominates, not scorer).

**Outputs:** Reward curve overlaid with C2 (same axes). Gradient variance per step (should be lower in GRPO). Steps-to-convergence comparison. Wall-clock time per step.

**Decision rule:** If GRPO reaches C2's best EditLens score in fewer than 80% of C2's steps, use GRPO for C4/C5. Otherwise, REINFORCE is simpler and sufficient.

**Estimated H100 time:** 6–7 hours.

---

### Experiment C4: KL Penalty Ablation — Pareto Frontier

**What:** Using the better algorithm from C2/C3, run three conditions varying only `kl_penalty`:

| Run | kl_penalty | Expected behavior |
|---|---|---|
| C4-a | 0.00 | No constraint — likely reward hacks quickly |
| C4-b | 0.10 | Default from C2 — moderate constraint |
| C4-c | 0.50 | Strong constraint — stays close to initialization |

**Why this is the most important ablation:** The KL penalty is the primary guard against reward hacking and the primary driver of the fluency/evasion trade-off. This experiment produces the **Pareto frontier** — the main result for the final presentation.

```
R_fluency
    ▲
    │  ● C4-c (high KL)        — high fluency, modest EditLens improvement
    │
    │         ● C4-b (medium KL)  — balanced
    │
    │                  ● C4-a (no KL)  — best EditLens, possible fluency collapse
    └──────────────────────────────────► 1 - EditLens score (higher = more human)
```

Showing this curve at the final presentation is far stronger than a single operating point. It demonstrates that the system's behavior is principled and controllable, not a black box.

**Run all three in parallel if multiple H100s are available, or sequentially.**
**Estimated H100 time:** ~10–12 hours total (3 runs × ~4 hrs each).

---

### Experiment C5: Reward Weight Ablation

**What:** Fix the best (algorithm, KL) from C2/C3/C4. Vary α and β:

| Run | α (EditLens) | β (fluency) | Hypothesis |
|---|---|---|---|
| C5-a | 1.0 | 0.0 | Detection-only → reward hacks |
| C5-b | 1.0 | 0.5 | Balanced (C2 default) |
| C5-c | 1.0 | 1.0 | Strong fluency |
| C5-d | 0.0 | 1.0 | Fluency-only → won't reduce EditLens |

C5-a and C5-d are the control conditions. The interesting comparison is C5-b vs C5-c — how much does doubling the fluency weight change the EditLens score at convergence?

**Outputs:** 4-way comparison of (EditLens score, R_fluency) at training end. 2 qualitative examples from C5-a showing reward hacking behavior (what does the model produce when there is no fluency constraint?).

**What we learn:** The marginal value of the fluency term. If C5-a (no fluency) produces similar EditLens scores to C5-b without obvious quality degradation, the fluency term is acting mainly as a regularizer against edge cases, not as a meaningful shaper of writing style. If C5-a produces clearly worse outputs, the fluency term is doing essential work.

**Estimated H100 time:** ~12–14 hours total (4 runs).

---

## 7. Group D — Architecture & Model Size

### Experiment D1: Model Size Sweep

**What:** Train the best RL configuration (from C2/C3/C4) at three model sizes, same number of steps, same reward:

| Size | Policy Model | Training VRAM | Notes |
|---|---|---|---|
| Small | TinyLlama 1.1B or Qwen 0.5B | ~5 GB | Sanity / lower bound |
| **Medium** | **Llama 3.2 3B Instruct** | **~15 GB** | **C2 baseline** |
| Large | Llama 3.1 8B Instruct | ~35 GB | Upper bound on H100 |

**H100 fit for 8B:** Policy (15 GB) + EL-3B reward (6 GB, shared ref) + activations for 48 rollouts × 512 tokens (~30 GB with gradient checkpointing) = ~51 GB. Fits on 80 GB with gradient checkpointing enabled on the policy. Use `torch.utils.checkpoint` on the policy's transformer layers. Run a quick memory check before committing.

**Outputs:** EditLens score reduction vs model size at step 500. R_fluency vs model size. Wall-clock time per step. The key question: does the 8B model make qualitatively different (more targeted, more human-like) edits, or just quantitatively larger improvements to the same kinds of edits?

**Estimated H100 time:** ~10 hours total (3 sizes).

---

### Experiment D2: Rewriter vs Generator Architecture

**What:** Compare two policy designs:

**(A) Rewriter (default, used in C2–C5):** Policy receives the original AI essay and produces an edited version. Sees the source text; naturally biased toward targeted edits.

**(B) Generator:** Policy receives only the essay topic (a one-sentence prompt) and generates an essay from scratch. No source text. Must learn to write human-like essays from scratch.

**The Grammarly dataset insight applies here:** Rewriters that make small, targeted edits score near LABEL_0 on EditLens. Generators that produce new text from scratch tend to score in the LABEL_2–LABEL_3 range. Architecture (A) has a structural advantage for the EditLens reward. But (B) has more creative freedom and may produce more diverse, interesting outputs.

**Same reward, same training steps, same base model.** Only the input format changes.

**Outputs:** EditLens score, R_fluency, and edit distance to source (only measurable for (A)) at training end. Qualitative comparison of output diversity and style.

**Estimated H100 time:** ~6 hours total (2 runs).

---

## 8. Group E — Evaluation & Analysis

**Purpose:** Rigorous evaluation of the best system from Groups C and D. These experiments produce the final presentation results.

---

### Experiment E1: Multi-Detector Robustness

**What:** Score 100 essays (50 from the best deslopifier's outputs, 50 unmodified AI inputs) against multiple detectors:

| Detector | Type | Expected behavior |
|---|---|---|
| `editlens_Llama-3.2-3B` | Training reward (local) | Should improve — by construction |
| `editlens_roberta-large` | Same family, lighter (local) | Should also improve if signal generalizes within EditLens family |
| GPTZero API | External commercial detector | **True generalization test** |
| Binoculars / Fast-DetectGPT | Statistical, no ML training | Tests distributional change |

For Binoculars/Fast-DetectGPT: these measure perplexity ratios between two LMs. Improvement here means the deslopifier changed the text's fundamental statistical properties, not just its EditLens score.

**Results table format:**

| | EL-3B score ↓ | EL-RoBERTa score ↓ | GPTZero % AI ↓ | Binoculars ↓ |
|---|---|---|---|---|
| Input (raw AI) | ~0.999 | ~0.999 | ~95% | ~high |
| Best deslopifier | ? | ? | ? | ? |
| ∆ | ? | ? | ? | ? |

**Success criteria:** EL-3B improves (by construction). At least one of GPTZero or Binoculars also shows improvement. Transfer = evidence the model learned generalizable human-writing signals.

**Estimated time:** 2 hours + ~$5 GPTZero API cost.

---

### Experiment E2: Human Essay Preservation Test

**What:** Pass 50 human essays (EditLens score ≈ 0.007–0.035) through the best deslopifier. Measure:
- EditLens score before and after (should stay low — ideally unchanged)
- Token-level edit distance (should be small — the deslopifier should barely touch already-human text)
- R_fluency before and after (should not degrade)

**Why:** A well-calibrated deslopifier is approximately the identity function on human text. If it aggressively rewrites human essays and makes them sound more AI-like (EditLens score increases), it has learned to apply a style uniformly rather than respond to the AI signal in the input. That is a failure mode worth reporting.

**Success criteria:** Mean EditLens score of human essays changes by less than 0.05 after deslopification. Mean edit distance is less than 15% of tokens changed.

---

### Experiment E3: Qualitative Analysis — What Did the Model Learn?

**What:** Deep analysis of 20 (input, output) pairs from the best deslopifier. For each pair, annotate which specific changes were made and categorize them:

| Change type | Grammarly analogy | Expected EditLens impact |
|---|---|---|
| Grammar / typo fix | "Fix any mistakes" | Large reduction (near LABEL_0) |
| Hedging language removal | "Make it more direct" | Medium reduction |
| Sentence length variation | "Vary my sentence structure" | Medium reduction |
| AI vocabulary replacement | "Use simpler words" | Medium reduction |
| Specific detail addition | "Make it more detailed" | Small reduction or increase |
| Structural change (bullets → prose) | "Remove formatting" | Small reduction |
| Wholesale rewrite | "Rewrite this" | Small reduction or worse |

The **Grammarly dataset insight** provides a direct benchmark: small targeted edits (top 3 rows) should drive the most score reduction. If the deslopifier is doing mostly the bottom rows, it is less efficient than it should be. This directly tests whether the RL training learned the right strategy.

**Method:**
1. Token-level diff between input and output for each pair
2. DistilBERT v2 token heatmap: per-token P(human) score to identify which tokens were AI-flagged in the input and whether they were changed in the output
3. Manual annotation using the table above

**Outputs:** Annotation table. Summary: what fraction of edits fall into each category? Were the most impactful edit types (grammar, hedging) the most common? This is the richest result for the final presentation.

---

### Experiment E4: Ablation Summary Table

**What:** Compile the clean results table across all experiments:

| Condition | EditLens ↓ | Fluency | Notes |
|---|---|---|---|
| No RL (hill climbing, old reward) | baseline | baseline | Pre-project state |
| No RL (retroactively scored with EditLens) | ~0.999 | — | True starting point |
| REINFORCE, EditLens-only (C5-a) | ? | ? | Reward hacking condition |
| REINFORCE, EditLens + fluency (C2) | ? | ? | **Primary result** |
| GRPO, EditLens + fluency (C3) | ? | ? | Algorithm comparison |
| KL=0.0 (C4-a) | ? | ? | No constraint |
| KL=0.5 (C4-c) | ? | ? | Strong constraint |
| Small model — 1.1B (D1) | ? | ? | Size lower bound |
| Large model — 8B (D1) | ? | ? | Size upper bound |
| Generator architecture (D2) | ? | ? | Architecture comparison |

This table is the core story of the final presentation.

---

### Experiment E5: EditLens Ternary Progression *(New — only possible with Open Pangram)*

**What:** At every eval checkpoint during C2 (every 50 steps), record the full ternary distribution of the 100 held-out essays:
- What fraction are in LABEL_3 (fully AI, score > 0.76)?
- What fraction are in LABEL_2 (heavily edited, 0.43–0.76)?
- What fraction are in LABEL_1 (lightly edited, 0.10–0.43)?
- What fraction are in LABEL_0 (human, score < 0.10)?

Plot as a **stacked area chart over training steps.** Target visualization for the opening slide of the final presentation:

```
100% ┤████████████████████████████████████████████████  Fully AI (LABEL_3)
     │████████████████████████████
     │████████████████████████████████
     │███████████████
 50% ┤               ░░░░░░░░░░░░░░░░  Lightly edited (LABEL_1)
     │        ░░░░░░░
     │
  0% ┤─────────────────────────────────────────────────▶ Training steps
       0     100     200     300     400     500
```

**Why this is the best opening visualization:** It shows the trajectory of learning, not just an endpoint. Even if the model never fully crosses into LABEL_0, a visible shift from "95% fully AI" to "60% lightly edited" is a compelling and honest result. It lets the audience understand what the system is doing without any jargon.

**Implementation:** This requires no additional inference — it is just a different aggregation of the EditLens scores already computed at each eval checkpoint. Add a `plot_ternary_progression()` function to the eval script. The EditLens ternary thresholds come from the paper's calibration procedure (find threshold that maximizes F1 on the validation set).

---

## 9. Implementation Order & Dependencies

```
A1 ✅ COMPLETE
  └── A2 (fluency reward, ~1 day)
        └── A3 (reward formula — locked pending A2 correlation result)
              └── C1 (smoke test — 15 min on H100)
                    ├── C2 (REINFORCE 500 steps) ─────────────────────────┐
                    ├── C3 (GRPO, parallel or after C2)                    │
                    ├── C4 (KL ablation, 3 runs) ← best of C2/C3          ├─ E1–E5
                    └── C5 (reward ablation, 4 runs) ← best of C2/C3/C4  │
                                                                           │
B1 (OOD eval, any time after A1) ────────────────────────────────────────┘
  └── B2 (fine-tune DistilBERT on EditLens data)
B3 (ternary classifier) ← deferred until C2 shows AI-edited progress

D1 (model size sweep) ← best config from C2/C3/C4
D2 (rewriter vs generator) ← parallel with D1
```

**Critical path (minimum viable for final presentation):**
A2 → A3 → C1 → C2 → E5 → E3 → E4

**Next most valuable if time permits:** C3, C4, D1 large model.

**Timeline:**
- Today–tomorrow: A2, A3, B1, B2 (all local/T4, no H100 needed)
- Wednesday: C1 smoke test + C2 first run on H100
- Wednesday–Thursday: C2 full 500 steps; begin C3 in parallel
- Thursday: C4, C5 ablations
- Friday: D1 large model; E1–E5 evaluation
- Weekend: Final presentation prep

---

## 10. Compute Budget

All API costs eliminated. Total external cost: ~$5.

| Experiment | GPU | Est. Time | Cost |
|---|---|---|---|
| A1 extended (300 essays) | T4 / local | 1 hr | $0 |
| A2 (fluency reward validation) | T4 | 1–2 hrs | $0 |
| B1 (OOD classifier eval) | T4 | 1 hr | $0 |
| B2 (fine-tune DistilBERT) | T4 Colab | 2–3 hrs | $0 |
| C1 (smoke test, 10 steps) | H100 | 15 min | $0 |
| C2 (REINFORCE, 500 steps, bs=48) | H100 | 4–5 hrs | $0 |
| C3 (GRPO, 500 steps) | H100 | 6–7 hrs | $0 |
| C4 (KL ablation, 3 runs) | H100 | 10–12 hrs | $0 |
| C5 (reward ablation, 4 runs) | H100 | 12–14 hrs | $0 |
| D1 (model sweep, 3 sizes) | H100 | 10 hrs | $0 |
| D2 (rewriter vs generator) | H100 | 6 hrs | $0 |
| E1 (multi-detector eval) | T4 + API | 2 hrs | ~$5 |
| E2–E5 (evaluation suite) | T4 | 3–4 hrs | $0 |
| **Total H100** | | **~50–55 hrs** | **$0** |
| **Total all compute** | | | **~$5** |

*Recommended H100 session order:* C1 → C2 → (if promising) C3 in parallel → C4 → C5 → D1.

---

## 11. Confirmed Decisions Log

All questions from v1 and v2 are now resolved. This section is a record of confirmed decisions, not open questions.

| # | Decision | Confirmed value | How confirmed |
|---|---|---|---|
| Q1 | Policy model | Llama 3.2 3B Instruct | Memory estimate: 14.9 GB in Scenario 2 |
| Q2 | Reward output format | Continuous [0,1] via softmax → weighted avg → /3 | `probe_editlens.py` |
| Q3 | Fluency reference model | `meta-llama/Llama-3.2-3B` base (shared with reward) | Architectural coherence |
| Q4 | LoRA target modules | q_proj, k_proj, v_proj, o_proj | EditLens QLoRA defaults |
| Q5 | Input/output max length | 512 / 512 tokens | EditLens-RoBERTa limit; EL-3B supports 1024 |
| Q6 | Rollout count | 48 rollouts / step (REINFORCE); K=8 groups×6 (GRPO) | Memory headroom confirmed |
| Q7 | Reward hacking guard | Reject outputs < 50 tokens or < 40% of input length | Spec design |
| Q8 | EditLens normalization | softmax → weighted avg [0,1,2,3] / 3 — NOT sigmoid | `probe_editlens.py` + EditLens repo source |

---

## 12. Final Presentation Narrative

The presentation should tell this story in order:

1. **Opening (E5 chart):** Show the ternary progression chart. "We trained a model to move AI-generated text along this spectrum." This needs no jargon and is immediately compelling.

2. **The problem:** AI-generated text is detectable. Here are real EditLens scores — raw AI slop at 0.999, human text at 0.007. The gap is enormous.

3. **What EditLens measures:** Not binary detection but a continuous spectrum. The key insight from the Grammarly analysis: small targeted edits score near-human. Large rewrites don't. Our deslopifier must learn to make surgical, human-like edits.

4. **Our approach:** REINFORCE / GRPO RL on Llama 3.2 3B, with EditLens as the reward and log-prob fluency as a regularizer. Show the pipeline diagram.

5. **The Pareto frontier (C4):** The KL ablation produces a curve trading off detection evasion vs. fluency. Every application sits at a different point on this curve. This shows the system is principled and tunable.

6. **The ablation story (E4):** Each component's contribution. EditLens-only reward hacks. Fluency-only doesn't reduce detection. The combination works. Larger models do better.

7. **What the model learned (E3):** Before/after examples. The model removed hedging language, varied sentence length, replaced AI-signature vocabulary. These are real stylistic changes, not noise injection.

8. **Robustness (E1):** Transfers to GPTZero — the model learned something general about human writing, not EditLens-specific artifacts.

9. **Limitations and future work:** Non-commercial license. Optimizes against automated detectors, not human judgment. Future: RLHF from human judges; adversarial training as EditLens updates; longer documents.

---

## 13. Confirmed Artifacts

Scripts created during spec validation, committed to the repo:

| File | Purpose | Status |
|---|---|---|
| `scripts/probe_editlens.py` | Canonical EditLens scoring — confirmed inference pipeline | ✅ Confirmed working |
| `scripts/estimate_vram.py` | Memory budget estimation for training configs | ✅ Confirmed working |

These should be moved to `slop_scripts/` and extended as the foundation for all Group A, C, and E scripts.

---

## 14. References

- Thai, K., Emi, B., Masrour, E., & Iyyer, M. (2025). EditLens: Quantifying the Extent of AI Editing in Text. *ICLR 2026*. arXiv:2510.03154.
- Pangram Labs. (2026, March 24). Introducing Open Pangram. https://www.pangram.com/blog/introducing-open-pangram
- `pangram/editlens_Llama-3.2-3B`, `pangram/editlens_roberta-large`. HuggingFace. CC BY-NC-SA 4.0.
- `pangram/editlens_iclr`, `pangram/editlens_iclr_grammarly`. HuggingFace. CC BY-NC-SA 4.0.
- pangramlabs/EditLens source code. https://github.com/pangramlabs/EditLens

---

## Iteration Checklist — Pre-Script Conversion

- [x] A1: EditLens inference pipeline confirmed — formula locked, real score ranges confirmed
- [x] Q8: Normalization confirmed — softmax → weighted avg → /3, not sigmoid
- [x] Memory: Llama 3.2 3B + EL-3B + shared ref = 14.9 GB, 65 GB H100 headroom
- [x] Batch size: upgraded 16 → 48 based on memory headroom
- [x] Model choice: Llama 3.2 3B Instruct confirmed as policy
- [ ] A2: Fluency reward correlation with EditLens — run before setting β in A3
- [ ] A3: Final reward formula config file committed to repo
- [ ] C1: Smoke test passes all checklist items — run before C2
- [ ] C2: First 50 steps show non-zero reward movement — verify before full 500-step run
