# PromptLab v2 — Development Log

**Ian Lent · STAT 4830 · University of Pennsylvania · Spring 2026**

This log documents the progression of the project: what was built, what broke, what was learned, and what changed as a result. Entries are ordered chronologically and focus on decisions rather than implementation minutiae.

---

## Phase 0 — Problem Framing and Initial Infrastructure

**Objective:** Establish a working scaffold: a fixed detector, a topic set, and a basic co-training loop that could accumulate essay pairs.

The first non-trivial decision was to fix the detector throughout the entire project. The alternative — co-training both the detector and the optimizer — creates an adversarial arms race where neither side converges stably. Fixing `editlens/roberta-large` as the scoring oracle gives the optimizer a stable landscape to search over. The cost is that the trained rewriter may not generalize to other detectors. This trade-off was accepted as appropriate for a course project with constrained compute.

Initial topic selection was manual: ~25 argumentative and ethical essay topics chosen to produce the most formulaic AI text. These were later expanded to 100 via sampling from the Stanford Alpaca instruction dataset, filtered for essay-compatible prompts.

**Outcome:** Working co-training loop (`cotrain/loop.py`), detector wrapper (`detector/model.py`), YAML topic config (`configs/topics_alpaca_diverse.yaml`).

---

## Phase 1 — Prompt-Mode Baseline (v1)

**Architecture:** T5-base + LoRA trained on (generic prompt → rewritten prompt) pairs. The rewritten prompt is passed to an LLM to generate the final essay.

**Initial results:** Val slop 0.391 at step ~150 on ~34 organic pairs.

**Problem identified — overfitting on long runs:** Running for 1300 steps on the same ~34 pairs produced val slop 0.759. `eval_loss` continued declining while `slop_mean` rose — the classic signature of memorization. The model learned to predict the token sequences of training prompts rather than learning the underlying transformation. This observation directly motivated two subsequent decisions: (1) early stopping on `slop_mean` rather than `eval_loss`, and (2) accumulating more diverse training data before scaling training steps.

**Problem identified — Alpaca augmentation:** Adding 2272 Alpaca-derived pairs degraded val slop from 0.391 to 0.580. These pairs were generated from instruction-following prompts rather than the evolutionary optimizer's best prompts — a different distribution. Mixed-distribution training produced contradictory gradients that degraded both signals. Lesson: data distribution alignment dominates data quantity.

**Problem identified — prompt-mode indirection:** The rewriter produces a prompt; an LLM generates an essay from that prompt; the essay is scored. The LLM in the middle introduces variance the rewriter cannot control. The same rewritten prompt does not always produce the same essay, so the training signal is noisy and confounded.

**Decision:** Pivot to essay-to-essay mode.

---

## Phase 2 — Essay-Mode Pivot (v2)

**Architecture change:** T5-base + LoRA trained directly on (baseline essay → optimizer's best essay) pairs. The rewriter maps the thing the detector scores directly.

**Immediate result:** Val slop 0.189 at step 50 with only 11 pairs — a 52% improvement over the v1 baseline from architecture change alone.

**Observations on the best-checkpoint location:** The best checkpoint was always at step 50, regardless of how many steps were run. With ~11-40 training pairs and ~800K trainable parameters, the noise-limited regime is reached very quickly. After ~50 steps, gradient noise dominates signal. Continuing to train degrades the task metric even as `eval_loss` continues to fall.

**Custom early stopping callback introduced:** `SlopEarlyStoppingCallback` monitors `rewriter_slop_mean` rather than `eval_loss`. Requires running full inference every `eval_steps` — expensive relative to loss computation but the only honest measure of task performance.

**Deterministic beam search introduced for eval:** `num_beams=4, do_sample=False` ensures checkpoint comparisons are stable — same input always produces same output, so score differences reflect weight differences not sampling randomness.

**Curriculum sorting introduced:** Training pairs ordered by `slop_gap` descending. Clearest supervision signal presented first; ambiguous pairs last.

**Infrastructure fix — `best_essay` saving:** `_write_pair_record_with_essays()` modified to save `best_essay` text alongside each pair record. Without this, the fast-path in `generate_essay_pairs.py` re-generates essays from the saved prompt, reintroducing generation variance.

**Dataset built to 21 pairs:** Val slop 0.247 on a harder val set (12 pairs vs 6). Absolute improvement held.

---

## Phase 3 — Scaling and Hyperparameter Tuning (v3)

**Dataset scaled to 40 contrastive pairs / 20 val pairs** across 77 unique topics, using the full few-shot injection mechanism. Filtering criterion: slop_gap > 0.1, deduplicated by topic.

**LR sweep:**
- LR=3e-4, r=16: val slop 0.196
- LR=1e-4, r=16: val slop 0.0998 (**best**)

The twofold improvement from the more conservative learning rate was the largest single gain in Phase 3. With only ~50 useful training steps, LR=3e-4 overshoots the optimum in the early steps and spends the remaining budget oscillating.

**LoRA rank ablation:**
- LR=1e-4, r=16: val slop 0.0998
- LR=1e-4, r=8: val slop 0.1058

Gap of 0.006 confirmed r=16 appropriate. Bottleneck is data, not model capacity.

**Final model configuration:**
- Architecture: T5-base + LoRA r=16, essay-to-essay mode
- Optimizer: AdamW, weight_decay=0.01, LR=1e-4, cosine decay
- Training data: 40 contrastive pairs, curriculum-ordered
- Val data: 20 pairs
- Early stopping: `rewriter_slop_mean`, deterministic beam search n=4
- Best checkpoint: step 50
- **Val slop: 0.0998** (−74% vs v1 baseline, −88% vs raw AI output)

**Adapter paths (Google Drive):**
- v1 prompt-mode: `promptlab-v2-outputs/2026-04-14_17-48/outputs/rewriter/t5_lora_r8/lora_adapter`
- v2 essay pivot: `promptlab-v2-outputs/2026-04-15_04-40_essay_mode_exp_A/outputs/rewriter/t5_lora_r8/lora_adapter`
- v3 final: `promptlab-v2-outputs/2026-04-19_16-47_FINAL_MODEL/lora_adapter/lora_adapter`

---

## Key Decisions Summary

| Decision | Motivation | Outcome |
|----------|-----------|---------|
| Fix detector throughout | Stable optimization landscape | Tractable search problem |
| Essay→essay vs prompt→prompt | Remove indirection, clean signal | −52% slop from architecture alone |
| Reject Alpaca augmentation | Distribution mismatch → contradictory gradients | Confirmed: data quality > quantity |
| Early stopping on slop_mean | eval_loss diverges from task metric after step 50 | Correct best checkpoint selection |
| LR=1e-4 vs 3e-4 | Small budget + noisy gradients → conservative LR wins | −49% additional improvement |
| Few-shot injection | Self-improving search via in-context learning | Compounding gain across rounds |
| Curriculum ordering | Clearest signal first, ambiguous last | Consistent with learning theory |
| deepcopy before PeftModel load | In-place adapter loading clobbers prior adapters | Fixed demo: v2 and v3 now distinct |

---

## Open Questions for Future Work

1. **Cross-detector generalization:** Does the rewriter transfer to GPTZero, Originality.ai, or perplexity-based methods? Untested.

2. **Closing the loop:** Use the rewriter's own outputs to seed the optimizer's few-shot library, then retrain. Each stage currently runs once; iterating should compound.

3. **Backbone replacement:** T5-large or a generative decoder resolves the length compression problem while preserving the training pipeline.

4. **Systematic topic scoring:** Pre-filter topics by baseline slop score — concentrate compute where improvement headroom is largest.

5. **Few-shot pool selection strategy:** Currently top-k by score. Alternatives: maximum diversity, recency-weighted, or learned retrieval.
