# LLM Exploration Log — PromptLab v2

**Project:** Few-shot co-training pipeline for AI text deslopification
**Author:** Ian Lent · STAT 4830 · University of Pennsylvania · Spring 2026
**Model used throughout:** Claude Sonnet (claude.ai)

This log documents the substantive AI-assisted reasoning across the project: architectural decisions, self-critiques, design specs, and the feedback loops that shaped the final system. Trivial exchanges are omitted.

---

## 1. Architectural Pivot: Prompt→Prompt to Essay→Essay

**Context:** After the organic baseline run (val slop 0.391, step ~150), the question was whether to continue scaling the prompt-mode approach or change the architecture entirely.

**AI critique of prompt-mode:**
> The core problem with prompt→prompt distillation is that the training signal reaches the rewriter through two hops: rewriter produces a prompt, an LLM generates an essay from that prompt, the essay is scored. The LLM in the middle introduces variance the rewriter cannot control — the same rewritten prompt does not always produce the same essay. This variance dilutes the gradient signal. The rewriter cannot cleanly attribute a good slop score to the quality of its rewritten prompt because the generator's contribution is confounded with it. Essay→essay removes this intermediate step: the rewriter directly maps what the detector scores.

**Decision:** Switch to essay-to-essay mode. Result: 52% improvement (0.391 → 0.189) from architecture change alone, before any hyperparameter tuning.

**Key insight logged:** One step of indirection versus two is not a minor implementation detail — it fundamentally changes how much of the training signal reaches the model. This generalizes: any time you have a proxy between your model's output and the loss signal, that proxy introduces noise that degrades learning.

---

## 2. The Alpaca Experiment: Data Quality vs. Quantity

**Context:** With ~34 organic co-training pairs, the question was whether augmenting with 2272 Alpaca-derived essay pairs would help.

**Pre-experiment reasoning:**
> Alpaca pairs come from instruction-following prompts, not from the evolutionary optimizer's best prompts. The distribution is different. The model will receive gradients from both the evolutionary signal (reduce slop) and the Alpaca signal (follow instructions well). If these are contradictory — and they likely are, since Alpaca essays are generated to be coherent and complete, not to evade detectors — the two signals will interfere.

**Result:** Val slop went from 0.391 to 0.580 — degradation confirmed. The prediction was correct.

**Lesson formalized:** Data distribution alignment dominates data quantity. 34 well-distributed pairs outperform 2306 mixed-distribution pairs. This is a clean empirical confirmation of a theoretical principle that is sometimes treated as obvious but rarely demonstrated so directly in a course project.

---

## 3. Few-Shot Injection: Spec and Rationale

**Original spec (from design session):**

The optimizer should maintain a growing pool of winning essays (lowest detector scores). Before each generation call, retrieve top-k essays from this pool and prepend them as in-context examples. The language model then generates conditioned on demonstrated low-slop style, not from its prior.

**Self-critique of the mechanism:**
> The risk is context dilution: as the pool grows, including all examples bloats the prompt and the model's attention is spread across too many examples. The practical fix is to cap at top-k (e.g. k=5) by score, not by recency. This preserves the strongest signal without diluting attention.

**Why it works (AI explanation):**
> This exploits in-context learning — one of the most well-documented properties of large language models. Given examples of a task in the prompt, a capable model can perform that task without any gradient updates to its weights. The examples shift the model's generation distribution toward the demonstrated pattern at inference time. The improvement compounds: better search → better examples in pool → better context for future rounds → better search.

**Infrastructure requirement identified:** `_write_pair_record_with_essays()` must save `best_essay` text alongside each pair record. Without this, the fast-path in `generate_essay_pairs.py` re-generates essays from the saved prompt rather than using the already-discovered best essay — reintroducing the generation variance that essay mode was designed to eliminate.

---

## 4. T5-Base Limitations: Self-Critique

**AI self-critique of T5-base choice:**
> T5 was pretrained on summarization. Its outputs are shorter than its inputs — the pretraining objective actively reinforces compression. Essay rewriting should preserve length while changing style. These are in conflict. The slop reduction the rewriter achieves is real and mechanistically sound. The output quality as a standalone essay suffers from this pretraining mismatch. Repetition in outputs is beam search looping when the model loses confidence — it happens when T5 has no confident continuation and cycles on a high-probability n-gram.

**What this predicts for demo behavior:**
> v2 and v3 outputs will be consistently shorter than the baseline. On topics where the model is uncertain (low training signal, unusual topic), expect repetitive loops. On topics with strong training signal (social media, moral responsibilities of wealthy nations), expect cleaner outputs.

**Observed:** Confirmed. The vaccination policies topic produced clean v3 output; the social media topic showed repetitive loops in v2 (degenerate beam search) and reasonable prose in v3 (better generalization from wider topic coverage).

**Future fix:** T5-large or a generative decoder (e.g. Llama 3.2 1B with LoRA). The decoder has no summarization pretraining bias and would preserve length. The rest of the training pipeline (essay-to-essay, LoRA, curriculum, early stopping on slop_mean) transfers directly.

---

## 5. Training Discipline: The Metric Divergence Problem

**Problem identified:**
> After step 50, val_loss and slop_mean diverge in opposite directions: val_loss continues falling (model predicts training tokens more confidently) while slop_mean rises (essays become more detectable). This is the signature of memorization in a small-data regime. The model is learning the training essays, not the transformation.

**AI explanation of why this happens:**
> With 40 training pairs and ~800K trainable parameters, the parameter-to-data ratio is 20,000:1. The noise-to-signal ratio in the stochastic gradient (the NQM) implies that after ~50 steps, gradient noise dominates. Each subsequent step is as likely to hurt as to help. The loss keeps falling because cross-entropy has no floor — the model can always become more confident in its memorized outputs. But slop_mean has a floor set by the detector, and memorization does not help reach it.

**Implementation spec:**
- Custom `SlopEarlyStoppingCallback` runs full inference on val pairs every `eval_steps`
- Computes mean detector score across all val outputs
- Saves checkpoint when `rewriter_slop_mean` is minimized
- Uses `num_beams=4, do_sample=False` (deterministic beam search) so checkpoint comparisons are stable — same input → same output → score differences reflect weights not sampling noise

**Generalization noted:**
> This is the same failure mode as reward hacking in RLHF — the model optimizes the proxy (cross-entropy loss / reward model) without improving on the true task. The solution is identical: evaluate on the true metric, use it for stopping decisions.

---

## 6. LR Sweep: Why Conservative Wins

**Pre-sweep prediction:**
> With ~50 useful steps before the noise floor, LR=3e-4 is probably too high. Large steps in a noisy gradient landscape overshoot the optimum within the first few steps. LR=1e-4 moves more carefully and should track the signal more closely in the available budget.

**Result:** LR=1e-4 → 0.0998; LR=3e-4 → 0.196. Prediction confirmed. 2× improvement from a single hyperparameter.

**AI analysis:**
> The reason the gap is so large is the combination of small dataset (noisy gradients) and small training budget (~50 steps). In a larger-data setting with more steps, the two LRs would converge — the optimizer at LR=3e-4 would eventually settle. Here, 50 steps is not enough recovery time. The training budget is too small to waste early steps on overshooting.

---

## 7. LoRA Rank Ablation

**Question:** Is r=16 necessary or does r=8 provide sufficient capacity?

**AI prediction before ablation:**
> With 40 training pairs, both ranks are likely sufficient. The bottleneck is not model capacity — it's data and the noise floor. Expect the gap to be small, probably under 0.01.

**Result:** r=16 → 0.0998; r=8 → 0.1058. Gap: 0.006. Prediction confirmed.

**Conclusion:** r=16 is confirmed appropriate. Neither rank is underfitting. The regularization provided by both is sufficient for this dataset size. Future work with more pairs might find a larger rank beneficial, but at 40 pairs the marginal gain from r=16 over r=8 is minimal.

---

## 8. Demo Notebook Bug: Adapter Weight Clobbering

**Bug identified:**
> `PeftModel.from_pretrained()` modifies the base model object in-place during adapter loading. If the same base model object is passed to three sequential `PeftModel.from_pretrained()` calls (v1, v2, v3), each call overwrites the previous adapter's weights. Result: all three "adapters" run the last-loaded weights — v2 and v3 produce identical outputs.

**This was observed in the demo:** v2 and v3 slop scores were identical (0.3416 each) and outputs were character-for-character the same.

**Fix:** `copy.deepcopy(base_model)` before each `PeftModel.from_pretrained()` call gives each adapter its own isolated copy of the base weights.

**Why this is non-obvious:** PEFT's documentation does not prominently flag that adapter loading is in-place. The bug produces no error — it silently produces wrong behavior. The only signal is that multiple adapters produce identical outputs, which is only detectable if you know they should differ.

---

## 9. Pipeline Diagrams: Design Specs

The following diagrams were produced during the project and are committed to `figures/`:

- **fig1_optimization_formula.svg** — horizontal layout of the drift penalty formula with labeled terms (task reward / meaning / overlap / fluency)
- **fig2_stage1_pipeline.svg** — two-track flow: generic prompt → baseline essay (high-slop) vs. deslop → best prompt → best essay (low-slop), both converging to essay pairs
- **fig3_few_shot_injection.svg** — three-panel progression: Round 1 (no examples), Round 2 (one winning essay prepended), Round N (full pool prepended), showing slop improvement across rounds
- **fig4_unified_pipeline.svg** — full end-to-end system from essay topics through optimizer, co-training accumulation, training config, early stopping, to inference output

All diagrams use a consistent dark color scheme (purple for optimizer components, teal for co-training/rewriter, coral for detector/scoring).
