# Self-Critique: Week 12

*Focused, actionable critique of the current ML project draft. Max 1 page.*

---

## OBSERVE

- **Artifacts reviewed:** Final T5+LoRA rewriter (essay mode, r=16, LR=1e-4), full co-training pair store (372 pairs, 40 contrastive, 77 topics), LR sweep results (3e-4 vs 1e-4), LoRA rank ablation (r=8 vs r=16), demo notebook (`notebooks/promptlab_v2_demo.ipynb`), all three adapter checkpoints on Drive.
- **Code re-check:** Full pipeline runs: few-shot injection accumulating pairs → contrastive filtering → T5+LoRA training with `SlopEarlyStoppingCallback` → inference via `rewrite_essay()`. Demo notebook loads all three adapters correctly (deepcopy bug fixed). Best result: val slop 0.0998 at step 50 (LR=1e-4, r=16, 40 pairs, 20 val). Ablation confirms r=16 over r=8 by 0.006.
- **Reactions:** The LR sweep result (2× improvement from a single hyperparameter) is the most practically instructive finding of the project — it directly demonstrates the noise floor argument from §9. The demo notebook revealed the adapter weight clobbering bug immediately; without the deepcopy fix, v2 and v3 produced identical outputs. The batch evaluation results are mixed: gene editing topic improved 56.9%, but remote work degraded −115.5%, and the mean improvement across 5 topics is only 8.3% — far below the 74% on the held-out val pairs.

---

## ORIENT

**Strengths (max 3)**

- **LR sweep is textbook §9 validation:** The prediction (conservative LR wins in a noise-limited regime with ~50 useful steps) was made before the experiment and confirmed with a 2× improvement. This is a clean empirical demonstration of the NQM principle — the training budget is too small to waste early steps overshooting the optimum.
- **Few-shot injection works as specified:** Val slop improved monotonically as the pool grew from 0 examples (round 1) to a rich library (round N). The compounding mechanism — better examples → better search → better examples — is confirmed by the trajectory from 0.391 (cold start) to 0.0998 (77 topics, full injection).
- **Stopping on the right metric was decisive:** `SlopEarlyStoppingCallback` correctly identifies step 50 as the best checkpoint in every run. Without it, continued training would have selected a memorized checkpoint with higher slop and lower loss — exactly the divergence observed in the Phase 1 long run.

**Areas for improvement (max 3)**

- **Generalization across topics is weaker than val results suggest:** The 5-topic batch evaluation (mean 8.3% improvement) is far below the 74% on held-out val pairs. This discrepancy likely reflects the held-out val pairs being drawn from the same optimizer distribution as training — topics where the optimizer found strong low-slop essays. Out-of-distribution topics (remote work: −115.5%) perform worse than baseline, which is a real generalization failure.
- **T5-base output quality is degraded by pretraining mismatch:** v2/v3 outputs are consistently shorter than the baseline and produce repetitive loops on uncertain topics. The slop reduction is real but the essays are not standalone readable documents. The backbone choice (summarization-pretrained T5) is in tension with the rewriting task and was not ablated against alternatives.
- **Topic selection was unsystematic:** No principled methodology was used to select the 77 topics beyond an initial manual set and Alpaca-sampled additions. There is no evidence that the topic distribution covers the space where improvement is most needed or most achievable. Pre-scoring topics by baseline slop (focus compute where headroom is largest) was identified as a next step but not implemented.

**Critical risks/assumptions (2–3 sentences)**

The 74% val slop improvement is measured on pairs drawn from the same optimizer distribution as training — topics where the evolutionary search found strong low-slop essays — which inflates the reported result relative to out-of-distribution performance. The system optimizes specifically against `editlens/roberta-large`; generalization to other detectors (GPTZero, perplexity-based methods) is unknown and the −115.5% on remote work suggests the rewriter may have learned detector-specific shortcuts on some topics rather than genuine stylistic transformation. The 40-pair dataset, while sufficient to reach the noise floor, leaves no room to test whether additional data would yield further improvement without running the full optimizer again.

---

## DECIDE

**Concrete next actions (max 3)**

1. **Replace T5-base with a generative decoder backbone:** T5-large or Llama 3.2 1B with LoRA would eliminate the summarization pretraining bias and preserve essay length. The rest of the training pipeline (essay→essay, curriculum, early stopping on slop_mean) transfers directly. This addresses both the length compression and repetition artifacts.
2. **Close the loop — rewriter outputs feed back into the optimizer:** Use the rewriter's deslopified outputs as additional entries in the few-shot injection pool, then retrain. Each stage currently runs once; making it iterative would compound the improvement rather than treating search and distillation as a single-pass pipeline.
3. **Pre-score topics and run cross-detector evaluation:** Filter to topics where baseline slop > 0.8 before running the optimizer (maximize improvement headroom). Evaluate the final checkpoint against at least one perplexity-based detector to bound the generalization risk.

---

## ACT

**Resource needs (2–3 sentences)**

Backbone replacement requires a Colab A100 session for training (T5-large fits in ~16GB; Llama 3.2 1B with LoRA in ~8GB) and approximately 2–3 hours of training time. The iterative loop closure requires no new infrastructure — only re-running the existing pipeline with the rewriter's outputs added to the JSONL store before the next optimizer round. Cross-detector evaluation requires API access to GPTZero or similar; the existing `eval/metrics.py` framework can accommodate additional scoring functions without major refactoring.
