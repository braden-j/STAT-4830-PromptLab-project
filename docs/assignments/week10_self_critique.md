# Self-Critique: Week 10

*Focused, actionable critique of the current ML project draft. Max 1 page.*

---

## OBSERVE

- **Artifacts reviewed:** Co-training loop (`cotrain/loop.py`), detector wrapper (`detector/model.py`), T5+LoRA rewriter in prompt-mode (`rewriter/train.py`), initial contrastive pair store (`outputs/cotrain/prompt_pairs.jsonl`, ~34 organic pairs), Alpaca augmentation run.
- **Code re-check:** Full pipeline runs end-to-end: topic → optimizer → essay generation via Groq → SlopDetector scoring → pair writing. Prompt-mode rewriter trains and produces checkpoints. Best result so far: val slop 0.391 at step ~150 on 34 organic pairs. Essay-mode pivot just completed: val slop 0.189 at step 50 on 11 pairs — a 52% improvement over prompt-mode baseline from architecture change alone.
- **Reactions:** The Alpaca augmentation experiment (2306 pairs → val slop 0.580) was a clear and informative failure — the distribution mismatch produced contradictory gradients. The essay→essay pivot result is striking and suggests the architecture was the binding constraint, not the data or hyperparameters. The step-50 best checkpoint is suspicious and needs investigation: is this an early stopping artifact or a real noise floor?

---

## ORIENT

**Strengths (max 3)**

- **Clean negative result from Alpaca augmentation:** The prediction was made before the experiment (distribution mismatch → contradictory gradients) and confirmed quantitatively. This is exactly the kind of ML experiment discipline the course asks for — hypothesis first, then evidence.
- **Architecture pivot is well-motivated and validated:** The prompt→prompt indirection problem was identified theoretically (two hops from detector signal, LLM variance confounds training) and then confirmed empirically with a 52% improvement. The causal reasoning is sound.
- **Fixed detector is the right call:** Keeping `editlens/roberta-large` frozen throughout gives the optimizer a stable reward surface. The alternative (co-training the detector) would have introduced an arms race with no clear convergence criterion.

**Areas for improvement (max 3)**

- **Step-50 noise floor is unexplained:** The best checkpoint is always step 50 regardless of dataset size or run length. The 1300-step long run degraded to 0.759 — but we don't yet have a principled explanation of *why* 50 steps is the limit. This needs to be framed in terms of the parameter-to-data ratio and NQM before it can inform the scaling strategy.
- **Early stopping is tracking the wrong metric:** `eval_loss` is being used for checkpoint selection but it diverges from `slop_mean` after step 50. The model is memorizing training token sequences, not learning the stylistic transformation. A custom callback that runs full inference on val pairs and tracks actual detector scores is needed.
- **Dataset is too small and few-shot injection is not yet implemented:** 11–34 organic pairs is insufficient to characterize whether the essay-mode architecture generalizes across topics. The few-shot injection mechanism — prepending winning essays as in-context examples to future generation calls — is specified but not yet running. Without it, each optimizer round starts cold and the search does not compound.

**Critical risks/assumptions (2–3 sentences)**

The current architecture assumes T5-base has sufficient capacity to learn a meaningful stylistic transformation from 11–34 examples, which may be true at step 50 but is untested at larger data scales. The drift penalty weights (α, β, γ) were set manually with no systematic ablation — if they are too loose, the optimizer will find degenerate low-slop outputs; if too tight, the search space is overconstrained. The step-50 noise floor, if it persists at 40+ pairs, will cap improvement regardless of architecture or learning rate.

---

## DECIDE

**Concrete next actions (max 3)**

1. **Implement `SlopEarlyStoppingCallback`:** Replace `eval_loss`-based early stopping with a callback that runs `num_beams=4, do_sample=False` inference on all val pairs every `eval_steps` and saves the checkpoint that minimizes `rewriter_slop_mean`. This is the single change most likely to correctly identify the best model.
2. **Implement and validate few-shot injection:** Modify `cotrain/loop.py` to retrieve top-k lowest-scoring essays from the accumulated pair store and prepend them as in-context examples before each generation call. Run 20+ topics to confirm the compounding improvement across rounds before scaling further.
3. **Frame the step-50 observation as NQM:** Compute the parameter-to-data ratio (800K params / N pairs) and use it to predict the noise floor step count. This turns an empirical observation into a principled prediction that can guide dataset scaling targets.

---

## ACT

**Resource needs (2–3 sentences)**

The `SlopEarlyStoppingCallback` requires only HuggingFace Trainer hooks and the existing SlopDetector — no new dependencies. Few-shot injection requires modifying the Groq generation call in `loop.py` and adding a retrieval step over the JSONL store — one afternoon of implementation. The NQM framing requires reviewing STAT 4830 §9 material on Adam and noise floor; no new tools needed.
