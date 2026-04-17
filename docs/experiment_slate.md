# High-Intensity Slop Experiment Slate

This document is the research-facing companion to [STAT4830_Experiment_Spec_v2.md](/Users/jgold/STAT-4830-PromptLab-project/docs/STAT4830_Experiment_Spec_v2.md), which remains the source of truth for the final tournament implementation. The goal here is slightly different: explain which experiments are most worth running if we want to learn something real about AI slop, how recognizable it is, and which transformer fine-tuning choices actually reduce it rather than merely hiding from one metric.

The organizing idea is a controlled tournament. Most experiments produce trained model checkpoints under the same Pangram judge, the same source pool, and the same comparability rules. The last two experiments are interpretation layers that tell us what the winners actually learned about slop recognition and whether any apparent gains are genuine.

In every experiment, `Why it matters` is meant to cover both scientific importance and project impact, and `Interesting if false` is there to make sure a null or negative result still advances the story.

## Tournament-Aligned Setup

- **Shared source pool:** derive all training and evaluation data from the same essay, prompt-response, and QA universe already used across the notebooks. Essays come from the Kaggle verifier and token-classifier notebooks, prompt-response text comes from HH-RLHF and Dolly-style setups, and QA examples come from the DefAn notebook.
- **Official judge:** `pangram/editlens_Llama-3.2-3B` is the final ranking model for checkpoint selection and headline results.
- **Fast development scorer:** `pangram/editlens_roberta-large` is the cheap inner-loop scorer for filtering, debugging, and early checkpoint selection.
- **Comparability rule:** `TinyLlama` entrants use the same source split, the same max sequence length, and the same effective training token budget. They can differ in parameterization, objective, or data weighting, but not in the basic task or source universe.
- **Qualification gates:** a model only counts as a meaningful improvement if it passes semantic similarity, length-discipline, and validity checks in addition to improving Pangram score.
- **Baselines to beat:** frozen TinyLlama, the current hill-climbed prompt baseline, and the existing prompt-rewriter setup are reference points, but they are not the core experiments in this slate.

## Ranked Experiments

### 1. Attention-Only LoRA vs Full-Module LoRA

- **Question:** On the same rewrite task and the same base pair pack, is narrow adapter tuning enough to reduce recognizable slop, or do we need broader low-rank adaptation?
- **Spec mapping:** `M1` vs `M2`
- **Source notebooks:** `notebooks/Hill_Climb_notebooks/slop_pipeline_colab.ipynb`, `notebooks/ai_slop_prompt_rewriter_colab_v3 (1).ipynb`, `notebooks/build_mirror_dataset (2).ipynb`, `notebooks/defan_slop_gan_unsloth (1).ipynb`
- **Method:** Train two `TinyLlama` rewrite models on the same `P0` data pack. The first uses attention-only LoRA as the cheapest serious PEFT baseline. The second expands LoRA coverage to attention plus MLP-style modules while keeping the task, split, and token budget fixed. Rank both under Pangram 3B and compare how much extra slop reduction the broader adapters buy.
- **Why it matters:** This is the cleanest low-cost test of whether slop reduction is a shallow adaptation problem or already complex enough to need broader parameter movement. If broader LoRA wins decisively, that tells us the model needs more than a light nudge to stop producing recognizable AI texture.
- **Interesting if false:** If attention-only LoRA is already competitive, then cheap PEFT may be sufficient and the harder parts of the tournament can focus on data and objective design instead of parameter count.
- **Tangible outputs:** Two trained checkpoints, a direct Pangram delta comparison, adapter-scope ablation plots, and held-out rewrite examples where the models diverge.
- **Rough compute:** About 8-12 GPU-hours total; roughly $15-$25 if both runs share the same cached data pack.

### 2. LoRA vs Last-Block Unfreeze vs Full Fine-Tune

- **Question:** Does recognizable slop live in shallow readout behavior, or does reducing it require deeper transformer reconfiguration?
- **Spec mapping:** `M2` vs `M3` vs `M4`
- **Source notebooks:** `notebooks/ai_slop_prompt_rewriter_colab_v3 (1).ipynb`, `notebooks/Hill_Climb_notebooks/slop_pipeline_colab.ipynb`, `notebooks/defan_slop_gan_unsloth (1).ipynb`
- **Method:** Hold the backbone, data pack, split, decode settings, and token budget fixed. Compare three update regimes: broad LoRA, last-block unfreeze plus LM head, and full-model fine-tuning. Evaluate all three with Pangram 3B and the qualification gates, then inspect where they differ by domain and failure mode.
- **Why it matters:** This is the strongest architectural learning experiment in the slate. It tells us whether slop recognition and slop removal are mostly governed by shallow adaptations or whether full-model changes are needed to push outputs into a less recognizable regime.
- **Interesting if false:** If full fine-tuning does not clearly beat PEFT, that is a powerful practical result because it says large-scale weight movement is not the main bottleneck.
- **Tangible outputs:** Three checkpoints, a parameterization leaderboard, compute-vs-performance plots, and examples showing whether deeper updates reduce boilerplate or just change style superficially.
- **Rough compute:** About 14-22 GPU-hours total, depending on the full fine-tune; roughly $30-$50.

### 3. Curriculum Fine-Tuning

- **Question:** Does an easy-to-hard training schedule help a rewrite model learn more human-looking editing behavior than a flat data order?
- **Spec mapping:** `M5`
- **Source notebooks:** `notebooks/stat4830_token_slop_classifier_v2_(1) (1).ipynb`, `notebooks/build_mirror_dataset (2).ipynb`, `notebooks/ai_slop_prompt_rewriter_colab_v3 (1).ipynb`
- **Method:** Keep the winning parameterization from the first bracket fixed, then train on the `P1` curriculum pack, which orders examples by Pangram gap or rewrite difficulty. Compare against the plain `P0` version under the same evaluation harness.
- **Why it matters:** The repo already hints that curriculum can matter for the recognition side. This experiment asks whether the same idea helps the generator learn edits that Pangram reads as more human-like instead of merely less blatant.
- **Interesting if false:** If curriculum does not help, that is still informative because it suggests generation-side deslopification is limited more by objective alignment or data type than by example ordering.
- **Tangible outputs:** One curriculum-trained checkpoint, easy/medium/hard validation curves, and a direct comparison to the matching non-curriculum model.
- **Rough compute:** About 6-10 GPU-hours; roughly $10-$18.

### 4. Hard-Negative Fine-Tuning

- **Question:** Does training on the examples Pangram finds most suspicious or hardest to improve make a rewrite model better at reducing recognizable slop?
- **Spec mapping:** `M6`
- **Source notebooks:** `notebooks/stat4830_token_slop_classifier_v2_(1) (1).ipynb`, `notebooks/build_mirror_dataset (2).ipynb`, `notebooks/Hill_Climb_notebooks/slop_pipeline_colab.ipynb`
- **Method:** Build a `P2` hard pack from the shared source pool by upweighting high-Pangram examples or examples that prior models fail to improve. Train the same architecture and update rule as the control, then compare the result to uniform training on `P0`.
- **Why it matters:** This is the generator-side analogue of hard-negative mining. If it works, it suggests that slop recognition and slop removal both benefit from concentrating on the hard boundary cases instead of average examples.
- **Interesting if false:** If hard-negative weighting hurts performance, that would suggest the rewrite model needs broad stylistic coverage more than concentrated pressure on extreme cases.
- **Tangible outputs:** One hard-negative checkpoint, a mined hard-example pack, before-and-after Pangram distributions, and a failure gallery of examples that remain stubbornly AI-looking.
- **Rough compute:** About 6-10 GPU-hours plus data filtering time; roughly $10-$18.

### 5. Preference Tuning with Pangram-Ranked Rewrites

- **Question:** Is Pangram-ranked preference supervision more aligned with recognizable slop reduction than plain supervised fine-tuning?
- **Spec mapping:** `M7`
- **Source notebooks:** `notebooks/ai_slop_prompt_rewriter_colab_v3 (1).ipynb`, `notebooks/build_mirror_dataset (2).ipynb`, `notebooks/Hill_Climb_notebooks/slop_pipeline_colab.ipynb`
- **Method:** For each shared-source input, generate multiple candidate rewrites, rank them by Pangram delta under a similarity filter, and train a DPO model on those best-vs-worst pairs. Compare the resulting checkpoint to the strongest SFT baseline on the same held-out set.
- **Why it matters:** This is the most direct offline test of whether Pangram can serve as a meaningful slop-preference signal rather than just a detector. If it works, it gives a cleaner alignment path than going straight to online RL.
- **Interesting if false:** If DPO fails to beat SFT, that tells us Pangram-induced pairwise preferences may be too noisy, too shallow, or too easy to exploit for ranking-based tuning.
- **Tangible outputs:** One DPO checkpoint, a reusable preference pair bank, pairwise win-rate summaries versus SFT, and examples where Pangram ranking and human-looking quality agree or disagree.
- **Rough compute:** About 8-12 GPU-hours plus candidate generation; roughly $15-$25.

### 6. Direct Pangram-Reward Optimization

- **Question:** When we optimize directly against Pangram, do we get genuinely better rewrites or just more sophisticated reward hacking?
- **Spec mapping:** `M8`
- **Source notebooks:** `notebooks/defan_slop_gan_unsloth (1).ipynb`, `notebooks/Hill_Climb_notebooks/slop_pipeline_colab.ipynb`, `notebooks/ai_slop_prompt_rewriter_colab_v3 (1).ipynb`
- **Method:** Start from the strongest offline checkpoint and run GRPO with Pangram as the online reward, plus the same semantic, length, and fluency safeguards described in the tournament spec. Compare the resulting model to the best DPO or SFT entrant.
- **Why it matters:** This is the most consequential optimization experiment in the slate because it answers whether direct anti-slop reinforcement learning is actually useful or just seductive in theory.
- **Interesting if false:** If GRPO underperforms or becomes unstable, that is still a strong result because it shows that online detector-driven optimization is riskier than offline preference learning for this task.
- **Tangible outputs:** One RL checkpoint, reward and KL curves, checkpoint-vs-checkpoint Pangram comparisons, and qualitative examples of where RL improves or drifts.
- **Rough compute:** About 10-18 GPU-hours; roughly $20-$35.

### 7. Hill-Climb Distillation

- **Question:** Can anti-slop behavior discovered by prompt search be compressed into model weights, or is it inherently tied to test-time search?
- **Spec mapping:** `M9`
- **Source notebooks:** `notebooks/Hill_Climb_notebooks/slop_pipeline_colab.ipynb`, `notebooks/Hill_Climb_notebooks/slop_prelim_experiment (2).ipynb`, `notebooks/ai_slop_prompt_rewriter_colab_v3 (1).ipynb`
- **Method:** Build a `P4` hill-distillation pack from the best hill-climbed prompt outputs that beat the frozen baseline under Pangram. Fine-tune a rewrite model on those outputs and compare it to both the hill-climbed baseline and the best directly trained entrant.
- **Why it matters:** This is the cleanest bridge between the team’s earlier search-based work and the new model-tournament framing. If it works, it means search can be turned into a reusable model rather than a runtime procedure.
- **Interesting if false:** If distillation fails, then hill climbing may be finding brittle prompt-local tricks that do not transfer into stable rewrite behavior.
- **Tangible outputs:** One distilled checkpoint, a hill-distillation dataset, a comparison to the hill-climbed prompt baseline, and examples of search behaviors that do or do not survive compression.
- **Rough compute:** About 6-10 GPU-hours plus hill-pack construction; roughly $12-$20.

### 8. Scale Wildcard: QLoRA on a Larger Backbone

- **Question:** If we keep the task, data pack, and judge fixed, does a larger backbone beat better fine-tuning strategy on a smaller model?
- **Spec mapping:** `M10`
- **Source notebooks:** `notebooks/defan_slop_gan_unsloth (1).ipynb`, `notebooks/build_mirror_dataset (2).ipynb`, `notebooks/ai_slop_prompt_rewriter_colab_v3 (1).ipynb`
- **Method:** Train the Qwen wildcard on the same `P0` task using QLoRA, then compare it against the strongest `TinyLlama` entrant under the same Pangram evaluation and qualification gates.
- **Why it matters:** This is the scale test. If the larger backbone wins, scale may dominate technique. If it loses, then the research story gets sharper: training strategy and data design matter more than simply adding parameters.
- **Interesting if false:** A loss here would be especially interesting because it would show that a smaller, better-aligned model can beat a larger one on recognizable slop reduction.
- **Tangible outputs:** One QLoRA checkpoint, a scale-vs-technique comparison table, memory/runtime notes, and domain-specific win/loss cases versus the best `TinyLlama` model.
- **Rough compute:** About 8-14 GPU-hours; roughly $15-$28.

### 9. Tournament Recognition Audit

- **Question:** Do the winning checkpoints actually look less like AI writing across detectors, or are they mainly optimized to Pangram alone?
- **Spec mapping:** Finalist interpretation layer over the tournament winners
- **Source notebooks:** `notebooks/STAT4830_KaggleEssays_Verifier.ipynb`, `notebooks/stat4830_token_slop_classifier_v2_(1) (1).ipynb`, `notebooks/ai_slop_prompt_rewriter_colab_v3 (1).ipynb`, `notebooks/Hill_Climb_notebooks/slop_pipeline_colab.ipynb`
- **Method:** Run the top finalists through Pangram 3B, Pangram RoBERTa, the essay verifier, and the token classifier on the same held-out inputs. Compare whether the models agree that the rewrites are less recognizable as AI and inspect the disagreements by domain.
- **Why it matters:** This is the main recognition experiment in the slate. It tells us whether the tournament is teaching the model to become broadly less recognizable as slop or merely to satisfy one detector family.
- **Interesting if false:** If the detectors disagree sharply, that is still useful because it reveals that “AI slop” is not a single stable construct and that different recognizers are responding to different cues.
- **Tangible outputs:** A cross-detector audit table, detector-agreement plots, a disagreement subset for qualitative analysis, and a short summary of what each detector seems to reward or penalize.
- **Rough compute:** About 3-5 GPU-hours for batched scoring; roughly $5-$10.

### 10. Semantic Drift and Reward-Hacking Audit

- **Question:** Are the best deslopifiers becoming more human-like, or are they just learning to game the Pangram objective through shorter, flatter, or evasive rewrites?
- **Spec mapping:** Finalist interpretation layer over the tournament winners
- **Source notebooks:** `notebooks/ai_slop_prompt_rewriter_colab_v3 (1).ipynb`, `notebooks/Hill_Climb_notebooks/slop_pipeline_colab.ipynb`, `notebooks/STAT4830_KaggleEssays_Verifier.ipynb`
- **Method:** Compare finalists on Pangram delta, semantic similarity, output-input length ratio, refusal or degeneracy rate, lexical change, and a small qualitative sample of good and bad rewrites. Focus on cases where Pangram improvement is large but human judgment is ambiguous.
- **Why it matters:** This is the safeguard against fake wins. If the best model also passes this audit, then the tournament result is much more credible as a real slop-reduction finding.
- **Interesting if false:** If the “best” model turns out to be evasive or semantically hollow, that is still a valuable result because it reveals exactly how detector-driven deslopification breaks.
- **Tangible outputs:** A reward-hacking audit table, drift scatterplots, a compact failure taxonomy, and a final shortlist of which models improve style without sacrificing meaning.
- **Rough compute:** About 2-4 GPU-hours plus qualitative review time; roughly $5-$10.

## Priority Note

- **Core experiments:** 1-6. These are the main tournament-learning experiments and should be enough to support the core report claim.
- **Strong wildcards:** 7-8. These test whether search can be distilled and whether scale can beat better fine-tuning.
- **Evaluation and interpretation:** 9-10. These explain what the winners actually learned about slop recognition and whether the apparent gains are real.

If time collapses, the most defensible minimum slate is 1, 2, 5, 6, 9, and 10. That still yields a coherent story: parameterization, deeper adaptation, offline vs online Pangram alignment, and a serious audit of whether the winning model is actually less recognizable as AI slop.
