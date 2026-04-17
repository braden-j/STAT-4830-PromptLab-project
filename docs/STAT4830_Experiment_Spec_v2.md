# STAT 4830 — Transformer Deslopifier Tournament
# Detailed Experiment Specification (v2.1 — transformer fine-tuning tournament revision)

**Date:** April 9, 2026  
**Final Presentation:** ~April 17, 2026  
**Compute:** H100 via Prime Intellect  
**Primary reward and judge:** Pangram EditLens (`pangram/editlens_Llama-3.2-3B`)  
**Fast development scorer:** `pangram/editlens_roberta-large`

---

## 0. What Changed in This Revision

The old spec mixed together reward setup, classifier hardening, RL ideas, and broad exploratory experiments. This revision changes the emphasis completely:

- The project is now a **controlled model tournament**.
- Every real experiment must output a **trained model checkpoint**.
- The main question is no longer "can we build a reward?" but rather:

> Which transformer fine-tuning strategy produces the best deslopifier under the same Pangram reward and the same evaluation harness?

- Hill climbing is still important, but it is now used in two narrower ways:
  - as a **baseline to beat**
  - as a **data engine** for one distillation experiment

This makes the final presentation cleaner. Instead of many loosely related claims, the report can tell one story:

1. We fixed a common rewrite task.
2. We trained multiple transformer variants under comparable conditions.
3. Pangram judged all of them.
4. We ran a tournament and got a winner.

---

## 1. North Star

Train a transformer that rewrites AI-slop text into lower-Pangram-score text while preserving meaning, approximate length, and fluency.

The final deliverable is not just a notebook result. It is a **leaderboard of trained models**, each produced by a different fine-tuning strategy, with one winner chosen under a fixed blind evaluation set.

---

## 2. Common Tournament Harness

### 2.1 Common Task

Each tournament model solves the same task:

- **Input:** a sloppy or AI-looking passage
- **Output:** a rewrite of that same passage
- **Goal:** reduce Pangram/EditLens score as much as possible without changing the meaning too much

This is a **text-to-text transformer fine-tuning problem**. The tournament is about how we fine-tune the transformer, not about changing the task between models.

### 2.2 Canonical Source Pool

Lock one shared source pool and derive every tournament dataset from it.

- **Essay source texts:** Kaggle essays and essay-style material from the token-classifier and verifier notebooks
- **Prompt-response source texts:** HH-RLHF and Dolly-style instruction data from the prompt-risk and prompt-rewriter notebooks
- **QA source texts:** DefAn-style definitive-answer examples from the GAN notebook

Use one fixed split for the whole tournament:

- **Train source pool:** 9,000 source texts
- **Validation source pool:** 1,500 source texts
- **Test source pool:** 1,500 source texts

Target balance:

- 1/3 essay-like longform
- 1/3 prompt-response explanation text
- 1/3 QA / definitive-answer text

Important constraint: every tournament entrant should train from text derived from this same source pool. Experiments may transform or filter the data differently, but they should not swap to a totally different corpus.

### 2.3 Canonical Data Packs

All experiments must use one of the following data packs, each derived from the same source pool.

| Pack | Description | Used For |
|---|---|---|
| `P0 Base Pair Pack` | `(slop_text -> cleaner reference text)` pairs from mirror prompting, paired human text, and existing rewrite assets | SFT entrants |
| `P1 Curriculum Pack` | Same pairs as `P0`, bucketed by Pangram score gap or rewrite difficulty: easy -> medium -> hard | Curriculum entrant |
| `P2 Hard Pack` | Same source texts, but weighted toward the highest-Pangram or most failure-prone examples | Hard-negative entrant |
| `P3 Preference Pack` | Same source texts, but each input has 3-5 candidate rewrites ranked by Pangram delta under a similarity filter | DPO entrant |
| `P4 Hill Pack` | Same prompts/source texts, but targets come from hill-climbed prompt outputs that beat the frozen baseline | Distillation entrant |

The comparability rule is simple:

- Experiments can change **how** the data is weighted, ordered, or labeled.
- Experiments should not change **what universe of source texts** they are drawn from.

### 2.4 Fixed Backbone Rule

To keep the tournament apples-to-apples, use one primary backbone family for most entrants:

- **Primary backbone:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

This choice is justified because:

- it already appears in the hill-climbing pipeline
- it is cheap enough to train multiple times in a week
- it is large enough that LoRA vs partial unfreeze vs full fine-tune is a meaningful comparison

One wildcard entrant is allowed to use a different backbone:

- **Wildcard backbone:** `Qwen/Qwen2.5-1.5B-Instruct` with QLoRA

That entrant is explicitly testing scale and architecture transfer, not strict same-backbone comparability.

### 2.5 Fixed Decode and Eval Rule

All tournament models must be evaluated the same way.

- Use the same effective training token budget for all `TinyLlama` entrants
- Decode with deterministic or near-deterministic settings
- Use the same maximum output length cap
- Use the same held-out input set
- Use the same Pangram scorer for official ranking

Recommended evaluation decode:

- `temperature = 0.0`
- `top_p = 1.0`
- `max_new_tokens` matched to the task cap
- no custom prompt hacks at evaluation time

The tournament is about model weights. Prompt-only tricks are not allowed to decide the winner.

For comparability:

- `M1-M9` should use the same source split, same max sequence length, and same total number of seen training tokens
- `M10` may use a different memory recipe because it is the scale wildcard, but it should still train on the same source pool and same objective family as `M1-M4`

### 2.6 Pangram Reward and Qualification Gates

All entrants are ranked primarily by Pangram.

### Primary tournament metric

For each held-out input:

`Delta_EditLens = EditLens(input) - EditLens(output)`

Higher is better.

Official ranking uses:

- **Primary judge:** `pangram/editlens_Llama-3.2-3B`

Cheap inner-loop filtering can use:

- `pangram/editlens_roberta-large`

But checkpoint selection for the leaderboard and all final reporting should use the 3B Pangram model on the same validation split.

### Qualification gates

A model only qualifies for the final leaderboard if it also passes:

- **Semantic similarity:** mean similarity >= 0.88 on the held-out set
- **Length discipline:** median output/input length ratio in `[0.80, 1.25]`
- **Validity:** fewer than 2% degenerate outputs

Among models that pass those gates, the winner is the model with the highest mean `Delta_EditLens`.

### Tie-breakers

If two models are close, break ties in this order:

1. lower mean final Pangram score
2. higher semantic similarity
3. lower variance across domains

---

## 3. Reference Baselines (Not Eligible to Win)

These are important comparison points, but they are not tournament entrants because they are not newly fine-tuned model checkpoints.

### B0. Frozen generator baseline

- Frozen `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Original prompt / no rewrite model

### B1. Hill-climbed prompt baseline

- Frozen TinyLlama
- Best prompt found by the existing hill-climbing pipeline

### B2. Existing prompt-rewriter baseline

- Current FLAN-T5 prompt-rewriter notebook setup, if available

These three baselines define the floor and the search-time bar. The tournament winner should ideally beat B1, because B1 is the strongest existing non-gradient method in the repo.

---

## 4. Tournament Entrants at a Glance

| ID | Model Family | Data Pack | Objective | Main Difference |
|---|---|---|---|---|
| `M1` | TinyLlama | `P0` | SFT | Attention-only LoRA |
| `M2` | TinyLlama | `P0` | SFT | Broader LoRA over attention + MLP |
| `M3` | TinyLlama | `P0` | SFT | Last-block full-weight unfreeze |
| `M4` | TinyLlama | `P0` | SFT | Full-model fine-tune |
| `M5` | TinyLlama | `P1` | SFT | Curriculum fine-tuning |
| `M6` | TinyLlama | `P2` | SFT | Hard-negative fine-tuning |
| `M7` | TinyLlama | `P3` | DPO | Preference tuning with Pangram-ranked pairs |
| `M8` | TinyLlama | online from source pool | GRPO | Direct Pangram-reward optimization |
| `M9` | TinyLlama | `P4` | SFT | Hill-climb distillation |
| `M10` | Qwen 1.5B | `P0` | QLoRA SFT | Scale and architecture wildcard |

The intended logic is:

- `M1-M4` isolate **which parameters should be trainable**
- `M5-M6` isolate **what to do with the data**
- `M7-M8` isolate **which training objective aligns best with Pangram**
- `M9` tests whether **search can be compressed into weights**
- `M10` tests whether **scale beats technique**

---

## 5. Detailed Experiments

### M1. Attention-Only LoRA Control

- **Question:** How far can we get by training only attention adapters on the shared rewrite task?
- **Backbone:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Trainable parameters:** LoRA on attention projections only; base model frozen; LM head saved if needed
- **Training data:** `P0 Base Pair Pack`
- **Objective:** standard supervised fine-tuning on `(slop_text -> cleaner reference text)` pairs
- **Source notebooks:** `notebooks/Hill_Climb_notebooks/slop_pipeline_colab.ipynb`, `notebooks/ai_slop_prompt_rewriter_colab_v3 (1).ipynb`, `notebooks/build_mirror_dataset (2).ipynb`
- **Why it matters:** This is the cheapest serious transformer fine-tune and should be the control entrant for the entire tournament.
- **Interesting if false:** If this already wins or nearly wins, the project learns that aggressive unfreezing is unnecessary and parameter-efficient fine-tuning is enough.
- **Outputs:** one checkpoint, validation curves, held-out rewrite samples, Pangram delta table
- **Rough compute:** low to medium

### M2. Full-Module LoRA

- **Question:** Does LoRA become much stronger when adapters cover both attention and MLP blocks instead of attention only?
- **Backbone:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Trainable parameters:** LoRA on attention plus MLP projections; base frozen
- **Training data:** `P0 Base Pair Pack`
- **Objective:** same SFT objective as `M1`
- **Source notebooks:** `notebooks/defan_slop_gan_unsloth (1).ipynb`, `notebooks/ai_slop_prompt_rewriter_colab_v3 (1).ipynb`
- **Why it matters:** This cleanly tests whether the bottleneck is adapter coverage rather than the choice between PEFT and full fine-tuning.
- **Interesting if false:** If broader LoRA barely helps, then either the task is easy enough for cheap adapters or the model needs true weight updates to improve.
- **Outputs:** checkpoint, adapter comparison plot versus `M1`, Pangram delta table
- **Rough compute:** low to medium

### M3. Last-Block Unfreeze

- **Question:** Is it enough to fully train only the final quarter of transformer blocks and the LM head?
- **Backbone:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Trainable parameters:** last 25% of transformer layers, final norms, and LM head; earlier blocks frozen
- **Training data:** `P0 Base Pair Pack`
- **Objective:** SFT
- **Source notebooks:** `notebooks/ai_slop_prompt_rewriter_colab_v3 (1).ipynb`, `notebooks/Hill_Climb_notebooks/slop_pipeline_colab.ipynb`
- **Why it matters:** This is the cleanest middle ground between LoRA and full fine-tuning.
- **Interesting if false:** If this underperforms LoRA, then broad low-rank adaptation may be more stable than partial full-weight updates.
- **Outputs:** checkpoint, layer-unfreeze ablation summary, Pangram and similarity metrics
- **Rough compute:** medium

### M4. Full-Model Fine-Tune

- **Question:** What is the same-backbone performance ceiling if we train all TinyLlama weights on the rewrite task?
- **Backbone:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Trainable parameters:** all model weights
- **Training data:** `P0 Base Pair Pack`
- **Objective:** SFT
- **Source notebooks:** `notebooks/ai_slop_prompt_rewriter_colab_v3 (1).ipynb`, `notebooks/defan_slop_gan_unsloth (1).ipynb`
- **Why it matters:** This gives the reference ceiling for the primary backbone and tells us whether PEFT is leaving real performance on the table.
- **Interesting if false:** If full fine-tuning does not beat `M2` or `M3`, that is a strong practical result because it says full unfreezing is not worth the cost or instability.
- **Outputs:** full checkpoint, overfitting curves, direct comparison to `M1-M3`
- **Rough compute:** high

### M5. Curriculum LoRA

- **Question:** Does training order matter if we present easy rewrites before hard ones?
- **Backbone:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Trainable parameters:** same adapter scope as `M2`
- **Training data:** `P1 Curriculum Pack`
- **Objective:** SFT
- **Source notebooks:** `notebooks/stat4830_token_slop_classifier_v2_(1) (1).ipynb`, `notebooks/build_mirror_dataset (2).ipynb`, `notebooks/ai_slop_prompt_rewriter_colab_v3 (1).ipynb`
- **Why it matters:** The repo already leans on curriculum ideas for classification. This asks whether the same training-order logic helps generation-side deslopification.
- **Interesting if false:** If curriculum does nothing, the project learns that rewrite quality depends more on data type and objective than easy-to-hard scheduling.
- **Outputs:** checkpoint, per-stage curves, easy/medium/hard validation breakdown
- **Rough compute:** medium

### M6. Hard-Negative LoRA

- **Question:** Is it better to spend training budget on the hardest, most obviously AI-looking or most failure-prone examples?
- **Backbone:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Trainable parameters:** same adapter scope as `M2`
- **Training data:** `P2 Hard Pack`
- **Objective:** SFT
- **Source notebooks:** `notebooks/stat4830_token_slop_classifier_v2_(1) (1).ipynb`, `notebooks/build_mirror_dataset (2).ipynb`, `notebooks/Hill_Climb_notebooks/slop_pipeline_colab.ipynb`
- **Why it matters:** This connects the Pangram hard-example intuition to the generator side of the project.
- **Interesting if false:** If the hard pack underperforms the base pair pack, then hard-negative mining may be more useful for detectors than for rewrite models.
- **Outputs:** checkpoint, hard-vs-uniform data comparison, failure-case gallery
- **Rough compute:** medium

### M7. Preference-Tuned Model (DPO)

- **Question:** Is pairwise preference tuning on Pangram-ranked candidate rewrites better aligned than plain SFT?
- **Backbone:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Trainable parameters:** same adapter scope as `M2`
- **Training data:** `P3 Preference Pack`
- **Objective:** DPO on best-vs-worst rewrite pairs, where ranking is induced by Pangram delta subject to similarity thresholds
- **Source notebooks:** `notebooks/ai_slop_prompt_rewriter_colab_v3 (1).ipynb`, `notebooks/build_mirror_dataset (2).ipynb`, `notebooks/Hill_Climb_notebooks/slop_pipeline_colab.ipynb`
- **Why it matters:** This is the strongest offline alignment experiment because it uses Pangram directly without the instability of online RL.
- **Interesting if false:** If DPO cannot beat SFT, then Pangram-induced preference pairs may be too noisy or too shallow for ranking-based optimization.
- **Outputs:** checkpoint, preference-pair bank, DPO-vs-SFT comparison on the same validation set
- **Rough compute:** medium

### M8. Direct Pangram-Reward RL (GRPO)

- **Question:** Does online reward optimization with Pangram actually outperform the best offline method?
- **Backbone:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Trainable parameters:** start from the best checkpoint among `M2`, `M5`, and `M7`; train with GRPO
- **Training data:** same canonical source inputs, but rewards are computed online from sampled rewrites
- **Objective:** maximize Pangram delta with KL, fluency, and length penalties
- **Source notebooks:** `notebooks/defan_slop_gan_unsloth (1).ipynb`, `notebooks/Hill_Climb_notebooks/slop_pipeline_colab.ipynb`
- **Why it matters:** This is the most direct optimization experiment in the entire tournament because the training signal is the actual Pangram reward.
- **Interesting if false:** If GRPO underperforms DPO or SFT, that is still a strong result because it shows that online reward optimization is too noisy or too hackable for this task.
- **Outputs:** checkpoint, reward curves, KL drift plot, RL-vs-offline comparison
- **Rough compute:** high

### M9. Hill-Climb Distilled Model

- **Question:** Can search-time improvements from hill climbing be compressed into a standalone rewrite model?
- **Backbone:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Trainable parameters:** same adapter scope as `M2`
- **Training data:** `P4 Hill Pack`
- **Objective:** SFT on targets produced by hill-climbed prompt outputs that beat the frozen baseline under Pangram
- **Source notebooks:** `notebooks/Hill_Climb_notebooks/slop_pipeline_colab.ipynb`, `notebooks/Hill_Climb_notebooks/slop_prelim_experiment (2).ipynb`, `notebooks/ai_slop_prompt_rewriter_colab_v3 (1).ipynb`
- **Why it matters:** This is the cleanest bridge between the existing hill-climbing work and the new fine-tuning tournament.
- **Interesting if false:** If distillation fails, then hill-climbing gains may be prompt-local search artifacts that do not compress well into weights.
- **Outputs:** checkpoint, hill-distillation dataset, comparison to B1 hill-climbed prompt baseline
- **Rough compute:** medium

### M10. QLoRA Scale Wildcard

- **Question:** If we keep the data and objective fixed but scale the backbone, does a bigger model beat clever fine-tuning on TinyLlama?
- **Backbone:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Trainable parameters:** 4-bit QLoRA using Unsloth-style training
- **Training data:** `P0 Base Pair Pack`
- **Objective:** SFT
- **Source notebooks:** `notebooks/defan_slop_gan_unsloth (1).ipynb`, `notebooks/build_mirror_dataset (2).ipynb`
- **Why it matters:** This gives the tournament one explicit scale wildcard. If it wins, scale matters. If it loses, training strategy matters more than naive parameter count.
- **Interesting if false:** A loss here would be very interesting because it would show that smaller, better-adapted models can beat a larger backbone under the same Pangram judge.
- **Outputs:** checkpoint, scale-vs-technique comparison, memory/runtime summary
- **Rough compute:** medium to high

---

## 6. Tournament Flow

Run the tournament in stages so compute is spent where it matters.

### Stage 1: Parameter Bracket

Train and compare:

- `M1`
- `M2`
- `M3`
- `M4`

Goal: determine whether LoRA, partial unfreezing, or full fine-tuning is the best basic update rule.

### Stage 2: Data Bracket

Using the best parameterization from Stage 1, train:

- `M5`
- `M6`

Goal: determine whether curriculum or hard-example emphasis beats the plain data mix.

### Stage 3: Objective Bracket

Train:

- `M7`
- `M8`

Goal: compare offline preference tuning against direct Pangram-reward optimization.

### Stage 4: Search Compression and Scale Wildcards

Train:

- `M9`
- `M10`

Goal: test the two most interesting ways to beat the main bracket: distill search, or scale the model.

### Stage 5: Finals

Send the top 4 qualifying models to a blind held-out evaluation:

- same test inputs
- same deterministic decoding
- official Pangram 3B judge
- same qualification gates

The best final-score model is the tournament winner.

---

## 7. Intended Reporting Story

The final report should be able to say something this concrete:

> We ran a controlled tournament of transformer deslopifiers. All models used the same Pangram reward, the same held-out evaluation set, and nearly the same source pool. The winner was the model that reduced Pangram score the most while preserving meaning.

That is much stronger than a vague collection of unrelated experiments.

This setup also gives several meaningful failure stories:

- LoRA may beat full fine-tuning
- curriculum may not help generation even if it helps classification
- DPO may beat RL
- hill climbing may produce gains that are hard to distill
- a larger wildcard model may still lose to a smaller but better-adapted one

Any of those outcomes would still be interesting.

---

## 8. Priority and Time Discipline

If time gets tight, keep the tournament spine intact and cut from the edges.

### Must run

- `M1`
- `M2`
- `M3`
- `M7`
- `M8`
- `M9`

This set is enough to tell the core story: basic PEFT comparison, partial unfreezing comparison, preference vs RL, and hill-climb distillation.

### Nice to run

- `M4`
- `M5`
- `M6`

These deepen the story but are not required for the tournament to make sense.

### Wildcard

- `M10`

This is valuable if compute allows, but the tournament is still coherent without it.

---

## 9. Source Notebook Mapping

This revision is still grounded in the current repo.

- **Prompt rewriting / text-to-text supervision:** `notebooks/ai_slop_prompt_rewriter_colab_v3 (1).ipynb`
- **Mirror data construction:** `notebooks/build_mirror_dataset (2).ipynb`
- **Unsloth / QLoRA training patterns:** `notebooks/defan_slop_gan_unsloth (1).ipynb`
- **Hill climbing and reward-driven prompt search:** `notebooks/Hill_Climb_notebooks/slop_pipeline_colab.ipynb`
- **Token-level curriculum and hard-example intuition:** `notebooks/stat4830_token_slop_classifier_v2_(1) (1).ipynb`
- **Essay verifier and human-vs-slop assets:** `notebooks/STAT4830_KaggleEssays_Verifier.ipynb`

The important change is not the raw materials. It is the framing: these notebooks now feed one controlled tournament instead of a loose experiment list.
