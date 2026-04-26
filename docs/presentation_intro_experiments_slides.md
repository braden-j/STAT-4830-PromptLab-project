# STAT 4830 Presentation — Intro and Experiments

This file is the slide-outline version of the talk. It is intentionally focused on:

- the problem framing
- the optimization view of the project
- the experiment logic
- the connection to the rest of the repo

It is not meant to cover the final results in detail, because those can hand off cleanly to later speakers.

## Slide 1 — Title and One-Sentence Framing

**Title**

Optimization Against "AI Slop": A Tournament of Deslopification Methods

**Core message**

- We treat AI slop reduction as an optimization problem: how do we transform text so it looks less detectably AI-generated while preserving meaning and fluency?
- The project compares multiple optimization strategies under one common evaluation harness instead of betting on a single method.

**What to show**

- Title
- Team / class / project name
- One clean visual line:
  `AI text -> rewrite / optimize -> detector score drops, meaning stays`

**Why this slide matters**

- It tells the audience immediately that this is not just a detector project and not just a generation project.
- It is an optimization project with a clear objective, constraints, and competing solution methods.

## Slide 2 — Why This Is an Optimization Problem

**Title**

The Project as an Optimization Problem

**Bullets**

- **Objective:** minimize AI-slop detectability.
- **Constraint 1:** preserve semantics.
- **Constraint 2:** preserve fluency and readable style.
- **Constraint 3:** avoid degenerate shortcuts like refusals, truncation, or token repetition.
- This creates a **multi-objective optimization** problem rather than a single-score maximization problem.

**Optimization formulation**

- Input: a suspiciously AI-like passage
- Decision variable: the rewrite policy
- Objective: lower the detector score
- Constraints / penalties: similarity, length discipline, validity, fluency
- Output: a rewrite that moves the text toward the human side without breaking it

**What to show**

- A simple 4-part diagram:
  - text in
  - optimization method
  - detector / reward
  - constrained output

**Why this slide matters**

- For an optimization class, this is the conceptual center of the project.
- It explains why we need ablations across methods, not just a single final model.

## Slide 3 — Project Lineage Inside the Repo

**Title**

How the Current Tournament Connects to the Rest of the Repo

**Bullets**

- The repo contains three related optimization stories:
- **Hill climbing:** optimize prompts against a learned slop verifier.
- **RL deslopifier:** optimize a rewrite model directly with REINFORCE using EditLens-based reward.
- **Tournament slate:** compare several training objectives and parameterizations under one shared benchmark.

**Connection to hill climbing**

- Hill climbing showed that prompt search can find lower-slop outputs without changing model weights.
- It established the idea that slop is something we can optimize against, not just classify.
- It also suggested a key architectural shift:
  - from prompt-to-prompt optimization
  - to direct essay-to-essay rewriting

**Connection to the RL deslopifier**

- The RL deslopifier reframed the task as policy optimization.
- It used Pangram EditLens as the reward and showed that detector scores can move substantially.
- It also exposed the central risk of this project: reward hacking.

**Why the tournament was necessary**

- Hill climbing was promising but indirect.
- REINFORCE was powerful but fragile.
- The tournament gives us a controlled way to compare:
  - cheap PEFT updates
  - deeper supervised updates
  - data scheduling
  - offline preferences
  - online reward optimization

**What to show**

- A 3-box pipeline:
  - `Verifier + prompt hill climbing`
  - `RL deslopifier`
  - `Tournament comparison harness`

## Slide 4 — Common Experimental Harness

**Title**

What Makes the Experiments Comparable

**Bullets**

- Same shared source pool
- Same base rewrite task
- Same Pangram family of judges
- Same split logic
- Same semantic and validity gates
- Same leaderboard logic

**Common scoring**

- **Fast scorer:** `pangram/editlens_roberta-large`
- **Official judge:** `pangram/editlens_Llama-3.2-3B`

**Why this matters**

- Without a common harness, differences in model score might come from:
  - different data
  - different prompt formats
  - different evaluation subsets
  - different decoding settings
- The tournament turns method comparison into a controlled optimization study instead of a collection of anecdotes.

**What to show**

- One table with 3 columns:
  - Fixed across all runs
  - Allowed to vary
  - Why that matters

Suggested entries:

- Fixed:
  - source pool
  - split
  - judge
  - eval rules
- Vary:
  - update scope
  - optimizer
  - data weighting
  - objective

## Slide 5 — Round 1: Parameterization and Objective Baselines

**Title**

Round 1 Experiments: How Should We Optimize the Rewrite Model?

**Bullets**

- **M1: Attention-only LoRA**
  - Smallest update space
  - Tests whether slop reduction is a shallow adaptation problem
- **M2: Full-module LoRA**
  - Broader low-rank update space
  - Tests whether we need more expressive PEFT to change writing style
- **M3: AdamW-based supervised entrant**
  - Stronger optimizer-centered baseline
  - Tests whether careful first-order optimization is enough without online reward learning
- **M4: REINFORCE**
  - Direct detector-reward optimization
  - Closest to the full deslopifier idea from the RL project summary

**Optimization interpretation**

- `M1` and `M2` ask about **search space size**
- `M3` asks about **optimization dynamics under supervised signals**
- `M4` asks about **direct policy optimization against the detector**

**What to show**

- A 4-row table:
  - Experiment
  - Optimization method
  - Main question
  - Main risk

Suggested risk column:

- `M1`: too weak
- `M2`: still superficial
- `M3`: may optimize the proxy but not the true objective
- `M4`: reward hacking / instability

## Slide 6 — Round 2: Data and Preference Optimization

**Title**

Round 2 Experiments: Better Data, Better Objective, or Both?

**Bullets**

- **M5: Curriculum fine-tuning**
  - Easy-to-hard ordering
  - Tests whether training trajectory matters
- **M6: Hard-negative fine-tuning**
  - Upweights the examples the model or judge finds hardest
  - Tests whether focusing on the decision boundary improves performance
- **M7: DPO / Pangram-ranked preferences**
  - Converts detector preference into offline pairwise supervision
  - Tests whether preference learning is more stable than online RL

**Optimization interpretation**

- `M5`: scheduling / curriculum optimization
- `M6`: importance weighting / hard-example mining
- `M7`: offline alignment through pairwise preference optimization

**Connection to the rest of the repo**

- `M5` and `M6` connect back to the classifier and hard-example ideas in the earlier hill-climb/verifier work.
- `M7` is the offline counterpart to the REINFORCE deslopifier:
  - same intuition
  - less online instability
  - more controlled optimization

**What to show**

- A 3-row table:
  - Experiment
  - What changes
  - Optimization intuition
  - Why it might beat round 1

## Slide 7 — What Each Experiment Teaches Us

**Title**

What We Learn If Each Experiment Wins

**Bullets**

- If **M1** wins:
  - slop is a shallow adaptation problem
  - cheap PEFT may be enough
- If **M2** wins:
  - broader adapter coverage matters
  - style shifts need a larger update subspace
- If **M3** wins:
  - stable supervised optimization beats more exotic methods
  - optimizer choice and regularization matter more than RL
- If **M4** wins:
  - direct reward optimization is genuinely useful
  - detector-guided RL can discover edits supervision misses
- If **M5** wins:
  - training order matters
  - optimization path matters, not just endpoint
- If **M6** wins:
  - the hard boundary cases contain most of the useful signal
- If **M7** wins:
  - offline preference optimization is the best balance of stability and alignment

**Why this slide matters**

- It reframes the experiments as scientific questions, not just leaderboard entries.
- It shows why even a "negative" result is still useful in an optimization class.

## Slide 8 — Handoff to the Results Section

**Title**

What Comes Next

**Bullets**

- Up to this point, we have defined:
  - the optimization problem
  - the common evaluation harness
  - the family of optimization methods being compared
- The next part of the presentation answers:
  - which methods actually improved the detector score
  - whether the gains were meaningful or just reward hacking
  - what the rewritten outputs look like in practice

**Bridge to later speakers**

- The next speaker can move naturally into:
  - quantitative results
  - qualitative examples
  - reward-hacking cases
  - limitations and future work

**Suggested transition line**

- "So the core setup is a controlled optimization tournament. The next question is which optimization strategy actually moved text out of the clearly-AI regime without breaking it."

## Optional Slide Design Notes

- Use one recurring phrase on several slides:
  - **objective, constraints, search space, optimization method**
- Keep notation light. This is an optimization class, but the audience still needs a story.
- Favor diagrams over dense tables whenever possible.
- If you include one equation, make it conceptual:

```text
maximize human-likeness
subject to meaning preserved, fluent output, no degenerate hacks
```

- If a later teammate is presenting the REINFORCE results in detail, keep this section focused on:
  - why REINFORCE is one entrant in the tournament
  - not the full training curve

