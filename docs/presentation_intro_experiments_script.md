# STAT 4830 Presentation — Intro and Experiments Script

This is a speaker script for the intro and experiment-design section of the talk.
It is written to connect smoothly to later presenters who will cover results,
qualitative examples, and limitations.

The tone is intentionally optimized for an optimization or ML systems audience:
it emphasizes objective functions, constraints, comparability, and why the
ablation structure matters.

## Slide 1 — Title and One-Sentence Framing

**Suggested time:** 30-45 seconds

**Script**

"Our project is about optimizing against what we’re calling AI slop: text that is fluent enough to read, but still carries a recognizable AI signature. Rather than treating this only as a detection problem, we treat it as a constrained optimization problem. The goal is to rewrite text so it becomes less detectably AI-generated, while still preserving meaning, fluency, and overall readability. This presentation section focuses on how we designed that optimization problem and how we structured the experiments."

## Slide 2 — Why This Is an Optimization Problem

**Suggested time:** 60-75 seconds

**Script**

"The key reason this belongs in an optimization class is that the task is not just classification and it is not just generation. We have an explicit objective and several competing constraints. The objective is to minimize detectable AI-ness, using Pangram’s EditLens family as the main judge. But if we only optimize that one number, we get bad behavior very quickly. The model can shorten outputs too aggressively, drift semantically, or find degenerate shortcuts. So the actual problem is multi-objective: reduce detector score while preserving semantics, preserving fluency, and avoiding invalid outputs. That makes this a useful case study in constrained optimization under imperfect reward signals."

## Slide 3 — Project Lineage Inside the Repo

**Suggested time:** 75-90 seconds

**Script**

"The current tournament did not appear from nowhere. It sits on top of two earlier optimization stories that are already in the repo. The first is the hill-climbing pipeline. There, the team trained a verifier for slop and then optimized prompts against that verifier, using a frozen generator. That work showed two important things: first, slop can be optimized against; and second, prompt search is a real baseline, not just a toy. The second story is the RL deslopifier. That project used REINFORCE with EditLens reward to optimize a rewrite model directly. It showed that detector scores can be moved significantly, but it also exposed the risk of reward hacking. The tournament is our attempt to unify those threads. Instead of asking whether one method works in isolation, we compare several optimization strategies under one shared harness."

## Slide 4 — Common Experimental Harness

**Suggested time:** 60-75 seconds

**Script**

"To make the comparison meaningful, we had to make the experiments comparable. So we fix the shared source pool, the rewrite task, the evaluation rules, and the Pangram judging family. We use Pangram RoBERTa as the fast development scorer and Pangram 3B as the official judge. We also keep semantic similarity, length discipline, and validity checks as qualification gates. That matters because otherwise each method could look good for a different reason. One model might look better because it saw easier data; another because it used a different evaluation subset; another because it simply shortened everything. The common harness is what turns this from a collection of model runs into a controlled optimization tournament."

## Slide 5 — Round 1: Parameterization and Objective Baselines

**Suggested time:** 90-105 seconds

**Script**

"Round 1 asks the most basic optimization question: what kind of parameter update and objective do we actually need? `M1` is attention-only LoRA, which is our smallest and cheapest update space. It tests whether slop reduction is basically a shallow adaptation problem. `M2` expands LoRA to a broader module set, so it asks whether broader low-rank movement is needed to change writing style in a meaningful way. `M3` is a stronger supervised entrant built around standard first-order optimization, specifically AdamW. That ties back to the hill-climbing and T5-style work in the repo, where careful supervised training and direct rewrite supervision were already strong ideas. Then `M4` is REINFORCE, which connects directly to the deslopifier summary. That is the most aggressive optimization method in this round because it optimizes against the detector reward itself, not just labeled targets. So conceptually, round 1 compares search-space size, optimizer behavior, and direct policy optimization."

## Slide 6 — Round 2: Data and Preference Optimization

**Suggested time:** 90-105 seconds

**Script**

"Once round 1 gives us competitive baselines, round 2 shifts from parameterization toward data and objective design. `M5` uses curriculum fine-tuning, meaning we vary the training trajectory from easier to harder examples. That is an optimization question about path dependence: does the order of examples help the model land in a better part of parameter space? `M6` uses hard-negative fine-tuning, which means we overweight the examples that the model or judge finds hardest. That is essentially importance weighting near the decision boundary. `M7` is DPO, where we generate multiple candidate rewrites, rank them with Pangram under similarity constraints, and learn from those pairwise preferences. This is important because it gives us an offline alternative to REINFORCE. So if REINFORCE is the high-variance online optimization method, DPO is the more stable offline preference-learning alternative."

## Slide 7 — What Each Experiment Teaches Us

**Suggested time:** 75-90 seconds

**Script**

"A useful way to read this tournament is not just in terms of winners, but in terms of what each winner would mean. If attention-only LoRA wins, then slop reduction may be a shallow adaptation problem. If broader LoRA wins, then the model likely needs a larger effective search space to change its style. If AdamW wins, then stable supervised optimization may be enough, and we may not need RL to get the main effect. If REINFORCE wins, then direct reward optimization is telling us something real that supervision is missing. If curriculum wins, that says the optimization path matters. If hard negatives win, it says the boundary cases matter most. And if DPO wins, that suggests offline preference optimization may be the best tradeoff between stability and alignment. So even before we show results, the experiment design already gives us a scientifically meaningful map of the space."

## Slide 8 — Handoff to the Results Section

**Suggested time:** 30-45 seconds

**Script**

"So the big picture is that we built a controlled optimization tournament for deslopification. We start with a shared objective, shared constraints, and a shared evaluation harness, then compare multiple ways of searching over model behavior: PEFT, supervised optimization, curriculum, hard-example weighting, offline preference learning, and direct reward optimization. The next question is which of those methods actually moved outputs out of the clearly-AI regime, and whether those gains were genuine improvements or just another form of reward hacking. That’s what the next part of the presentation covers."

## Optional Shorter Version

If time gets compressed, use this condensed structure:

### Condensed message for Slide 3

"The repo evolved from prompt optimization, to RL deslopification, to a controlled tournament. That progression mirrors a shift from indirect optimization, to direct reward optimization, to systematic comparison."

### Condensed message for Slide 5

"Round 1 compares parameterization and objective: small PEFT, broader PEFT, stable supervised optimization, and REINFORCE."

### Condensed message for Slide 6

"Round 2 compares data and objective engineering: curriculum, hard negatives, and offline preference learning."

### Condensed transition to results

"So the experimental question is not just which model wins, but which optimization idea wins."

## Notes for the Presenter

- Keep saying "optimization problem" explicitly. That is the anchor for this audience.
- Do not get lost in implementation trivia here. Save detailed numbers and qualitative examples for the next speakers.
- When discussing REINFORCE, frame it as one method in a family, not the entire project.
- When discussing hill climbing, emphasize that it established the feasibility of optimizing against a slop signal, but that the tournament moves the project from prompt search into model-comparison science.

