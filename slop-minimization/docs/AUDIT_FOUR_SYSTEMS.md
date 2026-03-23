# Audit: Four New Systems (Quality Reward, Semantic Mutations, Exploration, Eval Cache)

**Date:** 2026-03-16  
**Scope:** Verify that the four systems are not just present but meaningfully active. No new features; inspection only.

---

## 1. Short Audit Summary

| System | Present | Meaningfully active? | Notes |
|--------|--------|----------------------|--------|
| **Quality reward** | Yes | Yes | Contribution is modest (~0.065 on avg_quality ~0.65); same order as semantic penalty. lambda_quality=0.10 is in a sensible range. |
| **Semantic mutations** | Yes | Yes (when triggered) | All four helpers fire and change PromptSpec fields in a meaningfully semantic way. Not logged in run artifacts; evidence from code + simulated examples. |
| **Exploration** | Yes | **Partially** | Random immigrants and epsilon exploration are in the loop and affect candidate generation, but **provenance is broken**: every candidate is labeled "carryover", so we cannot measure elite vs mutation vs exploration from artifacts. |
| **Eval cache** | Yes | Yes | Cache key includes rendered prompt + all eval/generator knobs. Hits/misses only in memory and script stdout; not persisted to run artifacts. No correctness risk if key is exact. |

**Main concerns:** (1) Provenance does not distinguish elite / mutation / exploration. (2) Cache stats and run-level quality/cache summary are not written to run artifacts. (3) Semantic mutation type is not logged, so we cannot count which helpers fire in practice without code changes.

---

## 2. Reward Decomposition Findings

**Data source:** Existing runs only (tiny runs: 1 iteration, population 2). So we have **2 unique prompts** across `run_20260316_133400` and `run_20260316_133548` leaderboards. A full 5-prompt audit would require a successful multi-candidate run (e.g. 2–3 iterations, population 6+, top_k 3+).

### Observed magnitudes (2 prompts)

| Prompt (abbrev) | avg_base_reward | structural_pen_contrib | semantic_pen_contrib | avg_quality_score | quality_reward_contrib | avg_reward |
|-----------------|-----------------|-------------------------|------------------------|--------------------|------------------------|------------|
| Run 133548 seed | -0.4390 | 0.0000 | 0.0491 | 0.6565 | 0.0657 | -0.4224 |
| Run 133400 seed | -0.4230 | 0.0000 | 0.0489 | 0.6412 | 0.0641 | -0.4078 |

**Interpretation:**

- **Quality contribution:** ~0.064–0.066 (lambda_quality=0.10 × avg_quality_score ~0.64–0.66). So quality is **modest but meaningful**: it’s on the same order as the semantic penalty contribution (~0.049) and adds a real upward nudge to the final reward.
- **Negligible?** No — it’s roughly 15% of |avg_base_reward| and similar in magnitude to the semantic penalty.
- **Too strong?** No — it doesn’t dominate; base reward and penalties still drive most of the spread.
- **lambda_quality=0.10:** In a **sensible range** relative to observed base reward (~-0.42 to -0.44) and penalties (~0.05 each). A 0.05–0.15 range remains reasonable for ablations.

**Diagnosis:** Quality reward is meaningfully active and lambda_quality=0.10 is reasonable. Recommend repeating with 5+ prompts when a larger run is available to confirm consistency.

---

## 3. Semantic Mutation Findings

**Evidence:** Code inspection + simulated parent→child examples from `scripts/audit_semantic_mutations.py` (no model load). Run artifacts do **not** record which mutation type (semantic vs slot vs structural) produced each candidate.

### Helpers and behavior

All four semantic helpers are implemented and, when invoked, produce **meaningful, non-cosmetic** changes:

1. **mutate_constraints_semantically**  
   - **Fired:** Yes (simulated).  
   - **Fields changed:** `constraints`.  
   - **Example:** "Be concise." → "Use only the detail needed to fully answer the task."  
   - **Assessment:** Semantic — shifts from vague to task-focused constraint.

2. **mutate_anti_slop_semantically**  
   - **Fired:** Yes (simulated).  
   - **Fields changed:** `anti_slop`.  
   - **Example:** "Be precise and avoid generic filler." → "Avoid talking about how to write; focus on answering the task itself."  
   - **Assessment:** Semantic — reduces meta/writing-advice in favor of on-task content.

3. **mutate_output_format_semantically**  
   - **Fired:** Yes (simulated).  
   - **Fields changed:** `output_format`.  
   - **Example:** "Plain paragraphs." → "Use 1–3 short paragraphs with one concrete example; avoid list-heavy formatting."  
   - **Assessment:** Semantic — more concrete format guidance aligned with prose preference.

4. **mutate_reasoning_style_semantically**  
   - **Fired:** Yes (simulated).  
   - **Fields changed:** `reasoning_style`.  
   - **Example:** "State claims directly; support with concrete examples when needed." → "Focus on concrete explanation rather than abstract writing advice."  
   - **Assessment:** Semantic — reframes from structure to content quality.

**When they fire:** With `semantic_mutation_probability=0.25`, about 25% of mutations are semantic; the rest are structural or slot. So all four helpers can fire in practice; we do not have per-helper counts from real runs because mutation type is not logged.

**Gap:** No artifact records "this candidate came from a semantic mutation" or "this candidate came from mutate_constraints_semantically". To measure which helpers fire how often, add optional logging of mutation type (e.g. `"semantic"` and helper name) when building the population and store it in the evaluation result / generations.

---

## 4. Exploration Findings

**Logic (from `evolve.py`):**

- **Elite carryover:** `top_k` specs (and optionally one unmutated best) are kept.
- **Normal mutation:** For each of `top_k`, we add `children_per_parent - 1` children via `mutate_spec(..., semantic_mutation_probability=sem_p)`; with probability `exploration_epsilon`, the parent is a random member of `top_k` instead of the current spec.
- **Random immigrants:** `n_immigrants = round(population_size * exploration_rate)` (or `num_random_immigrants` if set). Each immigrant is either a random seed copy or a mutated parent from `top_specs`/seeds.
- **Legacy:** `keep_random_explore` random mutants from seeds are also added.

So exploration **is** affecting candidate generation (immigrants and epsilon parent choice). However:

**Provenance is too coarse to measure it:**

- At the start of each iteration we set `provenance_by_id = {id(s): "carryover" for s in population}`.
- So **every** member of the current population is labeled `"carryover"`. We never assign `"elite"`, `"mutation"`, or `"exploration"` when building the population.
- Result: In artifacts, every candidate has `provenance: "carryover"`. We cannot answer:
  - How many candidates came from elite carryover vs normal mutation vs exploration/immigrants?
  - Whether epsilon exploration ever triggered?
  - Whether any exploration candidate survived into top_k?

**Minimal code change to make this measurable:**

- When building the next `population`, tag each spec with a provenance label at append time, e.g.:
  - `"elite"` for the one unmutated best and the copies of top_specs.
  - `"mutation"` for children from `mutate_spec` (including when parent was chosen via epsilon).
  - `"exploration"` for `keep_random_explore` mutants and for each of the `n_immigrants`.
- Store these in a list or dict keyed by `id(spec)` (or by a stable spec fingerprint if ids change), and set `res["provenance"] = provenance_by_id.get(id(spec), "unknown")` when evaluating. Then persist `provenance` in leaderboard and generations as already done.

---

## 5. Cache Findings

**Mechanics:**

- **Key:** `(prompt_text, samples_per_prompt, min_output_length, render_mode, lambda_structural, structural_threshold, lambda_semantic, task_keywords tuple, lambda_quality, quality_weights tuple, generator model_name, temperature, top_p, max_new_tokens, repetition_penalty, no_repeat_ngram_size)`.
- **Scope:** In-memory, per run only. No disk persistence.
- **When:** If `enable_eval_cache` is true and key is in cache → return cached result and increment `cache_hits`; else run `evaluate_prompt`, store result, increment `cache_misses`.

**Observed stats:** From the conversation summary, a tiny run reported 1 cache hit and 1 cache miss. So hits occur when the same prompt is evaluated again (e.g. same spec surviving in population or re-evaluated in a later iteration).

**Hit rate:** Depends on run. With many unique prompts and few iterations, hit rate can be low. When the same prompt appears multiple times (e.g. elite carryover, or duplicate specs in population), hit rate increases. So the cache is **meaningfully active** when there are repeated evaluations of the same (rendered prompt + settings).

**What is cached:** Full result dict from `evaluate_prompt` (prompt_spec, prompt_text, outputs, valid_outputs, all reward components, diagnostics). So repeated evaluations of the same prompt with same config and generator settings return the same result.

**Does the cache suppress useful stochastic reevaluation?**

- Yes, by design: the same prompt with the same generator/eval settings returns the same cached outputs and reward. So we do **not** get multiple stochastic samples for the same prompt within a run.
- That is consistent with the current design: cache key includes generator knobs (temperature, etc.), so different sampling would require a different key and would be a cache miss. So the only “suppression” is that we do not intentionally rerun the same (prompt, settings) to get a second stochastic outcome — which is a deliberate choice to make rewards comparable and to save cost.
- If you wanted to experiment with stochastic reevaluation (e.g. multiple samples per prompt across iterations), you could either disable the cache for that run or add a “sample id” or “run id” to the cache key so that repeated evaluations are not cached.

**Correctness:**

- Cache is **valid** only within-run and for **identical** (rendered prompt text + eval knobs + generator knobs). So no cross-run reuse and no reuse when any of those change.
- **Subtle risk:** If `evaluate_prompt` or the reward model ever depended on global state (e.g. random seed beyond the passed `rng`, or external API), cached results could be wrong. From the code, evaluation uses the passed `rng` and the provided generator/reward_model; the cache key includes the generator config. So no correctness risk identified for the current implementation.

**Recommendation:** Persist `eval_cache_hits` and `eval_cache_misses` in run artifacts (e.g. `report.md` or a small `stats.json`) so that cache behavior can be audited without re-running.

---

## 6. Top 3 Next Experiments (Ablation Suggestions)

Do not implement yet; use these to decide what to run next.

1. **lambda_quality ablation**  
   - **Question:** How does final reward and prompt quality change when lambda_quality is 0, 0.05, 0.10, 0.15, 0.20?  
   - **What it answers:** Whether 0.10 is a good trade-off and whether quality reward improves optimization or mostly adds noise.

2. **semantic_mutation_probability ablation**  
   - **Question:** With semantic_mutation_probability in {0, 0.15, 0.25, 0.40}, does diversity or leaderboard quality (e.g. top-5 avg reward, or human preference) improve?  
   - **What it answers:** Whether semantic mutations help search and what probability is sufficient.

3. **exploration_rate / enable_eval_cache comparison**  
   - **Question:** (a) exploration_rate 0 vs 0.15 vs 0.25 — do exploration candidates ever enter top_k and does convergence change? (b) enable_eval_cache true vs false — how much does wall-clock time change and does reward distribution change (e.g. fewer reevaluations might reduce variance)?  
   - **What it answers:** Whether exploration is necessary for good results and whether caching affects optimization quality or only speed.

---

## 7. Concerns and Implementation Weak Points

1. **Provenance:** Everyone is labeled "carryover"; elite vs mutation vs exploration cannot be measured. Fix: tag provenance when building the population (see §4).
2. **Artifacts:** Cache hits/misses and a brief quality/cache summary are not written to the run directory; they only appear in script stdout and return value. Fix: write them to `report.md` or `stats.json`.
3. **Semantic mutation logging:** Mutation type (and optionally which semantic helper) is not stored, so we cannot compute per-helper fire rates from runs. Fix: optional `mutation_type` / `mutation_helper` on the evaluation result when the spec was produced by a known mutation path.
4. **Reward decomposition sample size:** Audit used 2 prompts from tiny runs. For a full 5-prompt reward decomposition and more reliable lambda_quality diagnosis, run at least one successful optimization with 2–3 iterations and population ≥ 6, then re-run the same reward decomposition table on the leaderboard.
5. **Completeness_score:** In the observed runs, `completeness_score` is 0 in quality_diagnostics (e.g. no “conclusion” or similar signal). If that heuristic is always zero, one-third of the quality weight has no effect; worth checking the completeness heuristic and whether it’s tuned for the task.

---

*End of audit.*
