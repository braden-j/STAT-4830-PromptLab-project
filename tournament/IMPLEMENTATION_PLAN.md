# Tournament Phase 1 Implementation Plan

## Status

- `W&B` preflight: passed for entity `jgold23-university-of-pennsylvania-model-united-nations-` and project `PromptLab_STAT4830`.
- `Hugging Face` preflight: passed for Pangram datasets, Pangram scoring models, and the gated `meta-llama/Llama-3.2-3B` config files.
- Local caveat: this machine is nearly out of disk space, so downloading the full Meta weight shards locally is still risky even though access is now approved.
- Script implementation can proceed.

## Goal

Build an offline tournament workspace for the first five ranked experiment families from the slop slate, using Pangram as the common judge and a pilot-and-promotion bracket to stay within budget.

The phase-1 machinery must cover these entrants:

- `M1` attention-only LoRA
- `M2` full-module LoRA
- `M3` last-block unfreeze
- `M4` full fine-tune
- `M5` curriculum SFT
- `M6` hard-negative SFT
- `M7` DPO preference tuning

## Workspace Layout

```text
tournament/
  IMPLEMENTATION_PLAN.md
  README.md
  configs/
  scripts/
  data/
    raw/
      kaggle/
      defan/
  outputs/
  notebooks/
```

Reusable code now lives under `src/hill_climb/tournament/` and should remain the shared implementation layer for the new scripts.

## Auth Gate

All runtime auth must come from environment variables, never repo-tracked files:

- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`
- `WANDB_API_KEY`

The first script to implement after the HF token is fixed is `tournament/scripts/verify_access.py`. It must:

1. fail if HF or W&B env vars are missing
2. verify W&B login against the configured entity/project
3. verify HF identity with `HfApi().whoami()`
4. verify access to:
   - `meta-llama/Llama-3.2-3B`
   - `pangram/editlens_roberta-large`
   - `pangram/editlens_Llama-3.2-3B`
   - `pangram/editlens_iclr`
   - `pangram/editlens_iclr_grammarly`

## Data Contracts

### Source pool defaults

- Pangram target share: `80%`
- Kaggle essays target share: up to `15%`
- DefAn target share: up to `5%`
- If Kaggle or DefAn underfill after filtering, backfill with Pangram.

### Fixed paths

- Kaggle CSV default path: `tournament/data/raw/kaggle/aeon_essays.csv`
- DefAn raw download directory: `tournament/data/raw/defan/`

### Frozen phase-1 source split

- train: `3000`
- val: `500`
- test: `500`

### Output schemas

`source_pool.jsonl`

- `id`
- `split`
- `domain`
- `source_dataset`
- `text`
- `word_count`

`p0_pairs.jsonl`

- `example_id`
- `source_id`
- `split`
- `input_text`
- `target_text`
- `input_score_roberta`
- `target_score_roberta`
- `generation_mode`
- `length_ratio`

`p1_curriculum.jsonl`

- all `P0` fields
- `difficulty_bucket`

`p2_hard.jsonl`

- all `P0` fields
- `sample_weight`
- `hard_reason`

`p3_preferences.jsonl`

- `example_id`
- `source_id`
- `split`
- `input_text`
- `chosen_text`
- `rejected_text`
- `chosen_score_roberta`
- `rejected_score_roberta`
- `similarity_chosen`
- `similarity_rejected`

## Pair-Building Rules

### `P0`

- Build `input_text` by sloppifying clean targets.
- Guaranteed generation mode: existing rule-based sloppifier.
- Optional generation mode: prompt-based sloppification if it is available and stable.
- Keep the highest-scoring slop candidate if:
  - `input_score_roberta - target_score_roberta >= 0.15`
  - `length_ratio` in `[0.80, 1.30]`
- Otherwise keep the best candidate only if delta is at least `0.10`.
- Drop the sample if neither condition is met.

### `P1`

- `easy`: `target_score_roberta <= 0.10` and `input_score_roberta >= 0.80`
- `medium`: `input_score_roberta >= 0.60` and score delta `>= 0.20`
- `hard`: every other accepted sample

### `P2`

- `50%` top Pangram-input-score examples from `P0`
- `30%` lowest-delta examples from `P0`
- `20%` failures from the `M2` pilot eval outputs

### `P3`

- Generate `4` rewrite candidates per input.
- Rank with Pangram RoBERTa.
- Drop candidates with:
  - similarity `< 0.88`
  - length ratio outside `[0.80, 1.25]`
- Choose best-vs-worst among the remaining candidates.

## Training Interfaces

Planned entrypoints once auth is fixed:

- `tournament/scripts/train_sft.py`
- `tournament/scripts/train_dpo.py`
- `tournament/scripts/eval_checkpoint.py`
- `tournament/scripts/eval_leaderboard.py`

### Required config fields

- `experiment_id`
- `model_name`
- `training_mode`
- `data_pack_path`
- `eval_path`
- `max_seq_len`
- `token_budget`
- `batch_size`
- `gradient_accumulation_steps`
- `learning_rate`
- `num_epochs_max`
- `seed`
- `use_wandb`
- `wandb_entity`
- `wandb_project`

### Fixed model defaults

- Backbone for `M1-M7`: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `M1`: LoRA targets `q_proj,k_proj,v_proj,o_proj`
- `M2`: LoRA targets `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- `M3`: final 25% of blocks + final norm + LM head unfrozen
- `M4`: full-model fine-tune
- `M5`: same parameterization as winner of `M1` vs `M2`
- `M6`: same parameterization as `M5`
- `M7`: DPO on the `M5` backbone/parameterization

### Evaluation rules

- Reuse canonical Pangram normalization from `scripts/probe_editlens.py`
- Reuse fluency fallback logic from `scripts/fluency_reward_validation.py`
- Similarity model: `sentence-transformers/all-MiniLM-L6-v2`
- Qualification gates:
  - mean similarity `>= 0.88`
  - median length ratio in `[0.80, 1.25]`
  - invalid or degenerate outputs `< 2%`
- Official leaderboard metric:
  - `Delta_EditLens = EditLens(input) - EditLens(output)` using Pangram 3B
- Development metric:
  - Pangram RoBERTa delta

## Run Staging

1. Preflight
   - export env vars
   - place Kaggle CSV at the fixed path
   - run `verify_access.py`
   - stop immediately on any auth failure
2. Data build
   - build `source_pool`
   - build `P0`
   - build `P1`
   - build `P3`
   - delay `P2` until after `M2` pilot outputs exist
3. Smoke training
   - tiny `M2` smoke test on roughly `100` examples
4. Pilot bracket A
   - `M1` vs `M2`
   - anchor=`M2` only if it beats `M1` by at least `0.01` RoBERTa delta and passes gates
5. Pilot bracket B
   - `M3` and `M4`
   - promote `M3` if it beats anchor by at least `0.01`
   - promote `M4` only if it beats anchor by at least `0.02`
6. Pilot bracket C
   - build `P2`
   - run `M5`, `M6`, `M7`
   - promote at most the top `2` models that beat anchor by at least `0.01`
7. Full runs
   - always full-run anchor
   - full-run `M3` only if promoted
   - full-run up to `2` of `M5/M6/M7`
   - full-run `M4` only if promoted and budget allows
   - Pangram 3B eval only on frozen baseline, anchor, and promoted finalists

## Tracking

Once auth is fixed, every run should log to W&B:

- full config
- git SHA
- dataset manifest hash
- training loss
- validation loss
- Pangram RoBERTa delta
- similarity
- length ratio
- invalid rate
- grad norm
- learning rate
- token throughput
- peak GPU memory

Artifacts to save:

- `source_pool`
- `P0`, `P1`, `P2`, `P3`
- best checkpoints
- `leaderboard.csv`
- `finalist_samples.jsonl`

## Current Blocker

Auth is no longer the blocker. The main practical risk on this workstation is disk pressure for large model downloads, so full-weight local evals should be avoided when a Colab or remote GPU can handle them.
