# Slop minimization — machine learning pipeline

This document is the technical README for the **slop minimization** codebase (`slop_src`, `slop_scripts`, `slop_configs`, `slop_docs`). The repository’s root [`README.md`](README.md) is the STAT 4830 course template (schedule, grading, git workflow). Quick links to shared course assets: [STAT 4830 course materials](#stat-4830-course-materials) below.

---

# Project overview

This project trains a **token-level binary classifier** to distinguish human-quality text from “slop” (generic, low-quality phrasing), then uses that model as a **reward signal** to **optimize natural-language prompts** for a **fixed (frozen) generative LLM**. The goal is to steer model outputs toward higher predicted quality without fine-tuning the generator’s weights.

**Problem (plain language):** AI-written text often reads repetitive or hollow. We need a way to (1) **score** text for slop-like patterns and (2) **improve what we ask** the model so its answers look less sloppy under that scorer.

**Outcome:** A reproducible pipeline: build labeled data → train a DistilBERT + LoRA verifier → optionally train a T5 “sloppifier” for data augmentation → run **hill-climbing prompt search** with a frozen instruction-tuned LM (e.g. TinyLlama); evaluate with scripts that report mean reward, sequence-level proxies, and optional reward-model statistics on held-out JSONL.

---

# Approach & methodology

**Paradigm:** Primarily **supervised learning** (token classification with cross-entropy) plus **black-box optimization** over prompts (evolutionary / hill-climbing search using a learned scalar reward—not gradient-based RL on the frozen LM).

**Conceptual workflow:**

1. **Data:** Lines of “good” vs “slop” text → JSONL with word-aligned labels (0 = good, 1 = slop).
2. **Verifier:** Encoder (DistilBERT) + linear head + optional **LoRA** on attention projections; trained with **curriculum sampling** over difficulty strata when metadata exists.
3. **Augmentation (optional):** Rule-based `RuleSloppifier` builds `(human, slop)` pairs; **T5-small** can be fine-tuned to imitate that mapping for richer negatives.
4. **Prompt optimization:** Structured **PromptSpec** objects are mutated, rendered to strings, fed to a **frozen** causal LM; generated text is scored by the verifier and auxiliary **structural / semantic / quality** terms from `slop/scoring/diagnostics.py`. Search keeps high-reward prompt variants.

**Design choices:**

- **Token labels, not sentence-only:** Matches surface slop that is word-local; document score aggregates token probabilities (mean by default).
- **LoRA + frozen base:** Few trainable parameters, stable training on small data.
- **Classifier as reward:** \(R \approx 1 - \mathbb{E}[\text{slop probability}]\) so higher reward means “less slop” (see `compute_reward` in [`slop_src/slop/scoring/aggregation.py`](slop_src/slop/scoring/aggregation.py)).
- **Discrete prompt search** instead of a continuous “prompt embedding” policy: practical for instruction-following LMs and interpretable prompts.

---

# Data

**Sources (as implemented):**

- Default ** [`slop_scripts/build_data.py`](slop_scripts/build_data.py)** reads `data/raw/good.txt` and `data/raw/slop.txt` (one sentence/paragraph per line). If both are missing, it writes **placeholder** good/slop examples so the skeleton runs.
- Each JSONL line: `{"text": "...", "labels": [0|1, ...]}` with **one label per whitespace token** (see `create_token_labels`).
- **Splits:** 80% train / 10% val / 10% test (configurable via `--train-ratio`, `--val-ratio`). Outputs: `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`.

**Optional curriculum:** [`slop_configs/classifier_encoder.yaml`](slop_configs/classifier_encoder.yaml) enables `curriculum_enabled` with a `difficulty` column (easy/medium/hard) and epoch-dependent sampling weights (`curriculum_early_*`, `curriculum_late_*`). Data must include that field per example when used.

**Slop pairs for T5:** [`slop_scripts/train_slop_generator.py`](slop_scripts/train_slop_generator.py) `generate` loads “good” text from `data/train.jsonl` and/or `data/raw/good.txt`, applies [`RuleSloppifier`](slop_src/slop/slop_gen/rule_sloppifier.py), writes `data/slop_pairs.jsonl` with `human` / `slop` fields.

**Challenges / limitations:**

- **Placeholder data** is tiny and not representative; real runs need curated good/slop corpora.
- **Word-token alignment** is simplistic (`text.split()`); multilingual or complex tokenization should align labels with the tokenizer.
- **slop_pairs** quality depends on rules; T5 training inherits that distribution.

---

# Model architecture

## Token classifier (verifier)

- **Backbone:** `distilbert-base-uncased` (encoder), max sequence length **256** ([`classifier_encoder.yaml`](slop_configs/classifier_encoder.yaml)).
- **Head:** Linear layer to **2 classes** (good vs slop); probabilities converted to per-token slop probability.
- **PEFT:** LoRA `r=16`, `alpha=32`, dropout `0.05`, targets **`q_lin`, `k_lin`, `v_lin`**; base weights frozen except LoRA + classifier (via [`create_classifier_and_tokenizer`](slop_src/slop/models/classifier_factory.py)).
- **Loss:** Cross-entropy on token labels (`ignore_index=-100` for padding); see [`EncoderSlopClassifier.forward`](slop_src/slop/models/token_classifier.py).
- **Training loop:** [`slop_scripts/train_token_classifier.py`](slop_scripts/train_token_classifier.py) — AdamW, LR `2e-5`, weight decay `0.01`, gradient clipping `1.0`, **FP16** on CUDA, early stopping on validation **doc-level AUROC** (best checkpoint saved to `output_dir`).

**Checkpoints:** `pytorch_model.bin` (state dict), tokenizer files, and **`model_config.json`** (architecture + LoRA metadata for [`eval.py`](slop_scripts/eval.py) loading).

## T5 slop rewriter (optional)

- **Model:** `t5-small` default ([`train_rewriter`](slop_src/slop/slop_gen/train_rewriter.py)): seq2seq, source = human, target = slop; HF `Trainer` with MSE-style seq2seq loss.

## Frozen generator (prompt optimization)

- **Model:** Configurable causal LM (default **TinyLlama/TinyLlama-1.1B-Chat-v1.0** in [`prompt_opt.yaml`](slop_configs/prompt_opt.yaml)); weights are **not** updated.

## Prompt search

- **Population-based hill climbing** with mutation operators on **PromptSpec** ([`evolve.py`](slop_src/slop/prompt_opt/evolve.py), [`mutations.py`](slop_src/slop/prompt_opt/mutations.py), [`templates.py`](slop_src/slop/prompt_opt/templates.py)).
- **Reward:** Combines verifier-based score with structural penalties (e.g. abnormal punctuation density), semantic penalties (meta-instructions, off-task), and optional quality bonuses — weights in YAML `search:` block.

---

# Technical implementation

**Layout:**

| Path | Role |
|------|------|
| [`slop_src/slop/`](slop_src/slop/) | Package: `models/`, `scoring/`, `prompt_opt/`, `slop_gen/`, `data/`, `config.py`, `dataset_io.py`, `tokenizer_utils.py`, `metrics.py` |
| [`slop_scripts/`](slop_scripts/) | CLI: `build_data`, `train_token_classifier`, `train_slop_generator`, `optimize_prompts`, `eval`, `eval_reward_model`, `score_reward`, comparison utilities |
| [`slop_configs/`](slop_configs/) | YAML: `classifier_encoder.yaml`, `prompt_opt.yaml`, `reward.yaml`, etc. |
| [`slop_docs/`](slop_docs/) | [`COLAB_CELLS.md`](slop_docs/COLAB_CELLS.md) (Colab runbook), [`CURRENT_PHASE.md`](slop_docs/CURRENT_PHASE.md) (overview + next steps) |
| [`slop_tests/`](slop_tests/) | pytest: tokenizer, scoring, token labels ([`conftest.py`](conftest.py) adds `slop_src` to path; offline tokenizer fallback) |
| [`docs/`](docs/), [`notebooks/`](notebooks/) | STAT 4830 assignments, figures, course notebooks |

**Dependencies:** Declared in [`pyproject.toml`](pyproject.toml): `torch`, `transformers`, `peft`, `datasets`, `accelerate`, `tqdm`, `pyyaml`, `einops`, `scikit-learn`; optional `unsloth`, `wandb`, dev `pytest`.

**Patterns:**

- Scripts prepend `slop_src` to `sys.path` for imports.
- **SlopRewardModel** ([`scoring/reward.py`](slop_src/slop/scoring/reward.py)) wraps the same checkpoint for batch scoring used by optimization and eval scripts.

---

# Results & performance

**Evaluation (implemented):**

- **`slop_scripts/eval.py`:** Loads classifier via `model_config.json` + `pytorch_model.bin`, runs on `data/test.jsonl`, prints **`mean_reward`** (higher = less slop), **`sequence_accuracy`** (document-level proxy vs majority label), **`n_samples`**; can write `outputs/eval_results.json`.
- **`slop_scripts/eval_reward_model.py`:** Dataset-level mean/std of rewards and optional top/bottom examples.
- **Training metrics:** During training, validation reports token F1, token AUROC, doc AUROC ([`train_token_classifier.py`](slop_scripts/train_token_classifier.py) + [`metrics.py`](slop_src/slop/metrics.py)).

**Reporting:** This repo does **not** commit fixed benchmark numbers or plots for the slop pipeline; numbers depend on your data and runs. The course [`report.md`](report.md) describes related **prompt-level regression** experiments (TF-IDF + linear/MLP on HH-RLHF-style targets) with example metrics (e.g. Spearman ~0.2 range) — a **different** formulation from the token classifier + prompt-opt stack. Treat [`notebooks/`](notebooks/) and `docs/figures/` as optional EDA for course work.

**Limitations:** Reward hacking against structural/semantic heuristics; verifier calibration on out-of-domain text; compute for larger generators; Colab session persistence (artifacts under `outputs/`, gitignored).

---

# Setup & usage

**Python:** `>=3.11` per [`pyproject.toml`](pyproject.toml).

**Install (local):**

```bash
cd /path/to/repo
# uv or pip from pyproject
uv sync   # or: pip install -e ".[dev]"
```

**Always set `PYTHONPATH`** when running scripts without editable install:

```bash
export PYTHONPATH=$PWD/slop_src
```

**End-to-end (typical):**

```bash
python slop_scripts/build_data.py --output-dir data
python slop_scripts/train_token_classifier.py \
  --config slop_configs/classifier_encoder.yaml \
  --output-dir outputs/classifier_curriculum
python slop_scripts/train_slop_generator.py generate \
  --input data/train.jsonl --output data/slop_pairs.jsonl
python slop_scripts/train_slop_generator.py train \
  --train-path data/slop_pairs.jsonl \
  --output-dir outputs/slop_rewriter \
  --model-name t5-small --epochs 3
python slop_scripts/optimize_prompts.py --config slop_configs/prompt_opt.yaml
python slop_scripts/eval.py \
  --classifier-path outputs/classifier_curriculum \
  --test-path data/test.jsonl \
  --output-path outputs/eval_results.json
python slop_scripts/eval_reward_model.py \
  --data data/test.jsonl \
  --checkpoint outputs/classifier_curriculum \
  --show-examples 3
```

**Colab:** Follow [`slop_docs/COLAB_CELLS.md`](slop_docs/COLAB_CELLS.md) (clone, install, cells in order, optional “Show progress” section before zipping artifacts).

**Tests:**

```bash
pytest slop_tests -v
```

---

# STAT 4830 course materials

This repository is the **STAT 4830 PromptLab** team workspace. Course logistics, weekly deliverables, OODA cycle, and project examples remain available here:

- **Getting started & ideas:** [`docs/finding_project_ideas.md`](docs/finding_project_ideas.md)
- **Assignments:** [`docs/assignments/`](docs/assignments/) (e.g. week 4 instructions, self-critiques)
- **Figures:** [`docs/figures/`](docs/figures/), [`figures/noam.png`](figures/noam.png)
- **Notebooks:** [`notebooks/`](notebooks/) (e.g. verifier / slop experiments)
- **Report draft:** [`report.md`](report.md)

For the mathematical overview of the classifier + prompt-optimization story and a granular directory map, see [`slop_docs/CURRENT_PHASE.md`](slop_docs/CURRENT_PHASE.md).
