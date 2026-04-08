# Slop Minimization: Classifier + Prompt Optimization

*Current-phase doc for the slop pipeline: high-level formulation, granular repo guide, and next steps. For the Colab runbook, see [COLAB_CELLS.md](COLAB_CELLS.md).*

This repo implements a pipeline to **detect** low-quality (“slop”) text with a learned classifier and **reduce** slop by optimizing prompts for a frozen LLM using that classifier as a reward signal. The document below is in two parts: a high-level overview (like the slides, but more verbose) and a granular, project-specific guide to directories, models, and scripts.

---

## Part 1 — High-level: What the project does

### The problem

As model-generated text floods the web, “slop”—generic, low-quality, or obviously AI-written content—becomes a problem for search, ranking, and trust. We want to:

1. **Recognize** slop: score or classify text as human-quality vs slop.
2. **Reduce** slop: influence what an LLM outputs by choosing better prompts, so that the same model produces less sloppy text when given those prompts.

We do **not** fine-tune the LLM’s weights in this pipeline. We train a **verifier** (classifier) on text, then use it as a **reward model** and optimize **prompts** so that the frozen LLM’s outputs get higher reward (i.e., look less like slop under the verifier).

---

### Mathematical formulation (two stages)

**Stage 1 — Train a binary verifier (classifier)**  
We fit a model \( f_\phi \) so that \( f_\phi(x) = P(\text{human} \mid x) \): the probability that a piece of text \( x \) is human-quality (non-slop). We treat this as binary classification:

- **Input:** Text \( x \) (the “generated output” in the slide: we train on labeled human vs slop sentences).
- **Label:** \( y \in \{0,1\} \): \( 1 = \) human-quality / non-slop, \( 0 = \) slop / generic.
- **Objective:** Minimize cross-entropy (binary classification loss):
  \[
  \min_\phi \;\mathbb{E}_{(x,y)} \bigl[ y \log f_\phi(x) + (1-y) \log(1 - f_\phi(x)) \bigr].
  \]

This gives a **calibrated** predictor: confident wrong predictions are penalized heavily, and the resulting scores can be used later as a reward. In this project, the classifier is **token-level** (per-word labels) with a document-level score used as the reward.

**Stage 2 — Optimize prompts to maximize reward**  
With the verifier fixed, we then maximize the expected reward of the **output** of a frozen LLM when we feed it prompts we control:

- **Goal:** \( \max_{\pi} \;\mathbb{E}_{p \sim \pi}\bigl[ R(G(p)) \bigr] \), where  
  - \( p \) = prompt (sampled from some search distribution),  
  - \( G \) = **frozen** LLM (generator),  
  - \( G(p) \) = generated text,  
  - \( R \) = **learned verifier** (our classifier), used as the reward.

So: **Prompt → LLM → Output → Score with R.** We search over prompts (e.g., by hill climbing) to get outputs that score higher under \( R \) (i.e., less slop).

In the slide, “\( \pi_\theta \)” is a prompt distribution; in this codebase we use **discrete search** (hill climbing over a population of prompt strings) rather than a parameterized \( \pi_\theta \), but the high-level idea is the same: optimize what we feed the LLM so that \( R(G(p)) \) is higher.

---

### Pipeline in plain language

1. **Data:** Build train/val/test sets of text with labels (human vs slop). Optionally use curriculum (easy/medium/hard) for training.
2. **Classifier:** Train a token-level classifier (e.g., DistilBERT + LoRA) with cross-entropy. Save it; this is your **verifier** \( R \).
3. **T5 rewriter (optional):** Train a small T5 to turn “good” sentences into “slop” versions; used to generate more slop training pairs. Not required for the core “classifier + prompt optimization” loop.
4. **Prompt optimization:** With a **frozen** generator \( G \) (e.g., TinyLlama) and the **frozen** classifier \( R \), run hill climbing over prompts: mutate prompts, generate with \( G \), score with \( R \), keep high-reward prompts and iterate. Outputs are optimized prompts and run summaries.
5. **Evaluate:** Report classifier metrics (mean reward, sequence accuracy on test set), reward-model stats, and a short summary of the latest prompt-optimization run.

End-to-end: **Data → Train verifier \( R \) → (Optional: T5 slop rewriter) → Optimize prompts for frozen \( G \) using \( R \) → Quantify progress.**

---

## Part 2 — Granular: This repo’s layout and how to run it

### Repository layout (main pieces)

- **`src/hill_climb/`** — Python package: models, data loading, tokenization, scoring, prompt optimization.
- **`scripts/hill-climb_scripts/`** — CLI entry points: build data, train classifier, train T5, run prompt optimization, eval, comparisons.
- **`configs/hill-climb_configs/`** — YAML configs for classifier, prompt optimization, reward, and small/default runs.
- **`docs/hill-climb_docs/`** — Docs; **`COLAB_CELLS.md`** is the canonical “run in Colab” cell-by-cell guide.
- **`tests/hill-climb-tests/`** — Tests for tokenization, scoring, etc.
- **`data/`** — Built datasets (train/val/test JSONL); produced by `build_data.py` (inputs live under `data/raw/` if you use your own good/slop files).
- **`outputs/`** — All trained artifacts (classifier, T5 rewriter, prompt-opt runs). Gitignored; persist via zip or Google Drive.

The **canonical pipeline** described here is **`src/hill_climb` + `scripts/hill-climb_scripts/` + `configs/hill-climb_configs/`**, as in **`COLAB_CELLS.md`**.

---

### Directories and roles

| Path | Role |
|------|------|
| `src/hill_climb/` | Core library: classifier models, data, tokenizer utils, scoring (reward), prompt_opt (templates, mutations, generator, hill climbing). |
| `src/hill_climb/models/` | `token_classifier.py` (EncoderSlopClassifier, SlopTokenClassifier), `classifier_factory.py` (create_classifier_and_tokenizer), `slop_generator.py` (T5 rewriter). |
| `src/hill_climb/scoring/` | Reward model wrapper (`reward.py`), aggregation, diagnostics. |
| `src/hill_climb/prompt_opt/` | PromptSpec, render modes, mutations, `FrozenGenerator`, `run_hill_climbing`, `compare_seed_vs_optimized`. |
| `src/hill_climb/data/` | Dataset and token-label utilities. |
| `scripts/hill-climb_scripts/` | All runnable scripts; see table below. |
| `configs/hill-climb_configs/` | `classifier_encoder.yaml`, `prompt_opt.yaml`, `reward.yaml`, etc. |
| `data/` | `train.jsonl`, `val.jsonl`, `test.jsonl` (and optionally `slop_pairs.jsonl` for T5). |
| `outputs/` | `classifier_curriculum/`, `slop_rewriter/`, `prompt_opt/`, `eval_results.json`. |

---

### Key scripts (what runs what)

| Script | Purpose |
|--------|--------|
| `build_data.py` | Build train/val/test JSONL from good/slop text (or placeholders). Writes to `data/`. |
| `train_token_classifier.py` | Train the verifier (encoder + LoRA). Reads `configs/hill-climb_configs/classifier_encoder.yaml`; writes `outputs/classifier_curriculum` (or path you set). Saves `pytorch_model.bin`, tokenizer, and `model_config.json`. |
| `train_slop_generator.py` | Generate slop pairs and/or train T5 rewriter. Writes `data/slop_pairs.jsonl`, `outputs/slop_rewriter`. |
| `optimize_prompts.py` | Hill-climb prompts using frozen generator + reward model. Reads `configs/hill-climb_configs/prompt_opt.yaml`; writes to `outputs/prompt_opt/`. |
| `eval.py` | Evaluate classifier on test set (mean reward, sequence accuracy). Writes `outputs/eval_results.json`. Loads classifier via `model_config.json` (or backward-compat default). |
| `eval_reward_model.py` | Score a JSONL with the reward model; report mean/std and optional top/bottom examples. |
| `score_reward.py` | CLI: score raw text or a file with the reward model. |
| `eval_prompts.py` | Compare seed vs optimized prompts (mean reward, etc.) for a run. |
| `review_latest_run.py` | Print top prompts and diagnostics for the latest prompt-opt run. |
| `compare_*.py` | Compare generators, structure styles, rendering modes, or reward checkpoints. |
| `validate_dataset.py`, `build_classifier_dataset.py` | Data validation and alternate dataset building. |

All scripts assume you run from repo root with `PYTHONPATH` including the repo’s `src` directory (e.g. `PYTHONPATH=$PWD/src python scripts/hill-climb_scripts/script_name.py ...`).

---

### Configs (what to tweak)

- **`configs/hill-climb_configs/classifier_encoder.yaml`** — Backbone (e.g. DistilBERT), LoRA, max_length, training (batch size, epochs, early stopping), data paths, curriculum. Override output with `--output-dir`.
- **`configs/hill-climb_configs/prompt_opt.yaml`** — Reward checkpoint and aggregation; generator model (e.g. TinyLlama), temperature, max_new_tokens; search (population_size, num_iterations, mutation, structural/semantic/quality weights); default_task and output_dir. Used by `optimize_prompts.py`.
- **`configs/hill-climb_configs/reward.yaml`** — Reward-model-only options if you run scoring scripts with a shared config.

---

### Models and artifacts

- **Classifier (verifier)** — Encoder (e.g. DistilBERT) + LoRA + token classification head. Trained with cross-entropy on (text, human/slop) labels. Saved under `outputs/classifier_curriculum/`: `pytorch_model.bin`, tokenizer files, `model_config.json`. Used as \( R \) in prompt optimization and in `eval.py` / `eval_reward_model.py` / `score_reward.py`.
- **T5 slop rewriter** — Optional; generates (human → slop) pairs for data augmentation. Saved under `outputs/slop_rewriter/`.
- **Prompt-optimization runs** — Under `outputs/prompt_opt/`: per-run dirs (e.g. `run_*`), `best_prompts.json`, diagnostics. No separate “model”; the “output” is the set of optimized prompts and metadata.

---

### How to run: Colab vs local

- **Colab (recommended for full pipeline):** Follow **`COLAB_CELLS.md`** step by step. Clone repo, set `PROJECT_ROOT`, install deps, then run cells in order: GPU check → clone/setup → install → layout → build data → train classifier → generate slop pairs → train T5 → prompt optimization → “Show progress” cells (eval, reward-model stats, prompt-opt summary) → zip (and optionally Drive). Each cell has a short comment at the top describing what it does.
- **Local:** Same order, but run the commands from the cells in your shell (using `cd $PROJECT_ROOT && PYTHONPATH=$PROJECT_ROOT/src python scripts/hill-climb_scripts/...`). Ensure `data/` exists (run `build_data.py` or drop in your own train/val/test JSONL).
- **Quick validation after code changes:** Clone (or pull), install deps, restore artifacts from a previous run (e.g. unzip `slop_critical_artifacts.zip` into `outputs/` or copy from Drive). Then run only the “Show progress” and eval cells so you don’t retrain everything.

---

### Outputs and persistence

- **`outputs/`** is gitignored. To keep artifacts across Colab sessions: use **Cell 10** (zip) and download the zip, and/or the **Optional** cell to copy to Google Drive. To reload later: upload the zip and unzip into `outputs/`, or mount Drive and copy from `MyDrive/slop_pipeline` into `outputs/`.
- Eval and reward scripts write to paths like `outputs/eval_results.json`; prompt-opt writes under `outputs/prompt_opt/`. The “Show progress” section in `COLAB_CELLS.md` summarizes how to read these and where you are in the pipeline.

---

### Dependencies (high level)

- PyTorch, transformers, datasets, peft, accelerate, pyyaml, tqdm, scikit-learn, sentencepiece. Install with the commands in `COLAB_CELLS.md` (Cell 3) or your local env.

---

## Next steps

Planned or natural extensions of the current pipeline:

**Search and prompt optimization (more parameters)**  
- **Hill-climbing:** Larger population, more iterations, and more samples per prompt; tune mutation strength, exploration rate, and top-k selection. Config knobs live in `configs/hill-climb_configs/prompt_opt.yaml` under `search`.
- **Prompt mutations:** Richer mutation operators (structural, semantic, length, format) and higher semantic-mutation probability to escape local optima. The codebase already supports structural vs semantic penalties and quality-reward terms; these can be expanded or reweighted.
- **Rendering and structure:** Compare `render_mode` and structure preferences (e.g. prose vs list-friendly) systematically; use the existing `compare_structure_styles.py` and `compare_rendering_modes.py` as a base.

**Verifier and data**  
- Stronger or better-calibrated classifier: more/better human–slop data, curriculum refinements, or alternate backbones. Validate the slop proxy against human preferences (e.g. chosen/rejected) where available.
- T5 slop rewriter: improve rewriter quality or data mix so the classifier sees more diverse slop.

**Generators and reward**  
- Try other frozen generators (see `prompt_opt.yaml` comments: gpt2, Qwen2.5-0.5B, phi-2, SmolLM, etc.) and compare reward distributions with `compare_generators.py`.
- Reward shaping: adjust structural, semantic, and quality weights so high reward better matches “good” text without reward hacking.

**Evaluation and deployment**  
- Human eval or A/B tests on optimized vs seed prompts to confirm that higher reward corresponds to better perceived quality.
- Scale to larger data or longer runs; monitor Colab memory and runtime.

**Related work: GAN-style slop generator (peer)**  
A parallel direction in this project is an **AI Slop GAN-style pipeline** (Colab, GPU): load **DefAn** (definitive-answer QA), create slop answers with a **mixture of older open-weights Hugging Face models** (tunable), train a **Discriminator** to classify real vs slop, and train a **Generator** with **Unsloth** (QLoRA, 4-bit) on real answers then **adversarially nudge** it to produce plausible-but-wrong answers. That setup is complementary to this repo’s “frozen generator + prompt optimization” flow: one optimizes prompts for a fixed LM; the other trains a generator to produce slop. Integration (e.g. shared discriminator/verifier, or slop data from the GAN pipeline) is left as a future step.
