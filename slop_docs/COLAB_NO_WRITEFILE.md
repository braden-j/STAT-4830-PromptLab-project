# Colab: No %%writefile Patch Cells Required

The repository **already contains** the `slop.data` package under `slop_src/slop/data/`. After cloning, Colab does **not** need any `%%writefile` cells to create these files.

---

## 1. Exact files in the repo (already present)

| File | Purpose |
|------|--------|
| **slop_src/slop/data/__init__.py** | Exports `load_jsonl`, `SlopDataset`, `tokenize_and_align_labels`, `SlopTokenizer`, and token_labels helpers. |
| **slop_src/slop/data/dataset.py** | `load_jsonl(path)` and `SlopDataset` for token-level classification. |
| **slop_src/slop/data/tokenizer.py** | `tokenize_and_align_labels(examples, tokenizer, ...)` and `SlopTokenizer` wrapper. |
| **slop_src/slop/data/token_labels.py** | Helpers for span-based labeling (used by tests and optional pipelines). |

All paths use **slop_src/slop/** only. No references to `src/slop_minimization/`, `scripts/`, or `configs/`.

---

## 2. What was created / changed for this fix

- **Created (in repo):** `slop_src/slop/data/` with the four files above. No further edits were required; the package is already committed and tracked.
- **Changed:** Nothing in this pass. If your clone is missing `slop_src/slop/data/`, run `git pull` to get it.

---

## 3. Final contents (summary)

- **__init__.py** – Imports and re-exports from `.dataset`, `.tokenizer`, `.token_labels`.
- **dataset.py** – `load_jsonl(path)` reads JSONL into a list of dicts; `SlopDataset` wraps data + tokenizer for PyTorch.
- **tokenizer.py** – `tokenize_and_align_labels()` aligns word-level labels to subword tokens; `SlopTokenizer` wraps a HF tokenizer with `encode()` / `decode()` and `max_length`.
- **token_labels.py** – `detect_sloppy_spans`, `spans_to_token_labels`, `build_token_label_examples`, `DEFAULT_SLOP_PHRASES`.

No diffs needed; current repo state is the desired state.

---

## 4. Dataset schema assumptions

- **JSONL from `build_data.py`:** Each line is a JSON object with:
  - `"text"`: string (sentence or paragraph).
  - `"labels"`: list of int (0 = not slop, 1 = slop), one per word (word-level labels).
- **Config (slop_configs/classifier_encoder.yaml):** `data.train_path` / `data.val_path` point to these JSONL files; `data.text_column` = `"text"`, `data.label_column` = `"labels"`.
- **Slop pairs (for train_slop_generator):** JSONL with `"human"` and `"slop"` string fields per line.

---

## 5. Scripts that use slop.data

- **train_token_classifier.py** – `from slop.data.dataset import load_jsonl`; `from slop.data.tokenizer import tokenize_and_align_labels`. Runs from repo root with `PYTHONPATH=slop_src`.
- **train_slop_generator.py** – `from slop.data.dataset import load_jsonl`. Same.
- **optimize_prompts.py** – Does **not** import `slop.data`; uses `slop.scoring` and `slop.prompt_opt` only. No change needed.

---

## 6. Exact Colab commands that work (no writefile)

Set project root (after clone):

```bash
PROJECT_ROOT=/content/slop-repo
cd $PROJECT_ROOT
```

Then:

```bash
# 1. Build data
PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/build_data.py --output-dir data

# 2. Train classifier (saves to outputs/classifier_curriculum)
PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/train_token_classifier.py \
  --config slop_configs/classifier_encoder.yaml \
  --output-dir outputs/classifier_curriculum

# 3a. Generate slop pairs
PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/train_slop_generator.py generate \
  --input data/train.jsonl --output data/slop_pairs.jsonl

# 3b. Train T5 rewriter (saves to outputs/slop_rewriter)
PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/train_slop_generator.py train \
  --train-path data/slop_pairs.jsonl \
  --output-dir outputs/slop_rewriter \
  --model-name t5-small --epochs 3

# 4. Prompt optimization
PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/optimize_prompts.py \
  --config slop_configs/prompt_opt.yaml
```

Single-line form for Colab `!` cells:

```bash
!cd $PROJECT_ROOT && PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/build_data.py --output-dir data
!cd $PROJECT_ROOT && PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/train_token_classifier.py --config slop_configs/classifier_encoder.yaml --output-dir outputs/classifier_curriculum
!cd $PROJECT_ROOT && PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/train_slop_generator.py generate --input data/train.jsonl --output data/slop_pairs.jsonl
!cd $PROJECT_ROOT && PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/train_slop_generator.py train --train-path data/slop_pairs.jsonl --output-dir outputs/slop_rewriter --model-name t5-small --epochs 3
!cd $PROJECT_ROOT && PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/optimize_prompts.py --config slop_configs/prompt_opt.yaml
```

---

## 7. Removing notebook writefile patches

- Do **not** add cells that write `src/slop_minimization/data/__init__.py`, `dataset.py`, or `tokenizer.py`.
- Do **not** write to `scripts/` or `configs/`; use `slop_scripts/` and `slop_configs/` only.
- After `git clone` (and optional `git pull`), `slop_src/slop/data/` is present; set `PYTHONPATH=$PROJECT_ROOT/slop_src` and run the commands above. No file creation in the notebook is required.
