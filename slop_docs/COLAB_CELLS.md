# Colab cells: essential pipeline (functional, bug-free)

Clone the repo, set `PYTHONPATH` to `slop_src`, and run these cells in order. No `%%writefile` or patch cells needed. Repo layout: `slop_configs/`, `slop_scripts/`, `slop_src/slop/`.

---

## Cell 1 — GPU check

```python
import torch
print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU. In Colab: Runtime → Change runtime type → T4 GPU")
```

---

## Cell 2 — Clone repo and set root

```python
REPO_URL = "https://github.com/ian-lent/slop-minimization.git"
PROJECT_ROOT = "/content/slop-repo"

%cd /content
!rm -rf /content/slop-repo
!git clone $REPO_URL $PROJECT_ROOT
%cd $PROJECT_ROOT
print("Project root:", PROJECT_ROOT)
```

---

## Cell 3 — Install dependencies

```python
!pip -q install --upgrade pip
!pip -q install torch torchvision torchaudio
!pip -q install transformers datasets peft accelerate pyyaml tqdm scikit-learn sentencepiece
```

---

## Cell 4 — Verify layout

```python
!pwd
!ls slop_configs slop_scripts slop_src
```

---

## Cell 5 — Build data

```python
!cd $PROJECT_ROOT && PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/build_data.py --output-dir data
!ls $PROJECT_ROOT/data
!wc -l $PROJECT_ROOT/data/train.jsonl $PROJECT_ROOT/data/val.jsonl $PROJECT_ROOT/data/test.jsonl
```

---

## Cell 6 — Train classifier (saves to outputs/classifier_curriculum)

```python
!cd $PROJECT_ROOT && PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/train_token_classifier.py \
  --config slop_configs/classifier_encoder.yaml \
  --output-dir outputs/classifier_curriculum
!ls -la $PROJECT_ROOT/outputs/classifier_curriculum/
```

---

## Cell 7 — Generate slop pairs for T5

```python
!cd $PROJECT_ROOT && PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/train_slop_generator.py generate \
  --input data/train.jsonl --output data/slop_pairs.jsonl
!wc -l $PROJECT_ROOT/data/slop_pairs.jsonl
```

---

## Cell 8 — Train T5 slop rewriter (saves to outputs/slop_rewriter)

```python
!cd $PROJECT_ROOT && PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/train_slop_generator.py train \
  --train-path data/slop_pairs.jsonl \
  --output-dir outputs/slop_rewriter \
  --model-name t5-small --epochs 3
!ls -la $PROJECT_ROOT/outputs/slop_rewriter/
```

---

## Cell 9 — Prompt optimization

```python
!cd $PROJECT_ROOT && PYTHONPATH=$PROJECT_ROOT/slop_src python slop_scripts/optimize_prompts.py \
  --config slop_configs/prompt_opt.yaml
!ls -la $PROJECT_ROOT/outputs/prompt_opt/
```

Ensure `slop_configs/prompt_opt.yaml` has `reward.checkpoint_path: outputs/classifier_curriculum` so the trained classifier is used as the reward model.

---

## Cell 10 — Zip critical artifacts for download

```python
import os
import shutil
from pathlib import Path

os.chdir(PROJECT_ROOT)
for d in ["outputs/classifier_curriculum", "outputs/slop_rewriter", "outputs/prompt_opt"]:
    p = Path(d)
    print(f"{d}: exists={p.exists()}")
zip_path = Path("slop_critical_artifacts.zip")
if zip_path.exists():
    zip_path.unlink()
shutil.make_archive(zip_path.with_suffix(""), "zip", "outputs")
print(f"Saved: {zip_path.resolve()}")
print("Download this zip from the Colab file browser.")
```

---

## Optional — Copy to Google Drive

```python
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

DRIVE_BASE = "/content/drive/MyDrive/slop_pipeline"
Path(DRIVE_BASE).mkdir(parents=True, exist_ok=True)

for name, src in [
    ("classifier_curriculum", "outputs/classifier_curriculum"),
    ("slop_rewriter", "outputs/slop_rewriter"),
    ("prompt_opt", "outputs/prompt_opt"),
]:
    src_path = Path(src)
    if src_path.exists():
        dst = Path(DRIVE_BASE) / name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_path, dst, dirs_exist_ok=True)
        print(f"Copied {name} → {dst}")
print(f"Artifacts saved under {DRIVE_BASE}")
```

---

**Essential models trained:** (1) classifier → `outputs/classifier_curriculum`, (2) T5 rewriter → `outputs/slop_rewriter`, (3) prompt optimization → `outputs/prompt_opt`. Weights are saved by the scripts; zip or Drive copy persists them after the session.
