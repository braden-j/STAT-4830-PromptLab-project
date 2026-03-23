# Critical Models and Saving Weights (from Colab)

Based on the Colab notebook: after experimenting, the **critical models** are identified below. Other experiments (e.g. multiple generators) can be dropped.

---

## Critical models

| Role | Model | Notes |
|------|--------|------|
| **Classifier (reward model)** | **DistilBERT** (`distilbert-base-uncased`) | Trained with `train_token_classifier.py`. Used for slop scoring. Saves to `outputs/classifier`, `outputs/classifier_curriculum`, or `outputs/classifier_baseline_no_curriculum`. |
| **Generator** | **TinyLlama** (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) | Used as frozen causal LM for prompt optimization. No training in the notebook; loaded from HuggingFace. |

**Experiments you can drop**

- **Generator comparison** with `gpt2` and `Qwen/Qwen2.5-0.5B-Instruct` — keep only TinyLlama.
- If you’ve settled on one training setup: either **curriculum** or **baseline** classifier; you can keep just that output dir and skip the other.

---

## How weights are saved

### 1. Classifier (reward model)

The training script **already saves** the classifier:

- **Path**: `outputs/classifier` (or `outputs/classifier_curriculum`, `outputs/classifier_baseline_no_curriculum` if you use `--output-dir`).
- **Files**:
  - `pytorch_model.bin` — full `state_dict` of the classifier.
  - Tokenizer files: `config.json`, `tokenizer_config.json`, `tokenizer.json`, `vocab.txt`, etc.

Code inside `train_token_classifier.py` (and thus the Colab flow that runs it):

```python
save_path = out_dir / "pytorch_model.bin"
torch.save(model.state_dict(), save_path)
tokenizer.save_pretrained(out_dir)
```

So after training, the weights are on disk in that `outputs/...` folder. In Colab you only need to **persist or download** that folder (see below).

### 2. Generator (TinyLlama)

TinyLlama is **not trained** in this notebook; it’s loaded from HuggingFace each run. To **cache it locally** (e.g. under `outputs/` or Google Drive) so you don’t re-download every time, use the code in the next section.

---

## Code: Saving / persisting weights (for Colab)

Use these in the Colab notebook.

### A. Classifier — ensure checkpoint is saved (already done by script)

Training already writes to `outputs/classifier` (or your `--output-dir`). No extra save step needed unless you want to copy elsewhere.

### B. Classifier — copy to Google Drive (persist after session)

```python
# Run after training the classifier. Adjust paths if you used curriculum/baseline.
import shutil
from pathlib import Path

DRIVE_MOUNT = "/content/drive"  # mount with: from google.colab import drive; drive.mount(DRIVE_MOUNT)
CLASSIFIER_OUT = "outputs/classifier_curriculum"  # or outputs/classifier / outputs/classifier_baseline_no_curriculum
DRIVE_SAVE = f"{DRIVE_MOUNT}/MyDrive/slop/classifier"

Path(DRIVE_SAVE).mkdir(parents=True, exist_ok=True)
shutil.copytree(CLASSIFIER_OUT, DRIVE_SAVE, dirs_exist_ok=True)
print(f"Classifier weights and tokenizer copied to {DRIVE_SAVE}")
```

### C. Classifier — download as zip (for local use)

```python
# Run after training. Creates a zip you can download from Colab.
from pathlib import Path
import shutil

out_dir = Path("outputs/classifier_curriculum")  # or outputs/classifier
zip_path = Path("classifier_weights.zip")
if zip_path.exists():
    zip_path.unlink()
shutil.make_archive(zip_path.with_suffix(""), "zip", out_dir)
print(f"Download: {zip_path.resolve()} (e.g. from Colab file browser)")
```

### D. TinyLlama — save/cache to local directory (avoid re-downloading)

```python
# Cache TinyLlama to a local dir so later runs can load from disk.
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
cache_dir = Path("outputs/generator_cache/tinyllama")
cache_dir.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=str(cache_dir))

# Save to a single directory for later loading with from_pretrained(local_path)
save_dir = cache_dir / "saved"
save_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"TinyLlama saved to {save_dir}")
```

Later, use the cached copy so you don’t hit HuggingFace every time:

```python
# In configs/prompt_opt.yaml set generator.model_name to the path, e.g.:
#   model_name: outputs/generator_cache/tinyllama/saved
# Or load in code:
# model = AutoModelForCausalLM.from_pretrained("outputs/generator_cache/tinyllama/saved")
# tokenizer = AutoTokenizer.from_pretrained("outputs/generator_cache/tinyllama/saved")
```

### E. TinyLlama — copy cached generator to Drive

```python
# After running (D), persist TinyLlama to Drive.
import shutil
from pathlib import Path

DRIVE_MOUNT = "/content/drive"
src = Path("outputs/generator_cache/tinyllama/saved")
dst = Path(f"{DRIVE_MOUNT}/MyDrive/slop/generator_tinyllama")
dst.mkdir(parents=True, exist_ok=True)
shutil.copytree(src, dst, dirs_exist_ok=True)
print(f"TinyLlama weights copied to {dst}")
```

---

## Minimal Colab flow (critical models only)

1. **Setup**: Clone repo, install deps, build data (as in notebook).
2. **Train classifier** (curriculum only if that’s your choice):
   ```bash
   PYTHONPATH=src python scripts/train_token_classifier.py --config configs/classifier_encoder.yaml --output-dir outputs/classifier_curriculum
   ```
3. **Save classifier**: Run (B) and/or (C) above.
4. **Prompt optimization** (TinyLlama only): Use `configs/prompt_opt.yaml` with `generator.model_name: TinyLlama/TinyLlama-1.1B-Chat-v1.0` (or the local path from (D)).
5. **Optional**: Run (D) once to cache TinyLlama; then (E) to copy to Drive. Skip `compare_generators.py` with multiple generators.

This keeps only the critical models (DistilBERT classifier + TinyLlama generator) and gives you explicit code to save and reuse their weights.
