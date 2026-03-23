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

