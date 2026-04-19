import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

JSONL_PATH = "outputs/c2_eval.jsonl"
OUT_PATH = "outputs/e5_ternary_progression.png"

baseline = {"step": 0, "label_dist": {"LABEL_0": 0.0, "LABEL_1": 0.0, "LABEL_2": 0.0, "LABEL_3": 100.0}}

rows = [baseline]
with open(JSONL_PATH) as f:
    for line in f:
        rows.append(json.loads(line))

rows.sort(key=lambda r: r["step"])

steps = np.array([r["step"] for r in rows])
l0 = np.array([r["label_dist"]["LABEL_0"] for r in rows])
l1 = np.array([r["label_dist"]["LABEL_1"] for r in rows])
l2 = np.array([r["label_dist"]["LABEL_2"] for r in rows])
l3 = np.array([r["label_dist"]["LABEL_3"] for r in rows])

fig, ax = plt.subplots(figsize=(10, 6))

ax.stackplot(
    steps,
    l3, l2, l1, l0,
    colors=["#cc0000", "#ff8c00", "#7fc97f", "#1a7a1a"],
    labels=["LABEL_3 – Fully AI", "LABEL_2 – Heavily edited",
            "LABEL_1 – Lightly AI-edited", "LABEL_0 – Fully human"],
)

ax.axhline(50, color="black", linestyle="--", linewidth=1.5, label="Majority non-AI (50%)")

# Band annotations at step 500: midpoint of each visible band
# LABEL_3: 0–55 → mid=27.5 | LABEL_1: 55–89 → mid=72 | LABEL_0: 89–100 → mid=94.5
for y_mid, label_text in [
    (27.5, "55% Fully AI"),
    (72.0, "34% Lightly edited"),
    (94.5, "11% Fully human"),
]:
    ax.text(503, y_mid, label_text, fontsize=9, va="center", clip_on=False)

ax.set_xlim(0, 500)
ax.set_ylim(0, 100)
ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("% of Essays", fontsize=12)
ax.set_title(
    "Deslopifier: AI Detection Spectrum Over Training\n"
    "Llama 3.2 3B + REINFORCE + EditLens reward, 500 steps, 48 rollouts/step",
    fontsize=13, linespacing=1.6,
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="upper left", fontsize=9)

plt.subplots_adjust(right=0.82)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"Saved to {OUT_PATH}")
