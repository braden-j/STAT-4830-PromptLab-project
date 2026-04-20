"""
scripts/c4_pareto_chart.py — Experiment C4: KL-penalty ablation Pareto chart

Reads step-500 entries from c2, c4a, and c4c eval logs and plots detection
evasion (x) vs KL divergence (y, inverted so lower KL is at top) for each run.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUNS = [
    ("C2 (KL=0.1)",   "c2_eval.jsonl"),
    ("C4-a (KL=0.0)", "c4a_eval.jsonl"),
    ("C4-c (KL=0.5)", "c4c_eval.jsonl"),
]
OUTPUT = os.path.join(_REPO_ROOT, "outputs", "c4_pareto.png")


def load_step500(filename: str) -> dict:
    path = os.path.join(_REPO_ROOT, "outputs", filename)
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("step") == 500:
                return rec
    raise ValueError(f"No step=500 entry found in {filename}")


def main() -> None:
    points = []
    for label, filename in RUNS:
        rec = load_step500(filename)
        x  = 1.0 - rec["mean_editlens"]  # detection evasion: higher = better
        kl = rec["kl_div"]
        points.append((label, x, kl))
        print(f"{label}: evasion={x:.4f}  kl={kl:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))

    colors  = ["#1976D2", "#FF8C00", "#E53935"]
    markers = ["o", "s", "^"]

    # ── Shaded rectangle around C2 and C4-a (valid operating points) ─────────
    valid = [(x, kl) for label, x, kl in points if "C4-c" not in label]
    pad_x = 0.025
    pad_y = 0.06
    rect_x0 = min(v[0] for v in valid) - pad_x
    rect_x1 = max(v[0] for v in valid) + pad_x
    rect_y0 = min(v[1] for v in valid) - pad_y
    rect_y1 = max(v[1] for v in valid) + pad_y
    rect = mpatches.FancyBboxPatch(
        (rect_x0, rect_y0),
        rect_x1 - rect_x0,
        rect_y1 - rect_y0,
        boxstyle="round,pad=0.01",
        linewidth=1.5,
        edgecolor="#555555",
        facecolor="#F5F5F5",
        alpha=0.45,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(
        (rect_x0 + rect_x1) / 2,
        rect_y1 - 0.015,
        "Valid operating points",
        fontsize=9,
        color="#555555",
        ha="center",
        va="top",
        fontweight="bold",
        zorder=6,
    )

    # EditLens scores for each run (step-500 mean_editlens values)
    editlens_labels = {
        "C2 (KL=0.1)":   "EditLens: 0.670",
        "C4-a (KL=0.0)": "EditLens: 0.633",
        "C4-c (KL=0.5)": "EditLens: 0.431",
    }

    # ── Scatter points ────────────────────────────────────────────────────────
    for (label, x, kl), color, marker in zip(points, colors, markers):
        ax.scatter(x, kl, color=color, marker=marker, s=120, zorder=5, label=label)
        # C4-a offset right so its label doesn't overlap C2's
        xoff = 0.022 if "C4-a" in label else 0.007
        yoff = -0.06
        ax.annotate(
            label,
            xy=(x, kl),
            xytext=(x + xoff, kl + yoff),
            fontsize=10,
            color=color,
            fontweight="bold",
            zorder=7,
        )
        # Small gray EditLens score: above (visually) for C2/C4-a, below for C4-c
        if "C4-c" in label:
            el_y, el_va = kl + 0.13, "top"
        else:
            el_y, el_va = kl - 0.13, "bottom"
        ax.text(
            x, el_y,
            editlens_labels[label],
            fontsize=8,
            color="gray",
            ha="center",
            va=el_va,
            zorder=7,
        )

    # ── Arrow annotation for C4-c reward hacking ─────────────────────────────
    c4c = next(p for p in points if "C4-c" in p[0])
    ax.annotate(
        "Reward hacking: KL exploded to 3.73",
        xy=(c4c[1], c4c[2]),
        xytext=(c4c[1] - 0.11, c4c[2] + 0.30),
        fontsize=9,
        color="#E53935",
        arrowprops=dict(arrowstyle="->", color="#E53935", lw=1.4),
        ha="center",
        zorder=7,
    )

    ax.set_xlabel("1 - mean EditLens  (detection evasion, higher = better)", fontsize=11)
    ax.set_ylabel(
        "KL divergence from base model (lower = more constrained)", fontsize=11
    )
    ax.set_title(
        "KL Penalty Ablation: Detection Evasion vs Training Stability", fontsize=13, fontweight="bold"
    )
    ax.invert_yaxis()   # lower KL at the top
    ax.set_ylim(4.2, 0.0)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150)
    print(f"\nSaved -> {OUTPUT}")


if __name__ == "__main__":
    main()
