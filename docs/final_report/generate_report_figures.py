from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "docs" / "final_report" / "figures"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 220,
        }
    )


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.8)
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)


def make_editlens_reference_bands() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    bands = [
        ("LABEL_0", 0.007, 0.035, "Fully human", "#4c78a8"),
        ("LABEL_1", 0.10, 0.43, "Lightly AI-edited", "#72b7b2"),
        ("LABEL_2", 0.43, 0.76, "Heavily edited", "#f2cf5b"),
    ]
    y_positions = [3, 2, 1]

    for (label, x0, x1, desc, color), y in zip(bands, y_positions):
        ax.hlines(y, x0, x1, color=color, linewidth=12, zorder=2)
        ax.scatter([x0, x1], [y, y], s=18, color=color, edgecolor="black", linewidth=0.5, zorder=3)
        ax.text(1.06, y, f"{label}: {x0:.3f}-{x1:.3f}", va="center", ha="left", fontsize=9.5)

    ax.scatter([0.999], [0], s=95, color="#e45756", edgecolor="black", linewidth=0.8, zorder=3)
    ax.text(1.06, 0, "LABEL_3: typical score ≈ 0.999", va="center", ha="left", fontsize=9.5)

    ax.set_xlim(0.0, 1.48)
    ax.set_ylim(-0.6, 3.6)
    ax.set_xlabel("EditLens score")
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(
        [
            "Fully human",
            "Lightly AI-edited",
            "Heavily AI-edited",
            "Fully AI-generated",
        ]
    )
    ax.grid(axis="x", color="#cccccc", linewidth=0.6, alpha=0.6)
    ax.spines["left"].set_visible(False)
    ax.text(1.06, 3.42, "Score range", ha="left", va="center", fontsize=9.5, color="#444444")
    save(fig, "editlens_reference_bands.png")


def make_tournament_tradeoff() -> None:
    rows = load_csv(ROOT / "tournament" / "results" / "phase2" / "leaderboard.csv")
    x = [float(r["mean_similarity"]) for r in rows]
    y = [float(r["mean_delta_editlens"]) for r in rows]
    size = [6500 * float(r["pass_rate"]) + 40 for r in rows]
    color = [float(r["median_length_ratio"]) for r in rows]
    labels = [r["run_name"].split("_")[0].upper() for r in rows]

    fig, ax = plt.subplots(figsize=(6.8, 4.7))
    sc = ax.scatter(
        x,
        y,
        s=size,
        c=color,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.8,
    )
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), xytext=(5, 6), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Mean semantic similarity")
    ax.set_ylabel("Mean Δ EditLens (higher is better)")
    ax.grid(color="#d7d7d7", linewidth=0.6, alpha=0.7)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Median length ratio")
    ax.text(
        0.02,
        0.02,
        "Point size = pass rate",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "boxstyle": "round,pad=0.25"},
    )
    save(fig, "tournament_phase2_tradeoff.png")


def make_rl_progression() -> None:
    rows = load_jsonl(ROOT / "outputs" / "c2_eval.jsonl")
    steps = [0] + [int(r["step"]) for r in rows]
    mean_editlens = [0.9990] + [float(r["mean_editlens"]) for r in rows]
    label0 = [0.0] + [float(r["label_dist"]["LABEL_0"]) for r in rows]
    label1 = [0.0] + [float(r["label_dist"]["LABEL_1"]) for r in rows]
    label2 = [0.0] + [float(r["label_dist"]["LABEL_2"]) for r in rows]
    label3 = [100.0] + [float(r["label_dist"]["LABEL_3"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))

    axes[0].plot(steps, mean_editlens, color="#2f5597", marker="o", linewidth=2)
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Mean EditLens")
    axes[0].set_ylim(0.6, 1.02)
    axes[0].set_title("(a) Mean detector score", fontsize=11, pad=4)
    axes[0].grid(color="#d7d7d7", linewidth=0.6, alpha=0.7)

    axes[1].stackplot(
        steps,
        label0,
        label1,
        label2,
        label3,
        colors=["#4c78a8", "#72b7b2", "#f2cf5b", "#e45756"],
        labels=["LABEL_0", "LABEL_1", "LABEL_2", "LABEL_3"],
        alpha=0.95,
    )
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Share of evaluation set (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("(b) Label distribution", fontsize=11, pad=4)
    axes[1].legend(loc="upper right", ncol=2, frameon=True)
    axes[1].grid(color="#d7d7d7", linewidth=0.6, alpha=0.7)

    save(fig, "rl_progression_clean.png")


def make_kl_ablation() -> None:
    points = [
        ("C2", 0.5183, 0.6700, "#4c78a8"),
        ("C4-a", 0.6371, 0.6325, "#72b7b2"),
        ("C4-c", 3.7347, 0.4307, "#e45756"),
    ]

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    offsets = {"C2": (8, 10), "C4-a": (8, 6), "C4-c": (8, 6)}
    for label, kl, editlens, color in points:
        ax.scatter(kl, editlens, s=135, color=color, edgecolor="black", linewidth=0.8, zorder=3)
        dx, dy = offsets[label]
        ax.annotate(label, (kl, editlens), xytext=(dx, dy), textcoords="offset points", fontsize=10)

    ax.set_xlabel("KL divergence from reference model")
    ax.set_ylabel("Mean EditLens at step 500 (lower is better)")
    ax.grid(color="#d7d7d7", linewidth=0.6, alpha=0.7)
    ax.set_xlim(0.0, 4.1)
    ax.set_ylim(0.38, 0.72)
    ax.annotate(
        "preferred direction",
        xy=(0.16, 0.395),
        xytext=(1.35, 0.705),
        textcoords="data",
        fontsize=9.5,
        color="#555555",
        arrowprops={"arrowstyle": "->", "color": "#777777", "linewidth": 1.0},
    )
    save(fig, "kl_ablation_clean.png")


def draw_box(ax, xy, width, height, text, facecolor="#f7f7f7", fontsize=10):
    x0, y0 = xy
    box = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="black",
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(x0 + width / 2, y0 + height / 2, text, ha="center", va="center", fontsize=fontsize)


def make_promptlab_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.92, "Stage 1: evolutionary search", fontsize=11.8, weight="bold")
    ax.text(0.05, 0.42, "Stage 2: supervised distillation", fontsize=11.8, weight="bold")

    draw_box(ax, (0.05, 0.66), 0.16, 0.14, "Topic and\nseed prompt", fontsize=10.5)
    draw_box(ax, (0.26, 0.66), 0.18, 0.14, "Mutate and recombine\nwith Llama-3.1-8B", facecolor="#eef3fb", fontsize=10.5)
    draw_box(ax, (0.49, 0.66), 0.18, 0.14, "Generate essays\nwith Llama-3.3-70B", facecolor="#eef8ef", fontsize=10.5)
    draw_box(ax, (0.72, 0.66), 0.20, 0.14, "Score with detector,\ndrift penalty,\nand exemplar reuse", facecolor="#fff6e8", fontsize=10.3)
    draw_box(ax, (0.64, 0.46), 0.22, 0.10, "372 contrastive pairs\nacross 77 topics", facecolor="#f8f8f8", fontsize=10.4)
    draw_box(ax, (0.35, 0.16), 0.30, 0.15, "Train T5-base + LoRA\nfor essay-to-essay rewriting", facecolor="#f5effa", fontsize=10.8)

    ax.annotate("", xy=(0.26, 0.73), xytext=(0.21, 0.73), arrowprops={"arrowstyle": "->", "linewidth": 1.25})
    ax.annotate("", xy=(0.49, 0.73), xytext=(0.44, 0.73), arrowprops={"arrowstyle": "->", "linewidth": 1.25})
    ax.annotate("", xy=(0.72, 0.73), xytext=(0.67, 0.73), arrowprops={"arrowstyle": "->", "linewidth": 1.25})
    ax.annotate("", xy=(0.75, 0.56), xytext=(0.82, 0.66), arrowprops={"arrowstyle": "->", "linewidth": 1.25})
    ax.annotate("", xy=(0.50, 0.31), xytext=(0.75, 0.46), arrowprops={"arrowstyle": "->", "linewidth": 1.25})

    ax.text(
        0.50,
        0.05,
        "Training details: curriculum ordering, AdamW, cosine decay, early stopping on mean slop,\n"
        "and deterministic beam evaluation.",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#444444",
    )
    save(fig, "promptlab_v2_pipeline.png")


def make_promptlab_results() -> None:
    labels = ["Prompt rewriting\nbaseline", "T5-base LoRA\nbest run"]
    values = [0.3910, 0.0998]
    colors = ["#b9c8e6", "#4c78a8"]

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8, width=0.6)
    ax.set_ylabel("Mean slop on held-out validation pairs")
    ax.set_ylim(0, 0.45)
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.6, alpha=0.7)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.4f}", ha="center", va="bottom", fontsize=10)

    ax.text(
        0.98,
        0.90,
        "74% reduction\n(20 validation pairs)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "boxstyle": "round,pad=0.25"},
    )
    save(fig, "promptlab_v2_results.png")


def main() -> None:
    configure_style()
    make_editlens_reference_bands()
    make_tournament_tradeoff()
    make_rl_progression()
    make_kl_ablation()
    make_promptlab_pipeline()
    make_promptlab_results()


if __name__ == "__main__":
    main()
