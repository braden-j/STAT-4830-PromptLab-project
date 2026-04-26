import json
from pathlib import Path

INPUT = Path("outputs/c2_eval.jsonl")
OUTPUT = Path("outputs/e4_ablation_table.md")

rows = []
with open(INPUT) as f:
    for line in f:
        d = json.loads(line)
        rows.append(d)

rows.sort(key=lambda r: r["step"])

baseline = {
    "step": 0,
    "mean_editlens": 0.999,
    "label_dist": {"LABEL_0": 0.0, "LABEL_1": 0.0, "LABEL_3": 100.0},
    "kl_div": 0.000,
}

def fmt_row(r, label=""):
    ld = r["label_dist"]
    step = label if label else str(r["step"])
    return (
        f"| {step:>8} "
        f"| {r['mean_editlens']:>13.4f} "
        f"| {ld.get('LABEL_0', 0.0):>9.1f} "
        f"| {ld.get('LABEL_1', 0.0):>9.1f} "
        f"| {ld.get('LABEL_3', 0.0):>9.1f} "
        f"| {r['kl_div']:>12.4f} |"
    )

header = (
    "| {:>8} | {:>13} | {:>9} | {:>9} | {:>9} | {:>12} |".format(
        "Step", "Mean EditLens", "LABEL_0 %", "LABEL_1 %", "LABEL_3 %", "KL Divergence"
    )
)
sep = "|" + "|".join(["-" * (w + 2) for w in [8, 13, 9, 9, 9, 12]]) + "|"

last = rows[-1]
delta = {
    "step": "delta",
    "mean_editlens": last["mean_editlens"] - baseline["mean_editlens"],
    "label_dist": {
        "LABEL_0": last["label_dist"].get("LABEL_0", 0.0) - baseline["label_dist"]["LABEL_0"],
        "LABEL_1": last["label_dist"].get("LABEL_1", 0.0) - baseline["label_dist"]["LABEL_1"],
        "LABEL_3": last["label_dist"].get("LABEL_3", 0.0) - baseline["label_dist"]["LABEL_3"],
    },
    "kl_div": last["kl_div"] - baseline["kl_div"],
}

lines = [header, sep, fmt_row(baseline, label="0 (base)")]
for r in rows:
    lines.append(fmt_row(r))
lines.append(sep)
lines.append(fmt_row(delta, label="d 0->500"))

table = "\n".join(lines)
print(table)
OUTPUT.write_text(table + "\n")
print(f"\nSaved to {OUTPUT}")
