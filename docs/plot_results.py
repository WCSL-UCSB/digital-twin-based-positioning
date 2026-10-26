"""Render the published nominal-positioning summary from saved per-run metrics."""

import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results" / "nominal-positioning-2026-10-26.json"
OUTPUT = ROOT / "assets" / "positioning-results.png"
SETTINGS = [
    ("C03", "nominal", "Early fusion · nominal", "#227c91"),
    ("C03", "mixture", "Early fusion · mixed", "#227c91"),
    ("I-joint", "nominal", "Late fusion · nominal", "#b65c28"),
    ("I-joint", "mixture", "Late fusion · mixed", "#b65c28"),
]


def main():
    report = json.loads(DATA.read_text())
    rows = report["rows"]
    expected = {
        (weight, model, recipe, seed)
        for weight in (0.5, 50.0)
        for model, recipe, _, _ in SETTINGS
        for seed in (71, 72, 73)
    }
    actual = {(r["ce_weight"], r["model"], r["recipe"], r["seed"]) for r in rows}
    if len(rows) != 24 or actual != expected:
        raise ValueError("Expected one nominal evaluation per model/recipe/weight/seed")
    if any(r["condition"] != "nominal" or r["metrics"]["raw"]["n_finite"] != 3841
           or r["metrics"]["raw"]["n_failed"] != 0 for r in rows):
        raise ValueError("The figure requires complete nominal evaluations")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": "#233746",
        "axes.labelcolor": "#233746",
        "xtick.color": "#526572",
        "ytick.color": "#233746",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": "#d6dfe3",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.23, top=0.69, wspace=0.12)
    fig.text(0.035, 0.92, "Positioning from simulated radio fingerprints",
             fontsize=20, weight="bold")
    fig.text(0.035, 0.84,
             "Nominal test inputs · 3,841 held-out positions · 4 base stations",
             fontsize=12, color="#526572")

    for ax, weight in zip(axes, (0.5, 50.0)):
        ax.set_title(f"Cross-entropy weight λ = {weight:g}", loc="left",
                     fontsize=12, weight="bold", pad=17)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e8edf0", linewidth=0.8)
        ax.set_xlim(0, 15.8)
        ax.set_xticks([0, 5, 10, 15])
        ax.set_ylim(-0.55, 3.55)
        ax.set_xlabel("Mean position error (m)", labelpad=10)
        ax.tick_params(axis="both", length=0, pad=8)
        for i, (model, recipe, _, color) in enumerate(SETTINGS):
            values = [r["metrics"]["raw"]["mean_m"] for r in rows
                      if r["ce_weight"] == weight and r["model"] == model
                      and r["recipe"] == recipe]
            avg, sd = mean(values), stdev(values)
            y = 3 - i
            ax.axhline(y, color="#f2f5f6", linewidth=1, zorder=0)
            ax.errorbar(avg, y, xerr=sd, fmt="o", color=color, markersize=8,
                        elinewidth=2, capsize=5, capthick=1.5, zorder=3)
            ax.text(15.5, y, f"{avg:.2f}", ha="right", va="center", fontsize=11,
                    weight="bold", color=color,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    axes[0].set_yticks([3, 2, 1, 0], [s[2] for s in SETTINGS])
    fig.text(0.035, 0.075,
             "Dots: average of 3 run means   |   Bars: sample SD   |   Raw XY predictions   |   Lower is better",
             fontsize=10.5, color="#526572")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    print(f"Wrote {OUTPUT.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
