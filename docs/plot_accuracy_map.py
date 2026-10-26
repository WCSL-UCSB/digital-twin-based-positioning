"""Plot two positioning models on shared test locations from saved errors."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


ROOT = Path(__file__).resolve().parent


def main():
    with np.load(ROOT / "results" / "positioning-map.npz", allow_pickle=False) as data:
        xy = data["xy_m"]
        bounds = data["bounds_m"]
        errors = [data["early_fusion_errors_m"], data["late_fusion_errors_m"]]
    if xy.shape != (3841, 2) or any(e.shape != (3841, 3) for e in errors):
        raise ValueError("Expected shared test positions and three runs per model")
    if not np.isfinite(xy).all() or any(not np.isfinite(e).all() for e in errors):
        raise ValueError("Map inputs must be finite")
    background = plt.imread(ROOT / "assets" / "positioning-map-background.png")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "#233746", "axes.labelcolor": "#233746",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.88, bottom=0.15, top=0.84, wspace=0.12)
    norm = Normalize(vmin=0, vmax=30, clip=True)
    for ax, matrix, label in zip(axes, errors, ["Early-fusion CNN", "Late-fusion CNN"]):
        values = matrix.mean(axis=1)
        ax.imshow(background, extent=bounds, origin="upper", zorder=0)
        points = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap="viridis", norm=norm,
                            s=7, linewidths=0, alpha=1, zorder=2)
        ax.set(xlim=bounds[:2], ylim=bounds[2:], aspect="equal",
               xlabel="Local x (m)", xticks=[-150, -100, -50, 0, 50],
               yticks=[-200, -150, -100, -50, 0])
        ax.set_title(f"{label}\nMean error {values.mean():.2f} m", fontsize=13, pad=12)
        ax.tick_params(length=3, colors="#526572")
    axes[0].set_ylabel("Local y (m)")
    cax = fig.add_axes([0.905, 0.2, 0.018, 0.58])
    bar = fig.colorbar(points, cax=cax, extend="max")
    bar.set_ticks([0, 5, 10, 15, 20, 25, 30], labels=["0", "5", "10", "15", "20", "25", "30+"])
    bar.set_label("Positioning error (m)", labelpad=10)
    fig.savefig(ROOT / "assets" / "positioning-results.png", dpi=200,
                facecolor="white", metadata={"Software": "Matplotlib"})
    plt.close(fig)


if __name__ == "__main__":
    main()
