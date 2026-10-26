# =============================================================================
# visualize_deployment.py - Top-Down TX/RX Deployment Plot
# =============================================================================
#
# AIM:
#   Create a simple 2D overhead plot of the generated transmitter locations,
#   base station locations, and configured TX sampling box.
#
# USAGE:
#   1. Generate the TX/RX CSV first:
#        python src/module_1_scene_gen/tx_rx_location_csv_gen.py --config-name=nyc/nyc_high_rise_rooftop_200ep
#
#   2. Plot the generated layout:
#        python src/module_1_scene_gen/visualize_deployment.py --config-name=nyc/nyc_high_rise_rooftop_200ep
#
# OUTPUT:
#   data/scenes/<selected_id>/deployment_layout_<scenario_name>.png
#
# =============================================================================

import csv
import os

import hydra
import matplotlib
import numpy as np
from dotenv import load_dotenv
from matplotlib.patches import Rectangle
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

load_dotenv()


def resolve_coordinate_csv_path(cfg, project_dir, loc_id):
    """Use the same CSV routing convention as path_gen.py."""
    gps_path = cfg.gps_location_path
    if os.path.isdir(gps_path):
        gps_path = os.path.join(gps_path, "gps_locations.yaml")

    gps_db = OmegaConf.load(gps_path)
    if loc_id in gps_db and "coordinate_file" in gps_db[loc_id]:
        csv_path = gps_db[loc_id].coordinate_file
        csv_path = csv_path.replace("${project_dir}", project_dir)
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(project_dir, csv_path)
        return csv_path

    gps_dir = os.path.dirname(gps_path)
    return os.path.join(gps_dir, f"{loc_id}_tx_rx.csv")


def load_layout_csv(csv_path):
    """Load TX and RX rows from the generated layout CSV."""
    tx_rows = []
    rx_rows = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            point = {
                "name": row["name"],
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
            }

            if row["type"] == "tx":
                tx_rows.append(point)
            elif row["type"] == "rx":
                point["look_at_x"] = float(row["look_at_x"])
                point["look_at_y"] = float(row["look_at_y"])
                rx_rows.append(point)

    return tx_rows, rx_rows


def as_xy_array(rows):
    """Convert rows with x/y keys into a NumPy array for easy plotting."""
    if not rows:
        return np.empty((0, 2), dtype=float)
    return np.array([[row["x"], row["y"]] for row in rows], dtype=float)


def add_tx_box(ax, box):
    """Draw the configured mobile sampling box."""
    width = box.x_max - box.x_min
    height = box.y_max - box.y_min
    rectangle = Rectangle(
        (box.x_min, box.y_min),
        width,
        height,
        fill=False,
        edgecolor="black",
        linewidth=2.0,
        linestyle="--",
        label="TX sampling box",
    )
    ax.add_patch(rectangle)


def plot_base_stations(ax, rx_rows):
    """Plot base station points, labels, and 2D look-at arrows."""
    if not rx_rows:
        return

    rx_xy = as_xy_array(rx_rows)
    ax.scatter(
        rx_xy[:, 0],
        rx_xy[:, 1],
        marker="^",
        s=120,
        c="red",
        edgecolors="black",
        linewidths=0.8,
        label=f"Base stations (n={len(rx_rows)})",
        zorder=4,
    )

    for row in rx_rows:
        ax.annotate(
            row["name"],
            xy=(row["x"], row["y"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
        )

        dx = row["look_at_x"] - row["x"]
        dy = row["look_at_y"] - row["y"]
        ax.arrow(
            row["x"],
            row["y"],
            dx,
            dy,
            length_includes_head=True,
            head_width=4.0,
            head_length=6.0,
            linewidth=1.0,
            color="red",
            alpha=0.65,
            zorder=3,
        )


def set_plot_bounds(ax, box, tx_xy, rx_rows):
    """Choose plot limits that include both the TX box and outside base stations."""
    x_values = [box.x_min, box.x_max]
    y_values = [box.y_min, box.y_max]

    if len(tx_xy) > 0:
        x_values.extend(tx_xy[:, 0])
        y_values.extend(tx_xy[:, 1])

    for row in rx_rows:
        x_values.extend([row["x"], row["look_at_x"]])
        y_values.extend([row["y"], row["look_at_y"]])

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    padding = 0.08 * max(x_max - x_min, y_max - y_min, 1.0)
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)


@hydra.main(version_base=None, config_path="../../configs/scenarios")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    project_dir = cfg.project_dir
    loc_id = cfg.selected_id
    scenario_name = cfg.scenario_name

    csv_path = resolve_coordinate_csv_path(cfg, project_dir, loc_id)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"TX/RX CSV not found: {csv_path}\n"
            "Run tx_rx_location_csv_gen.py for this config before plotting."
        )

    tx_rows, rx_rows = load_layout_csv(csv_path)
    tx_xy = as_xy_array(tx_rows)

    out_dir = os.path.join(project_dir, "data", "scenes", loc_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"deployment_layout_{scenario_name}.png")

    fig, ax = plt.subplots(figsize=(9, 8))
    add_tx_box(ax, cfg.deployment.tx_box)

    if len(tx_xy) > 0:
        ax.scatter(
            tx_xy[:, 0],
            tx_xy[:, 1],
            s=8,
            c="tab:blue",
            alpha=0.45,
            linewidths=0,
            label=f"TX samples (n={len(tx_xy)})",
        )

    plot_base_stations(ax, rx_rows)
    set_plot_bounds(ax, cfg.deployment.tx_box, tx_xy, rx_rows)

    ax.set_title(f"Deployment Layout: {scenario_name}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Loaded layout CSV : {csv_path}")
    print(f"TX samples        : {len(tx_rows)}")
    print(f"Base stations    : {len(rx_rows)}")
    print(f"Saved plot        : {out_path}")


if __name__ == "__main__":
    main()
