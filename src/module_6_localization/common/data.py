"""Dataset loading, spatial grids, and split utilities for localization."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.utils.dataset_io import resolve_input  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Outdoor grid (cell centers filtered by building polygons)
# ---------------------------------------------------------------------------

class GridHelper:
    """
    Spatial grid for classification / soft-argmax output modes.

    Divides the deployment area into cells and optionally removes any
    cell whose center falls inside a building polygon. The surviving
    cell centers become the model's output vocabulary.
    """

    def __init__(self, x_min, x_max, y_min, y_max, cell_size=2.0,
                 building_polygons=None, margin=0.0):
        self.cell_size = cell_size
        self.x_min = x_min - margin
        self.x_max = x_max + margin
        self.y_min = y_min - margin
        self.y_max = y_max + margin

        xs = np.arange(self.x_min + cell_size / 2,
                       self.x_max, cell_size)
        ys = np.arange(self.y_min + cell_size / 2,
                       self.y_max, cell_size)
        self.nx = len(xs)
        self.ny = len(ys)
        self.full_grid_shape = (self.ny, self.nx)

        xx, yy = np.meshgrid(xs, ys)
        all_centers = np.column_stack([xx.ravel(), yy.ravel()])

        if building_polygons is not None and len(building_polygons) > 0:
            from matplotlib.path import Path

            outdoor_mask = np.ones(len(all_centers), dtype=bool)
            for poly_verts in building_polygons:
                path = Path(poly_verts[:, :2])
                inside = path.contains_points(all_centers)
                outdoor_mask &= ~inside

            self.outdoor_mask = outdoor_mask
            self.centers = all_centers[outdoor_mask].astype(np.float32)
        else:
            self.outdoor_mask = np.ones(len(all_centers), dtype=bool)
            self.centers = all_centers.astype(np.float32)

        self.num_cells = len(self.centers)

    def position_to_cell_index(self, xy):
        """Map (N, 2) positions to nearest outdoor cell index."""
        diffs = xy[:, None, :] - self.centers[None, :, :]
        dists = np.sum(diffs ** 2, axis=-1)
        return np.argmin(dists, axis=1)

    def visualize(self, save_path=None, building_polygons=None):
        fig, ax = plt.subplots(figsize=(12, 10))

        if building_polygons is not None:
            from matplotlib.collections import PatchCollection
            from matplotlib.patches import Polygon as MplPolygon
            patches = []
            for verts in building_polygons:
                try:
                    patches.append(MplPolygon(verts[:, :2], closed=True))
                except Exception:
                    continue
            if patches:
                pc = PatchCollection(patches, facecolor="#CCCCCC",
                                     edgecolor="#888888", linewidth=0.3,
                                     alpha=0.7)
                ax.add_collection(pc)

        ax.scatter(self.centers[:, 0], self.centers[:, 1],
                   c="green", s=0.5, alpha=0.4,
                   label=f"Outdoor ({self.num_cells})")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Outdoor Grid — {self.num_cells} cells "
                     f"({self.cell_size}m spacing)")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.autoscale_view()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"  Grid visualization saved: {save_path}")
        plt.close()


# ---------------------------------------------------------------------------
# Building footprints
# ---------------------------------------------------------------------------

def load_building_polygons(scene_dir):
    """Load building footprints from .npz as a list of Nx2 numpy arrays."""
    npz_path = os.path.join(scene_dir, "building_footprints.npz")
    if not os.path.exists(npz_path):
        print(f"  No building_footprints.npz in {scene_dir}")
        return []

    data = np.load(npz_path, allow_pickle=True)
    num_buildings = int(data["num_buildings"][0])

    polygons = []
    for i in range(num_buildings):
        coords = data[f"poly_{i:04d}"]
        if len(coords) >= 3:
            polygons.append(coords.astype(np.float64))

    print(f"  Loaded {len(polygons)} building polygons.")
    return polygons


# ---------------------------------------------------------------------------
# Preprocessed dataset loading
# ---------------------------------------------------------------------------

def get_latest_preprocessed(data_dir, scenario_name, explicit=None):
    """Newest `*_preprocessed_<scenario>.npz` by filename timestamp, or the
    `inputs.preprocessed_npz` pin when one is set (see src/utils/dataset_io.py)."""
    return resolve_input(
        os.path.join(data_dir, f"*_preprocessed_{scenario_name}.npz"),
        explicit, label="preprocessed dataset")


def _collapse_multi_bs_metadata(extras, key, rule="any"):
    """Collapse a per-BS metadata field from (N, K) to (N,) if needed."""
    if key not in extras:
        return
    v = extras[key]
    if v.ndim < 2:
        return

    if rule == "any":
        extras[key] = np.any(v, axis=1)
    elif rule == "max":
        extras[key] = np.max(v, axis=1)
    elif rule == "min":
        extras[key] = np.min(v, axis=1)


def load_preprocessed(data_dir, scenario_name, explicit=None):
    ds_file = get_latest_preprocessed(data_dir, scenario_name, explicit)

    data = np.load(ds_file, allow_pickle=False)
    X = data["X"]
    Y = data["Y"].astype(np.float32)

    print(f"  X shape     : {X.shape}  dtype={X.dtype}")
    print(f"  Y shape     : {Y.shape}")

    extras = {}
    for k in ("is_los_kept", "has_paths", "shift_ns",
              "snr_raw_db", "snr_bf_db", "snr_detector_db",
              "p_rx_total_dbm",
              "link_rx_idx", "link_tx_idx", "tx_idx",
              "tx_pos", "rx_pos", "tx_names", "rx_names"):
        if k in data.files:
            extras[k] = data[k]

    # Keep pre-collapse per-BS copies for per-BS-count metrics and ablation.
    for k in ("has_paths", "is_los_kept", "snr_detector_db"):
        if k in extras and getattr(extras[k], "ndim", 1) >= 2:
            extras[k + "_per_bs"] = extras[k].copy()

    # The split artifact is keyed on the dataset basename so a regenerated
    # dataset can never silently reuse a stale split.
    extras["_dataset_basename"] = os.path.basename(ds_file)

    _collapse_multi_bs_metadata(extras, "is_los_kept", rule="any")
    _collapse_multi_bs_metadata(extras, "has_paths",   rule="any")
    _collapse_multi_bs_metadata(extras, "shift_ns",    rule="min")
    _collapse_multi_bs_metadata(extras, "snr_raw_db",  rule="max")
    _collapse_multi_bs_metadata(extras, "snr_bf_db",   rule="max")
    _collapse_multi_bs_metadata(extras, "snr_detector_db", rule="max")
    _collapse_multi_bs_metadata(extras, "p_rx_total_dbm",  rule="max")

    K = X.shape[-1]
    if K > 1:
        print(f"  Multi-BS    : K={K} base stations")
    else:
        print(f"  Single-BS   : K=1")

    return X, Y, extras


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def make_split(X, Y, test_size, random_seed):
    """Deterministic train/test split shared across every model script."""
    indices = np.arange(len(Y))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_seed
    )
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    return train_idx, test_idx, X_train, X_test, Y_train, Y_test
