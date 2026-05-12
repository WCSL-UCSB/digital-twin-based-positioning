# =============================================================================
# tx_rx_location_csv_gen.py — TX/RX Position Layout Generator
# =============================================================================
#
# AIM:
#   Generates a CSV file defining transmitter (TX) and receiver (RX) positions
#   for a given deployment scenario. Supports two deployment modes:
#
#     "box"      — TX positions are randomly sampled within a configured
#                  bounding box.
#
#     "polygon"  — TX positions are sampled across the entire scene footprint
#                  with building exclusion via pre-extracted footprints from
#                  extract_building_footprints.py. Run that script first to
#                  verify the footprints look correct.
#
#   Supports multiple base stations via config.deployment.base_stations.
#   Each BS is written as a separate RX row in the output CSV.
#
# INPUTS:
#   - Hydra config (from configs/scenarios/*.yaml)
#   - For polygon mode: building_footprints.npz in the scene directory
#     (produced by extract_building_footprints.py)
#
# OUTPUTS:
#   - CSV: configs/gps_coordinates/<selected_id>_tx_rx.csv
#   - PNG: configs/gps_coordinates/<selected_id>_tx_deployment.png
#
# USAGE:
#   python tx_rx_location_csv_gen.py --config-name=<scenario>
#
# DEPENDENCIES:
#   numpy, matplotlib, hydra, omegaconf
#
# =============================================================================

import os
import csv
import time
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────
#  Parse base_stations from config
# ─────────────────────────────────────────────────────────────

def parse_base_stations(cfg):
    """
    Read the base_stations list from the deployment config.

    Returns
    -------
    list[dict]
        Each dict has keys: 'name', 'position' (list[3]),
        'look_at' (list[3] or None).
    """
    bs_list = cfg.deployment.base_stations

    parsed = []
    for bs in bs_list:
        entry = {
            "name":     str(bs.name),
            "position": [float(x) for x in bs.position],
            "look_at":  [float(x) for x in bs.look_at] if bs.get("look_at") else None,
        }
        parsed.append(entry)

    if len(parsed) == 0:
        raise ValueError(
            "deployment.base_stations is empty. "
            "At least one base station must be specified."
        )

    print(f"  Base stations ({len(parsed)}):")
    for bs in parsed:
        la = f" -> look_at {bs['look_at']}" if bs["look_at"] else ""
        print(f"    {bs['name']:12s}  pos {bs['position']}{la}")

    return parsed


# ─────────────────────────────────────────────────────────────
#  Load pre-extracted building footprints (numpy + matplotlib)
# ─────────────────────────────────────────────────────────────

def load_footprints(scene_dir):
    """
    Load building footprint polygons from the .npz file produced by
    extract_building_footprints.py.

    Returns
    -------
    polygons : list of dict, each with:
        'vertices' : (V, 2) ndarray of XY polygon vertices
        'bbox'     : (x_min, y_min, x_max, y_max) for fast pre-filtering
        'path'     : matplotlib.path.Path for point-in-polygon tests
    """
    from matplotlib.path import Path

    npz_path = os.path.join(scene_dir, "building_footprints.npz")

    if not os.path.exists(npz_path):
        print(f"  building_footprints.npz not found in: {scene_dir}")
        print(f"  Run extract_building_footprints.py first.")
        return []

    data = np.load(npz_path, allow_pickle=True)
    num_buildings = int(data["num_buildings"][0])

    polygons = []
    for i in range(num_buildings):
        coords = data[f"poly_{i:04d}"]
        if len(coords) < 3:
            continue
        xy = coords[:, :2].astype(np.float64)
        try:
            path = Path(xy)
            bbox = (xy[:, 0].min(), xy[:, 1].min(),
                    xy[:, 0].max(), xy[:, 1].max())
            polygons.append({
                "vertices": xy,
                "bbox":     bbox,
                "path":     path,
            })
        except Exception:
            continue

    print(f"  Loaded {len(polygons)} building footprints from: {npz_path}")
    return polygons


# ─────────────────────────────────────────────────────────────
#  Polygon-based outdoor check (matplotlib.path, no Shapely)
# ─────────────────────────────────────────────────────────────

def batch_polygon_check(xs, ys, polygons):
    """
    Batch point-in-polygon check using matplotlib.path.Path with
    bounding-box pre-filtering for speed.

    Parameters
    ----------
    xs, ys   : 1D arrays of point coordinates
    polygons : list of dicts from load_footprints()

    Returns
    -------
    is_outdoor : bool array, True = point is outside all buildings
    """
    points = np.column_stack([xs, ys])
    is_outdoor = np.ones(len(xs), dtype=bool)

    for poly in polygons:
        bx_min, by_min, bx_max, by_max = poly["bbox"]

        # Bounding-box pre-filter: skip points clearly outside this building
        candidates = (
            is_outdoor
            & (xs >= bx_min) & (xs <= bx_max)
            & (ys >= by_min) & (ys <= by_max)
        )
        if not np.any(candidates):
            continue

        cand_idx = np.where(candidates)[0]
        inside = poly["path"].contains_points(points[cand_idx])
        is_outdoor[cand_idx[inside]] = False

    return is_outdoor


# ─────────────────────────────────────────────────────────────
#  Verification image (points overlaid on buildings)
# ─────────────────────────────────────────────────────────────

def save_deployment_image(tx_positions, polygons, output_path,
                          base_stations=None):
    """
    Save a top-down PNG: buildings in gray, TX points in green,
    each BS as a distinctly colored star.

    Parameters
    ----------
    tx_positions : np.ndarray  (N, 3)
    polygons     : list of dicts from load_footprints()
    output_path  : str
    base_stations : list[dict]  with 'name' and 'position' keys
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    # Buildings
    if polygons:
        patches = []
        for poly in polygons:
            try:
                patches.append(MplPolygon(poly["vertices"], closed=True))
            except Exception:
                continue
        if patches:
            pc = PatchCollection(patches, facecolor="#CCCCCC",
                                 edgecolor="#888888", linewidth=0.5,
                                 alpha=0.8, label="Buildings")
            ax.add_collection(pc)

    # TX points (subsample if too many)
    if len(tx_positions) > 0:
        if len(tx_positions) > 10000:
            idx = np.random.choice(len(tx_positions), 10000, replace=False)
            plot_pts = tx_positions[idx]
        else:
            plot_pts = tx_positions
        ax.scatter(plot_pts[:, 0], plot_pts[:, 1],
                   c="green", s=1.5, alpha=0.5,
                   label=f"TX ({len(tx_positions)})")

    # Base stations — one star per BS, distinct colors
    if base_stations:
        bs_colors = ["red", "blue", "orange", "purple", "cyan",
                     "magenta", "brown", "deeppink"]
        for i, bs in enumerate(base_stations):
            color = bs_colors[i % len(bs_colors)]
            pos = bs["position"]
            ax.plot(pos[0], pos[1],
                    marker="*", color=color, markersize=18,
                    markeredgecolor="black", markeredgewidth=0.8,
                    label=f"BS: {bs['name']}", zorder=10)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("TX Deployment — Top-Down View")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.autoscale_view()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Deployment image saved: {output_path}")


# ─────────────────────────────────────────────────────────────
#  Sampling strategies
# ─────────────────────────────────────────────────────────────

def sample_box(cfg, rng):
    """Box-mode sampler — uniform random within bounding box."""
    box = cfg.deployment.tx_box
    N = cfg.deployment.num_samples

    rand_x = rng.uniform(box.x_min, box.x_max, N)
    rand_y = rng.uniform(box.y_min, box.y_max, N)
    rand_z = rng.uniform(box.z_min, box.z_max, N)

    return np.stack([rand_x, rand_y, rand_z], axis=1)


def sample_scene_polygon(cfg, rng):
    """
    Scene-wide sampler with building exclusion via pre-extracted footprints.
    Requires extract_building_footprints.py to have been run first.
    """
    dep = cfg.deployment
    target_n = dep.num_samples
    tx_z = float(dep.tx_height)
    oversample = int(dep.get("building_exclusion", {}).get(
        "oversample_factor", 4))

    scene_dir = os.path.join(cfg.output_root, cfg.selected_id)

    # ---- Load pre-extracted footprints ----
    polygons = load_footprints(scene_dir)

    if not polygons:
        raise FileNotFoundError(
            "No building footprints found.\n"
            "Run extract_building_footprints.py first to generate them."
        )

    # ---- Scene bounding box from Mitsuba ----
    scene_path = os.path.join(scene_dir, "scene.xml")
    try:
        from sionna.rt import load_scene
        scene = load_scene(scene_path)
        bbox = scene.mi_scene.bbox()
        x_min = float(bbox.min.x)
        x_max = float(bbox.max.x)
        y_min = float(bbox.min.y)
        y_max = float(bbox.max.y)
    except Exception as e:
        # Fallback: compute bounds from all polygon vertices
        all_verts = np.concatenate([p["vertices"] for p in polygons], axis=0)
        margin_x = (all_verts[:, 0].max() - all_verts[:, 0].min()) * 0.1
        margin_y = (all_verts[:, 1].max() - all_verts[:, 1].min()) * 0.1
        x_min = all_verts[:, 0].min() - margin_x
        x_max = all_verts[:, 0].max() + margin_x
        y_min = all_verts[:, 1].min() - margin_y
        y_max = all_verts[:, 1].max() + margin_y
        print(f"  Could not load scene.xml ({e}), using polygon bounds.")

    print(f"  Sampling extent: X [{x_min:.1f}, {x_max:.1f}]  "
          f"Y [{y_min:.1f}, {y_max:.1f}]")
    print(f"  Using {len(polygons)} building footprints for exclusion.")

    # ---- Rejection sampling ----
    valid_positions = []
    total_tested = 0
    t0 = time.time()

    while len(valid_positions) < target_n:
        remaining = target_n - len(valid_positions)
        batch_n = min(remaining * oversample, 500_000)

        xs = rng.uniform(x_min, x_max, batch_n)
        ys = rng.uniform(y_min, y_max, batch_n)

        is_outdoor = batch_polygon_check(xs, ys, polygons)

        new_valid = np.column_stack([
            xs[is_outdoor],
            ys[is_outdoor],
            np.full(int(is_outdoor.sum()), tx_z),
        ])
        valid_positions.extend(new_valid.tolist())

        total_tested += batch_n
        elapsed = time.time() - t0
        rate = len(valid_positions) / max(elapsed, 0.01)
        print(f"  ... {len(valid_positions):,}/{target_n:,} valid  "
              f"({total_tested:,} tested, "
              f"{len(valid_positions)/max(total_tested,1):.1%} survival, "
              f"{rate:.0f} pts/s)")

    positions = np.array(valid_positions[:target_n])
    elapsed = time.time() - t0

    print(f"\nScene sampling (polygon) complete:")
    print(f"  {target_n:,} valid from {total_tested:,} candidates")
    print(f"  Survival rate: {target_n/total_tested:.1%}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return positions, polygons


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../../configs/scenarios")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    project_dir = cfg.project_dir
    loc_id = cfg.selected_id

    seed = cfg.get("ray_tracing_params", {}).get("seed", 42)
    rng = np.random.default_rng(seed)

    # ── Parse all base stations from config ──
    base_stations = parse_base_stations(cfg)
    K = len(base_stations)

    mode = cfg.deployment.get("mode", "box")
    polygons = []

    if mode == "box":
        print(f"\n[box mode] Generating {cfg.deployment.num_samples} "
              f"TX samples for: {loc_id}")
        tx_positions = sample_box(cfg, rng)

    elif mode == "polygon":
        print(f"\n[polygon mode] Targeting "
              f"{cfg.deployment.num_samples} valid TX samples for: {loc_id}")
        tx_positions, polygons = sample_scene_polygon(cfg, rng)

    else:
        raise ValueError(
            f"Unknown deployment.mode: '{mode}'. "
            f"Use 'box' or 'polygon'."
        )

    N = len(tx_positions)

    # ---- Output directory ----
    output_dir = os.path.join(project_dir, "configs", "gps_coordinates")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Save deployment image ----
    if mode == "polygon":
        if not polygons:
            scene_dir = os.path.join(cfg.output_root, cfg.selected_id)
            polygons = load_footprints(scene_dir)

        img_path = os.path.join(output_dir, f"{loc_id}_tx_deployment.png")
        save_deployment_image(
            tx_positions, polygons, img_path,
            base_stations=base_stations,
        )

    # ---- Write CSV ----
    csv_filename = os.path.join(output_dir, f"{loc_id}_tx_rx.csv")

    fieldnames = [
        "name", "type", "x", "y", "z",
        "look_at_x", "look_at_y", "look_at_z",
    ]

    with open(csv_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # ── TX rows ──
        for i, pos in enumerate(tx_positions):
            writer.writerow({
                "name": f"tx{i+1}",
                "type": "tx",
                "x": f"{pos[0]:.3f}",
                "y": f"{pos[1]:.3f}",
                "z": f"{pos[2]:.3f}",
                "look_at_x": "",
                "look_at_y": "",
                "look_at_z": "",
            })

        # ── RX rows: one per base station ──
        for bs in base_stations:
            pos = bs["position"]
            look = bs["look_at"]
            writer.writerow({
                "name":       bs["name"],
                "type":       "rx",
                "x":          f"{pos[0]:.3f}",
                "y":          f"{pos[1]:.3f}",
                "z":          f"{pos[2]:.3f}",
                "look_at_x":  f"{look[0]:.3f}" if look else "",
                "look_at_y":  f"{look[1]:.3f}" if look else "",
                "look_at_z":  f"{look[2]:.3f}" if look else "",
            })

    print(f"\nSUCCESS! {N} TX + {K} RX (base stations) "
          f"saved to: {csv_filename}")
    if mode == "polygon":
        print(f"Deployment image: "
              f"{os.path.join(output_dir, f'{loc_id}_tx_deployment.png')}")


if __name__ == "__main__":
    main()