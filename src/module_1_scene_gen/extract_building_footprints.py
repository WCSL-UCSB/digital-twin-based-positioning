# =============================================================================
# extract_building_footprints.py — Building Footprint Extraction & Validation
# =============================================================================
#
# AIM:
#   Reads the rooftop .ply meshes produced by scenegen, projects their
#   vertices onto the XY plane (ground level) to get 2D building footprints,
#   and saves a top-down verification image.
#
#   Run this AFTER run_g2sm.py and BEFORE tx_rx_location_csv_gen.py.
#
# HOW IT WORKS:
#   Each building_N_rooftop.ply contains the 3D rooftop mesh. Projecting
#   all its vertices to XY (dropping Z) and taking the convex hull gives
#   the ground-level building footprint — the exact area to exclude when
#   sampling outdoor TX positions.
#
# OUTPUTS (in the scene directory):
#   - building_footprints.npz   — polygon vertices for each building
#   - building_footprints.png   — top-down verification image
#
# USAGE:
#   python extract_building_footprints.py --config-name=<scenario>
#
# =============================================================================

import os
import glob
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────
#  Footprint extraction from rooftop meshes
# ─────────────────────────────────────────────────────────────

def extract_footprints(scene_dir):
    """
    Extract 2D building footprints by projecting rooftop meshes onto XY.

    Reads only *_rooftop.ply files from the mesh/ subdirectory. For each
    rooftop, projects all vertices to the XY plane and computes the 2D
    convex hull — this is the ground-level building footprint.

    Returns:
        polygons: list of Shapely Polygon objects in scene-local coords.
        building_names: list of building name strings (parallel to polygons).
    """
    from shapely.geometry import MultiPoint
    from plyfile import PlyData

    mesh_dir = os.path.join(scene_dir, "mesh")
    ply_files = sorted(glob.glob(os.path.join(mesh_dir, "*.ply")))

    if not ply_files:
        print(f"  No .ply files found in: {mesh_dir}")
        return [], []

    print(f"  Found {len(ply_files)} .ply files in mesh directory.")

    polygons = []
    names = []

    for ply_path in ply_files:
        fname = os.path.basename(ply_path).lower()

        # Only use rooftop meshes
        if not fname.endswith("_rooftop.ply"):
            continue

        # Building name: "building_0_rooftop.ply" → "building_0"
        building_name = os.path.splitext(fname)[0].replace("_rooftop", "")

        try:
            plydata = PlyData.read(ply_path)
            vx = np.array(plydata['vertex']['x'])
            vy = np.array(plydata['vertex']['y'])
            print(f"    [load] {fname} → {building_name} "
                  f"({len(vx)} vertices)")
        except Exception as e:
            print(f"    [error] {fname}: {e}")
            continue

        if len(vx) < 3:
            print(f"    [skip] {building_name}: fewer than 3 vertices")
            continue

        # Project to XY and take convex hull → ground-level footprint
        xy = np.column_stack([vx, vy])
        try:
            hull = MultiPoint(xy).convex_hull
            if hull.is_empty or not hull.is_valid:
                continue
            if hull.geom_type == "Polygon":
                polygons.append(hull)
                names.append(building_name)
                print(f"    [hull] {building_name}: "
                      f"{hull.area:.1f} m² footprint")
        except Exception as e:
            print(f"    [error] {building_name}: {e}")
            continue

    print(f"\n  Total: {len(polygons)} building footprints extracted.")
    return polygons, names


# ─────────────────────────────────────────────────────────────
#  Save footprints
# ─────────────────────────────────────────────────────────────

def save_footprints(polygons, building_names, output_path):
    """
    Save building footprint polygons as vertex arrays (.npz).
    Each entry is an Nx2 array of (x, y) exterior ring coordinates.
    """
    poly_data = {}
    for i, (poly, name) in enumerate(zip(polygons, building_names)):
        coords = np.array(poly.exterior.coords)
        poly_data[f"poly_{i:04d}"] = coords
        poly_data[f"name_{i:04d}"] = np.array([name])

    poly_data["num_buildings"] = np.array([len(polygons)])
    np.savez(output_path, **poly_data)
    print(f"  Saved {len(polygons)} footprints to: {output_path}")


# ─────────────────────────────────────────────────────────────
#  Verification image
# ─────────────────────────────────────────────────────────────

def save_verification_image(polygons, building_names, output_path,
                            rx_position=None, scene_bbox=None):
    """
    Save a top-down PNG showing building footprints.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    # Draw buildings
    patches = []
    for poly in polygons:
        try:
            exterior = np.array(poly.exterior.coords)
            patches.append(MplPolygon(exterior[:, :2], closed=True))
        except Exception:
            continue

    if patches:
        pc = PatchCollection(patches, facecolor="#CCCCCC",
                             edgecolor="#444444", linewidth=0.8,
                             alpha=0.7)
        ax.add_collection(pc)

    # Label buildings at centroid
    for poly, name in zip(polygons, building_names):
        cx, cy = poly.centroid.x, poly.centroid.y
        ax.text(cx, cy, name, fontsize=5, ha="center", va="center",
                color="#333333", alpha=0.8)

    # Scene bounding box
    if scene_bbox is not None:
        x_min, x_max, y_min, y_max = scene_bbox
        bbox_rect = MplPolygon(
            [[x_min, y_min], [x_max, y_min],
             [x_max, y_max], [x_min, y_max]],
            closed=True, fill=False, edgecolor="blue",
            linewidth=1.5, linestyle="--", label="Scene Bbox"
        )
        ax.add_patch(bbox_rect)

    # RX position
    if rx_position is not None:
        ax.plot(rx_position[0], rx_position[1],
                marker="*", color="red", markersize=18,
                markeredgecolor="darkred", markeredgewidth=1.0,
                label="RX (Base Station)", zorder=10)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Building Footprints (from rooftop projection) — "
                 f"{len(polygons)} buildings")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.autoscale_view()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Verification image saved: {output_path}")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../../configs/scenarios")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    loc_id = cfg.selected_id
    scene_dir = os.path.join(cfg.output_root, loc_id)

    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(
            f"Scene directory not found: {scene_dir}\n"
            f"Run run_g2sm.py first."
        )

    print(f"=== Building Footprint Extraction: {loc_id} ===")
    print(f"Scene directory: {scene_dir}\n")

    # ---- Extract footprints from rooftop meshes ----
    polygons, names = extract_footprints(scene_dir)

    if not polygons:
        print("\nNo building footprints found. "
              "Check mesh/ directory for *_rooftop.ply files.")
        return

    # ---- Get scene bounding box ----
    scene_bbox = None
    scene_path = os.path.join(scene_dir, "scene.xml")
    if os.path.exists(scene_path):
        try:
            from sionna.rt import load_scene
            scene = load_scene(scene_path)
            bbox = scene.mi_scene.bbox()
            scene_bbox = (
                float(bbox.min.x), float(bbox.max.x),
                float(bbox.min.y), float(bbox.max.y),
            )
            print(f"\n  Scene bbox: "
                  f"X [{scene_bbox[0]:.1f}, {scene_bbox[1]:.1f}]  "
                  f"Y [{scene_bbox[2]:.1f}, {scene_bbox[3]:.1f}]")
        except Exception as e:
            print(f"\n  Could not load scene.xml for bbox: {e}")

    # ---- RX position for reference ----
    rx_pos = None
    if hasattr(cfg, "deployment") and hasattr(cfg.deployment, "rx_position"):
        rx_pos = list(cfg.deployment.rx_position)

    # ---- Save footprints ----
    footprint_path = os.path.join(scene_dir, "building_footprints.npz")
    save_footprints(polygons, names, footprint_path)

    # ---- Save verification image ----
    img_path = os.path.join(scene_dir, "building_footprints.png")
    save_verification_image(
        polygons, names, img_path,
        rx_position=rx_pos, scene_bbox=scene_bbox,
    )

    # ---- Summary ----
    total_area = sum(p.area for p in polygons)
    if scene_bbox:
        scene_area = ((scene_bbox[1] - scene_bbox[0])
                      * (scene_bbox[3] - scene_bbox[2]))
        coverage = total_area / scene_area * 100
        print(f"\n  Building coverage: {total_area:.0f} m² / "
              f"{scene_area:.0f} m² = {coverage:.1f}%")
        print(f"  Expected outdoor survival rate: ~{100 - coverage:.0f}%")

    print(f"\n=== Done. Check: {img_path} ===")


if __name__ == "__main__":
    main()