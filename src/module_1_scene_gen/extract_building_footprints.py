# =============================================================================
# extract_building_footprints.py — Building Footprint Extraction & Validation
# =============================================================================
#
# AIM:
#   Produces a building_footprints.npz / .png pair in the scene directory.
#   Auto-detects the scene source and uses the right extraction strategy:
#
#     * BostonTwin scenes (presence of <selected_id>.geojson + tileinfo) →
#       read polygon footprints directly from the GeoJSON, project from
#       WGS84 to local CRS metres, subtract the scene centre. This is
#       authoritative (BPDA ground footprints) and avoids the trap of
#       projecting BostonTwin's sophisticated rooftop meshes (which
#       include HVAC, parapets, overhangs, recessed setbacks) down to
#       XY — that projection does NOT match the building's ground
#       footprint.
#
#     * OSM / G2SM scenes (presence of mesh/*_rooftop.ply) →
#       project each rooftop mesh's vertices to XY and take the convex
#       hull. Original behaviour; correct for OSM-built scenes which
#       use simple extruded footprints with no roof clutter.
#
#   Run this AFTER stage-1 scene generation (run_bostontwin.py or
#   run_g2sm.py) and BEFORE tx_rx_location_csv_gen.py.
#
#   EXCLUSION BUFFER:
#     Each polygon is grown by a small margin before being saved, so the
#     downstream point-in-polygon test treats a thin strip around each
#     building as off-limits for TX sampling. Default is 1.0 m. This
#     compensates for BostonTwin's polygon-vs-3D-mesh thickness gap
#     (mesh extrudes slightly past the BPDA polygon → TX samples just
#     outside the polygon visually land on/inside the rendered wall).
#     Override via the scenario YAML:
#         deployment:
#           building_exclusion:
#             buffer_m: 1.0      # set to 0.0 to disable buffering
#
# OUTPUTS (in the scene directory):
#   - building_footprints.npz   — polygon vertices for each building
#                                 (scene-local metres, Nx2 arrays;
#                                 includes the exclusion buffer)
#   - building_footprints.png   — top-down verification image
#
# USAGE:
#   python extract_building_footprints.py --config-name=<scenario>
#
# =============================================================================

import os
import glob
import json
import numpy as np
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────
#  BostonTwin local CRS (feet) — same projection used by the
#  scene-gen wrapper for translating polygon centroids.
# ─────────────────────────────────────────────────────────────

_FT2M = 0.3048
_PRJ_PATH = (
    Path(__file__).resolve().parents[2]
    / "third_party" / "boston_twin" / "data" / "Metro_Boston_3D_CRS.prj"
)


# ─────────────────────────────────────────────────────────────
#  BostonTwin path: read polygons from GeoJSON
# ─────────────────────────────────────────────────────────────

def _geom_rings(geom):
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    if geom["type"] == "MultiPolygon":
        return [p[0] for p in geom["coordinates"]]
    raise ValueError(f"unsupported geometry type: {geom['type']}")


def extract_footprints_bostontwin(scene_dir, scene_name, buffer_m=0.0):
    """
    Read polygon footprints from <scene_name>.geojson and project them to
    scene-local metres using <scene_name>_tileinfo.geojson for the centre.

    If `buffer_m > 0`, each polygon is grown by `buffer_m` metres before being
    saved. This is the recommended setting for TX-exclusion use because the
    BostonTwin 3D PLY mesh for a building can extend slightly past its
    GeoJSON polygon (parapets, eaves, wall thickness), and TX positions
    sampled inside that gap land visually on the mesh even though they're
    correctly outside the polygon. 1.0 m is a reasonable default.

    Returns (polygons, building_names) just like extract_footprints(), so the
    rest of the script (save_footprints / save_verification_image) is
    unchanged. MultiPolygon features are emitted as multiple entries with
    `_part0`, `_part1`, ... suffixes on the name.
    """
    from shapely.geometry import Polygon
    from pyproj import CRS, Transformer

    gj_path = os.path.join(scene_dir, f"{scene_name}.geojson")
    ti_path = os.path.join(scene_dir, f"{scene_name}_tileinfo.geojson")
    for p in (gj_path, ti_path):
        if not os.path.isfile(p):
            print(f"  [skip] BostonTwin path needs {p}; not found.")
            return [], [], []

    with open(ti_path) as f:
        ti = json.load(f)
    props = ti["features"][0]["properties"]
    sc_x_m = float(props["Centr_X_m"])
    sc_y_m = float(props["Centr_Y_m"])
    print(f"  Scene centre (local m): ({sc_x_m:.3f}, {sc_y_m:.3f})")

    local_crs = CRS.from_wkt(_PRJ_PATH.read_text().strip())
    to_local = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)

    with open(gj_path) as f:
        gj = json.load(f)
    print(f"  Reading {len(gj['features'])} features from "
          f"{os.path.basename(gj_path)}...")

    polygons = []        # buffered (what TX exclusion uses)
    raw_polygons = []    # un-buffered (the actual BPDA ground footprint)
    names = []
    n_skipped_meta = 0
    n_skipped_status = 0
    n_skipped_no_ground = 0
    for feat in gj["features"]:
        props = feat["properties"]
        mid = props.get("Model_ID")
        if mid is None:
            n_skipped_meta += 1
            continue
        # Mirror BostonTwin's generate_scene_from_model_gdf filters
        # (BostonModel.py:234-237) so footprints match the buildings that
        # actually exist in the 3D scene. Without these filters, "History"
        # features (demolished buildings) and "Under Construction" features
        # overlap with current buildings on the same parcel.
        if props.get("Status") != "Current":
            n_skipped_status += 1
            continue
        if not props.get("Gnd_El_Ft"):
            n_skipped_no_ground += 1
            continue
        try:
            rings = _geom_rings(feat["geometry"])
        except ValueError:
            n_skipped_meta += 1
            continue

        for j, ring in enumerate(rings):
            coords_m = []
            for lon, lat in ring:
                x_ft, y_ft = to_local.transform(lon, lat)
                coords_m.append([x_ft * _FT2M - sc_x_m,
                                 y_ft * _FT2M - sc_y_m])
            if len(coords_m) < 4:  # need at least 3 unique + 1 closing
                continue
            try:
                raw_poly = Polygon(coords_m)
                if not raw_poly.is_valid:
                    raw_poly = raw_poly.buffer(0)
                if raw_poly.is_empty or raw_poly.area < 1e-3:
                    continue
                if raw_poly.geom_type != "Polygon":
                    # buffer(0) can turn into MultiPolygon; pick the largest part
                    raw_poly = max(raw_poly.geoms, key=lambda g: g.area)
                if buffer_m > 0:
                    poly = raw_poly.buffer(buffer_m)
                    if poly.is_empty:
                        continue
                    if poly.geom_type != "Polygon":
                        poly = max(poly.geoms, key=lambda g: g.area)
                else:
                    poly = raw_poly
            except Exception:
                continue
            suffix = "" if len(rings) == 1 else f"_part{j}"
            polygons.append(poly)
            raw_polygons.append(raw_poly)
            names.append(f"{mid}{suffix}")

    buf_msg = f" (buffered by {buffer_m:.2f} m)" if buffer_m > 0 else ""
    print(f"  Extracted {len(polygons)} polygons{buf_msg}  "
          f"(skipped {n_skipped_status} non-Current, "
          f"{n_skipped_no_ground} without ground elevation, "
          f"{n_skipped_meta} without Model_ID / invalid geometry)")
    return polygons, raw_polygons, names


# ─────────────────────────────────────────────────────────────
#  OSM / G2SM path: project rooftop meshes
# ─────────────────────────────────────────────────────────────

def extract_footprints(scene_dir, buffer_m=0.0):
    """
    Extract 2D building footprints by projecting rooftop meshes onto XY.

    Reads only *_rooftop.ply files from the mesh/ subdirectory. For each
    rooftop, projects all vertices to the XY plane and computes the 2D
    convex hull — this is the ground-level building footprint.

    If `buffer_m > 0`, each hull is grown by `buffer_m` metres before being
    returned, so TX-exclusion includes a small margin around each building.

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
        return [], [], []

    print(f"  Found {len(ply_files)} .ply files in mesh directory.")

    polygons = []        # buffered (what TX exclusion uses)
    raw_polygons = []    # un-buffered (rooftop-mesh convex hull)
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
            raw_hull = MultiPoint(xy).convex_hull
            if raw_hull.is_empty or not raw_hull.is_valid:
                continue
            if raw_hull.geom_type != "Polygon":
                continue
            if buffer_m > 0:
                hull = raw_hull.buffer(buffer_m)
                if hull.is_empty:
                    continue
                if hull.geom_type != "Polygon":
                    hull = max(hull.geoms, key=lambda g: g.area)
            else:
                hull = raw_hull
            polygons.append(hull)
            raw_polygons.append(raw_hull)
            names.append(building_name)
            print(f"    [hull] {building_name}: "
                  f"{raw_hull.area:.1f} m² footprint")
        except Exception as e:
            print(f"    [error] {building_name}: {e}")
            continue

    buf_msg = f" (buffered by {buffer_m:.2f} m)" if buffer_m > 0 else ""
    print(f"\n  Total: {len(polygons)} building footprints extracted{buf_msg}.")
    return polygons, raw_polygons, names


# ─────────────────────────────────────────────────────────────
#  Save footprints
# ─────────────────────────────────────────────────────────────

def save_footprints(polygons, raw_polygons, building_names, output_path,
                    buffer_m=0.0):
    """
    Save building footprint polygons as vertex arrays (.npz).

    Schema (each polygon's exterior coords as Nx2 (x, y) arrays):
        poly_NNNN         buffered polygon (what TX exclusion reads)
        poly_NNNN_raw     un-buffered polygon (true BPDA / mesh footprint)
        name_NNNN         1-element string array
        num_buildings     1-element int array
        buffer_m          1-element float array — metadata recording the
                          buffer that was applied (so downstream can detect
                          stale .npz vs. current YAML setting).
    """
    poly_data = {}
    for i, (poly, raw_poly, name) in enumerate(
        zip(polygons, raw_polygons, building_names)
    ):
        poly_data[f"poly_{i:04d}"] = np.array(poly.exterior.coords)
        poly_data[f"poly_{i:04d}_raw"] = np.array(raw_poly.exterior.coords)
        poly_data[f"name_{i:04d}"] = np.array([name])

    poly_data["num_buildings"] = np.array([len(polygons)])
    poly_data["buffer_m"] = np.array([float(buffer_m)])
    np.savez(output_path, **poly_data)
    print(f"  Saved {len(polygons)} footprints (buffered + raw, "
          f"buffer={buffer_m:.2f} m) to: {output_path}")


# ─────────────────────────────────────────────────────────────
#  Verification image
# ─────────────────────────────────────────────────────────────

def save_verification_image(polygons, building_names, output_path,
                            rx_position=None, scene_bbox=None,
                            source_label="rooftop projection"):
    """
    Save a top-down PNG showing building footprints.
    `source_label` is used in the figure title (e.g. "BostonTwin GeoJSON"
    vs the default "rooftop projection") so the image documents how the
    footprints were derived.
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
    ax.set_title(f"Building Footprints (from {source_label}) — "
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

def _detect_scene_source(scene_dir, scene_name):
    """Return 'bostontwin' or 'mesh' based on what files are in scene_dir.

    BostonTwin scenes ship <selected_id>.geojson + <selected_id>_tileinfo.geojson
    alongside the XML. OSM/G2SM scenes ship a `mesh/` subfolder with
    *_rooftop.ply files. If neither pattern matches we return None and the
    caller decides what to do.
    """
    gj = os.path.join(scene_dir, f"{scene_name}.geojson")
    ti = os.path.join(scene_dir, f"{scene_name}_tileinfo.geojson")
    if os.path.isfile(gj) and os.path.isfile(ti):
        return "bostontwin"
    if glob.glob(os.path.join(scene_dir, "mesh", "*_rooftop.ply")):
        return "mesh"
    return None


@hydra.main(version_base=None, config_path="../../configs/scenarios")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    loc_id = cfg.selected_id
    scene_dir = os.path.join(cfg.output_root, loc_id)

    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(
            f"Scene directory not found: {scene_dir}\n"
            f"Run a stage-1 generator first (run_bostontwin.py or run_g2sm.py)."
        )

    # Exclusion buffer (metres) — grow each polygon before saving so TX
    # sampling stays a small margin clear of building edges. This compensates
    # for the BostonTwin polygon-vs-3D-mesh thickness gap and avoids TX
    # samples landing visually on walls in the rendered scene.
    buffer_m =1
    if hasattr(cfg, "deployment") and hasattr(cfg.deployment, "building_exclusion"):
        be = cfg.deployment.building_exclusion
        if "buffer_m" in be:
            buffer_m = float(be.buffer_m)

    print(f"=== Building Footprint Extraction: {loc_id} ===")
    print(f"Scene directory: {scene_dir}")
    print(f"Polygon exclusion buffer: {buffer_m:.2f} m\n")

    source = _detect_scene_source(scene_dir, loc_id)
    if source == "bostontwin":
        print(f"  Source: BostonTwin (reading polygons from GeoJSON)")
        polygons, raw_polygons, names = extract_footprints_bostontwin(
            scene_dir, loc_id, buffer_m=buffer_m,
        )
    elif source == "mesh":
        print(f"  Source: OSM/G2SM (projecting rooftop meshes)")
        polygons, raw_polygons, names = extract_footprints(
            scene_dir, buffer_m=buffer_m,
        )
    else:
        raise FileNotFoundError(
            f"Could not detect scene source in {scene_dir}.\n"
            f"Expected either:\n"
            f"  - {loc_id}.geojson + {loc_id}_tileinfo.geojson  (BostonTwin)\n"
            f"  - mesh/*_rooftop.ply files                      (OSM/G2SM)"
        )

    if not polygons:
        print("\nNo building footprints found.")
        return

    # ---- Get scene bounding box (best-effort) ----
    scene_bbox = None
    # BostonTwin writes XML as <loc_id>.xml; G2SM writes scene.xml.
    for fname in (f"{loc_id}.xml", "scene.xml"):
        scene_path = os.path.join(scene_dir, fname)
        if not os.path.exists(scene_path):
            continue
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
            print(f"\n  Could not load {fname} for bbox: {e}")
        break

    # ---- RX position for reference ----
    rx_pos = None
    if hasattr(cfg, "deployment") and hasattr(cfg.deployment, "rx_position"):
        rx_pos = list(cfg.deployment.rx_position)

    # ---- Save footprints (both buffered and raw, plus buffer metadata) ----
    footprint_path = os.path.join(scene_dir, "building_footprints.npz")
    save_footprints(polygons, raw_polygons, names, footprint_path,
                    buffer_m=buffer_m)

    # ---- Save verification image (use RAW polygons so the buffer halo
    #      doesn't obscure the real building outlines in the PNG) ----
    img_path = os.path.join(scene_dir, "building_footprints.png")
    source_label = ("BostonTwin GeoJSON" if source == "bostontwin"
                    else "rooftop projection")
    save_verification_image(
        raw_polygons, names, img_path,
        rx_position=rx_pos, scene_bbox=scene_bbox,
        source_label=source_label,
    )

    # ---- Summary (areas reported on the buffered polygons, since those
    #      are what TX exclusion actually uses) ----
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