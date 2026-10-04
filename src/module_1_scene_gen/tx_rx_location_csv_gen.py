# =============================================================================
# tx_rx_location_csv_gen.py — TX/RX Position Layout Generator
# =============================================================================
#
# AIM:
#   Generates a CSV file defining transmitter (TX) and receiver (RX) positions
#   for a given deployment scenario. Supports four deployment modes:
#
#     "box"          — TX positions are randomly sampled within a configured
#                      bounding box. No building exclusion.
#
#     "polygon"      — TX positions are sampled across the entire scene
#                      footprint with building exclusion via pre-extracted 2D
#                      footprints from extract_building_footprints.py. Run
#                      that script first to verify the footprints look correct.
#
#     "box_polygon"  — TX positions are sampled within the configured
#                      bounding box, then rejected against the same
#                      pre-extracted 2D footprints used by "polygon" mode.
#                      Use when you want box-restricted sampling with full
#                      building-footprint exclusion (not just rooftops).
#
#     "rooftop"      — TX positions sampled uniformly in the configured box,
#                      then filtered against rooftop footprints read directly
#                      from the scene's PLY mesh files. Height-aware: only
#                      roofs above the configured z_query height count as
#                      exclusion zones, so street-level sampling under
#                      elevated buildings is allowed. No pre-extraction step
#                      required.
#
#   Supports multiple base stations via config.deployment.base_stations.
#   Each BS is written as a separate RX row in the output CSV.
#
# INPUTS:
#   - Hydra config (from configs/scenarios/*.yaml)
#   - For polygon modes: footprints matching the configured buffer
#     (produced by extract_building_footprints.py)
#   - For rooftop mode: *_rooftop.ply files in data/scenes/<loc_id>/mesh/
#
# OUTPUTS:
#   - CSV: configs/gps_coordinates/<selected_id>_tx_rx.csv
#   - PNG: configs/gps_coordinates/<selected_id>_tx_deployment.png  (polygon mode)
#
# USAGE:
#   python tx_rx_location_csv_gen.py --config-name=<scenario>
#
# DEPENDENCIES:
#   numpy, matplotlib, hydra, omegaconf
#
# =============================================================================

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.scene_io import resolve_scene_path, footprint_buffer_m, buffered_footprint_path
import csv
import glob
import time
import struct
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

def load_footprints(scene_dir, buffer_m=None):
    """Load exclusion polygons, validating the requested buffer for sampling.

    Configured samplers always pass ``buffer_m``. Prefer the per-buffer cache;
    a legacy scene-level file is accepted only if its metadata matches. Never
    infer an unknown buffer or shrink a previously buffered polygon.

    Omitting ``buffer_m`` is for plotting existing layouts: use the legacy
    file, or a unique cache if there is no legacy file. Raw polygons are used
    for display when available.

    Returns dictionaries with vertices, vertices_raw, bbox, and a matplotlib
    Path for the buffered exclusion polygon, all in scene-local metres.
    """
    from matplotlib.path import Path

    legacy_path = os.path.join(scene_dir, "building_footprints.npz")
    regenerate = (
        "Run extract_building_footprints.py with the same scenario and buffer "
        "before generating the TX/RX CSV."
    )
    if buffer_m is not None:
        npz_path = buffered_footprint_path(scene_dir, buffer_m)
        buffer_m = float(buffer_m)
        if not os.path.isfile(npz_path):
            if os.path.isfile(legacy_path):
                npz_path = legacy_path
            else:
                raise FileNotFoundError(f"No footprints at {npz_path}. {regenerate}")
    else:
        npz_path = legacy_path
        if not os.path.isfile(npz_path):
            caches = sorted(glob.glob(os.path.join(
                scene_dir, "footprints", "buffer_*m", "building_footprints.npz")))
            if len(caches) > 1:
                raise ValueError("Multiple footprint caches found; specify buffer_m for plotting.")
            if not caches:
                print(f"  No building footprints found in: {scene_dir}")
                return []
            npz_path = caches[0]

    with np.load(npz_path, allow_pickle=False) as data:
        keys = set(data.files)
        npz_buffer_m = float(data["buffer_m"][0]) if "buffer_m" in keys else None
        if buffer_m is not None and (
            npz_buffer_m is None or not np.isfinite(npz_buffer_m)
            or not np.isclose(npz_buffer_m, buffer_m, rtol=0, atol=1e-9)
        ):
            raise ValueError(
                f"Footprint buffer mismatch: requested {buffer_m:g} m, "
                f"but {npz_path} records {npz_buffer_m!r} m. {regenerate}"
            )
        if npz_buffer_m is not None:
            print(f"  Footprint buffer: {npz_buffer_m:g} m")

        polygons = []
        for i in range(int(data["num_buildings"][0])):
            coords = data[f"poly_{i:04d}"]
            if len(coords) < 3:
                continue
            xy = coords[:, :2].astype(np.float64)
            raw_key = f"poly_{i:04d}_raw"
            xy_raw = data[raw_key][:, :2].astype(np.float64) if raw_key in keys else xy
            path = Path(xy)
            bbox = (xy[:, 0].min(), xy[:, 1].min(),
                    xy[:, 0].max(), xy[:, 1].max())
            polygons.append({"vertices": xy, "vertices_raw": xy_raw,
                             "bbox": bbox, "path": path})

    print(f"  Loaded {len(polygons)} building footprints from: {npz_path}")
    return polygons


# ─────────────────────────────────────────────────────────────
#  Polygon-based outdoor check 
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
#  PLY-based rooftop exclusion (height-aware, no extraction step)
# ─────────────────────────────────────────────────────────────

PLY_TO_STRUCT = {
    "char": "b",
    "uchar": "B",
    "int8": "b",
    "uint8": "B",
    "short": "h",
    "ushort": "H",
    "int16": "h",
    "uint16": "H",
    "int": "i",
    "uint": "I",
    "int32": "i",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def parse_ply_header(path):
    """Read the small PLY header so we know how to parse vertices and faces."""
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading PLY header: {path}")
            text = line.decode("ascii").strip()
            header_lines.append(text)
            if text == "end_header":
                break
        data_offset = f.tell()

    elements = []
    current = None
    for text in header_lines:
        parts = text.split()
        if not parts:
            continue
        if parts[0] == "element":
            current = {"name": parts[1], "count": int(parts[2]), "properties": []}
            elements.append(current)
        elif parts[0] == "property" and current is not None:
            if parts[1] == "list":
                current["properties"].append({
                    "kind": "list",
                    "count_type": parts[2],
                    "item_type": parts[3],
                    "name": parts[4],
                })
            else:
                current["properties"].append({
                    "kind": "scalar",
                    "type": parts[1],
                    "name": parts[2],
                })

    return data_offset, elements


def read_rooftop_ply(path):
    """Read vertices and polygon faces from a binary little-endian rooftop PLY."""
    data_offset, elements = parse_ply_header(path)
    vertices = None
    faces = []

    with open(path, "rb") as f:
        f.seek(data_offset)
        for element in elements:
            count = element["count"]
            properties = element["properties"]

            if element["name"] == "vertex":
                names = [prop["name"] for prop in properties]
                types = [prop["type"] for prop in properties]
                row_fmt = "<" + "".join(PLY_TO_STRUCT[t] for t in types)
                row_size = struct.calcsize(row_fmt)
                x_idx = names.index("x")
                y_idx = names.index("y")
                z_idx = names.index("z")

                vertices = np.empty((count, 3), dtype=float)
                for i in range(count):
                    row = struct.unpack(row_fmt, f.read(row_size))
                    vertices[i] = [row[x_idx], row[y_idx], row[z_idx]]

            elif element["name"] == "face":
                for _ in range(count):
                    face_indices = None
                    for prop in properties:
                        if prop["kind"] == "scalar":
                            scalar_fmt = "<" + PLY_TO_STRUCT[prop["type"]]
                            f.read(struct.calcsize(scalar_fmt))
                        else:
                            count_fmt = "<" + PLY_TO_STRUCT[prop["count_type"]]
                            item_count = struct.unpack(
                                count_fmt, f.read(struct.calcsize(count_fmt))
                            )[0]
                            item_fmt = "<" + PLY_TO_STRUCT[prop["item_type"]] * int(item_count)
                            values = struct.unpack(
                                item_fmt, f.read(struct.calcsize(item_fmt))
                            )
                            if prop["name"] == "vertex_indices":
                                face_indices = [int(v) for v in values]
                    if face_indices is not None and len(face_indices) >= 3:
                        faces.append(face_indices)

            else:
                skip_ply_element(f, count, properties)

    if vertices is None:
        raise ValueError(f"No vertices found in rooftop PLY: {path}")
    return vertices, faces


def skip_ply_element(file_obj, count, properties):
    """Skip PLY elements that are not needed by the TX sampler."""
    for _ in range(count):
        for prop in properties:
            if prop["kind"] == "scalar":
                scalar_fmt = "<" + PLY_TO_STRUCT[prop["type"]]
                file_obj.read(struct.calcsize(scalar_fmt))
            else:
                count_fmt = "<" + PLY_TO_STRUCT[prop["count_type"]]
                item_count = struct.unpack(
                    count_fmt, file_obj.read(struct.calcsize(count_fmt))
                )[0]
                item_fmt = "<" + PLY_TO_STRUCT[prop["item_type"]] * int(item_count)
                file_obj.read(struct.calcsize(item_fmt))


def load_rooftop_triangles(scene_dir, z_query_m):
    """Load rooftop footprint triangles whose roof is above the TX height."""
    mesh_dir = os.path.join(scene_dir, "mesh")
    rooftop_files = sorted(
        os.path.join(mesh_dir, name)
        for name in os.listdir(mesh_dir)
        if name.endswith("_rooftop.ply")
    )
    if not rooftop_files:
        raise FileNotFoundError(f"No rooftop PLY files found in: {mesh_dir}")

    triangles = []
    for path in rooftop_files:
        vertices, faces = read_rooftop_ply(path)
        for face in faces:
            for i in range(1, len(face) - 1):
                tri = vertices[[face[0], face[i], face[i + 1]]]
                if np.min(tri[:, 2]) >= z_query_m:
                    triangles.append(tri[:, :2])

    return triangles


def points_inside_triangle(points_xy, triangle_xy):
    """Return a boolean mask for points inside one 2D triangle."""
    x0, y0 = triangle_xy[0]
    x1, y1 = triangle_xy[1]
    x2, y2 = triangle_xy[2]
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denominator) < 1e-12:
        return np.zeros(len(points_xy), dtype=bool)

    x = points_xy[:, 0]
    y = points_xy[:, 1]
    a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denominator
    b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denominator
    c = 1.0 - a - b
    return (a >= 0.0) & (b >= 0.0) & (c >= 0.0)


def points_on_rooftop_mask(points_xy, rooftop_triangles):
    """Return True for points that land inside any rooftop footprint."""
    on_rooftop = np.zeros(len(points_xy), dtype=bool)
    for triangle in rooftop_triangles:
        remaining = ~on_rooftop
        if not np.any(remaining):
            break
        on_rooftop[remaining] = points_inside_triangle(points_xy[remaining], triangle)
    return on_rooftop


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

    # Buildings — draw the RAW (un-buffered) outlines so the image shows
    # the true building footprints, not the fattened exclusion polygons.
    # `vertices_raw` falls back to `vertices` for older .npz files.
    if polygons:
        patches = []
        for poly in polygons:
            try:
                patches.append(
                    MplPolygon(poly.get("vertices_raw", poly["vertices"]),
                               closed=True)
                )
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

def load_placement_terrain(cfg):
    """Keep existing scene-Z behavior unless terrain-relative placement is selected."""
    reference = cfg.deployment.get("height_reference", "scene")
    if reference == "scene":
        return None
    if reference != "terrain":
        raise ValueError("deployment.height_reference must be 'scene' or 'terrain'")
    mode = cfg.deployment.get("mode", "box")
    if mode not in ("polygon", "box_polygon"):
        raise ValueError("Terrain-relative placement requires polygon or box_polygon mode")
    from src.utils.terrain import TerrainSurface
    return TerrainSurface(resolve_scene_path(cfg), cfg.deployment.get("terrain_shape_ids"))


def place_above_terrain(positions, terrain):
    if terrain is None:
        return positions
    if not np.isfinite(positions[:, 2]).all() or (positions[:, 2] <= 0).any():
        raise ValueError("Terrain-relative TX heights must be finite and positive")
    ground = terrain.heights(positions[:, :2])
    positions = positions[np.isfinite(ground)].copy()
    positions[:, 2] += ground[np.isfinite(ground)]
    return positions


def sampling_batch_size(dep, remaining, tested, oversample):
    limit = int(dep.get("max_tx_sampling_draws", int(dep.num_samples) * 100))
    if int(dep.num_samples) <= 0 or oversample <= 0 or limit <= 0:
        raise ValueError("Sample count, oversample_factor and draw limit must be positive")
    if tested >= limit:
        raise RuntimeError(
            f"Only accepted {int(dep.num_samples) - remaining}/{dep.num_samples} TX points "
            f"after {tested} draws; check outdoor area and terrain coverage, or "
            "increase max_tx_sampling_draws."
        )
    return min(remaining * oversample, 500_000, limit - tested)


def sample_box(cfg, rng):
    """Box-mode sampler — uniform random within bounding box."""
    load_placement_terrain(cfg)
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
    terrain = load_placement_terrain(cfg)
    target_n = dep.num_samples
    if target_n <= 0:
        raise ValueError("deployment.num_samples must be positive")
    tx_z = float(dep.tx_height)
    oversample = int(dep.get("building_exclusion", {}).get(
        "oversample_factor", 4))

    scene_path = resolve_scene_path(cfg)
    scene_dir = os.path.dirname(scene_path)

    # ---- Load pre-extracted footprints ----
    polygons = load_footprints(scene_dir, buffer_m=footprint_buffer_m(cfg))

    if not polygons:
        raise FileNotFoundError(
            "No building footprints found.\n"
            "Run extract_building_footprints.py first to generate them."
        )

    if terrain is not None:
        x_min, y_min = terrain.bounds[:, [0, 2]].min(axis=0)
        x_max, y_max = terrain.bounds[:, [1, 3]].max(axis=0)
    else:
        # ---- Scene bounding box from Mitsuba ----
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
            print(f"  Could not load {scene_path} ({e}), using polygon bounds.")

    print(f"  Sampling extent: X [{x_min:.1f}, {x_max:.1f}]  "
          f"Y [{y_min:.1f}, {y_max:.1f}]")
    print(f"  Using {len(polygons)} building footprints for exclusion.")

    # ---- Rejection sampling ----
    valid_positions = []
    total_tested = 0
    t0 = time.time()

    while len(valid_positions) < target_n:
        remaining = target_n - len(valid_positions)
        batch_n = sampling_batch_size(dep, remaining, total_tested, oversample)

        xs = rng.uniform(x_min, x_max, batch_n)
        ys = rng.uniform(y_min, y_max, batch_n)

        is_outdoor = batch_polygon_check(xs, ys, polygons)

        new_valid = np.column_stack([
            xs[is_outdoor],
            ys[is_outdoor],
            np.full(int(is_outdoor.sum()), tx_z),
        ])
        new_valid = place_above_terrain(new_valid, terrain)
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


def sample_box_polygon(cfg, rng):
    """
    Box-bounded sampler with building exclusion via pre-extracted 2D footprints.
    Samples uniformly within tx_box and rejects points inside any building
    polygon loaded from building_footprints.npz.

    Requires extract_building_footprints.py to have been run first.
    """
    dep = cfg.deployment
    box = dep.tx_box
    terrain = load_placement_terrain(cfg)
    target_n = dep.num_samples
    if target_n <= 0:
        raise ValueError("deployment.num_samples must be positive")
    oversample = int(dep.get("building_exclusion", {}).get(
        "oversample_factor", 4))

    scene_dir = os.path.dirname(resolve_scene_path(cfg))
    polygons = load_footprints(scene_dir, buffer_m=footprint_buffer_m(cfg))

    if not polygons:
        raise FileNotFoundError(
            "No building footprints found.\n"
            "Run extract_building_footprints.py first to generate them."
        )

    print(f"  Sampling extent: X [{box.x_min:.1f}, {box.x_max:.1f}]  "
          f"Y [{box.y_min:.1f}, {box.y_max:.1f}]  "
          f"Z [{box.z_min:.1f}, {box.z_max:.1f}]")
    print(f"  Using {len(polygons)} building footprints for exclusion.")

    valid_positions = []
    total_tested = 0
    t0 = time.time()

    while len(valid_positions) < target_n:
        remaining = target_n - len(valid_positions)
        batch_n = sampling_batch_size(dep, remaining, total_tested, oversample)

        xs = rng.uniform(box.x_min, box.x_max, batch_n)
        ys = rng.uniform(box.y_min, box.y_max, batch_n)
        zs = rng.uniform(box.z_min, box.z_max, batch_n)

        is_outdoor = batch_polygon_check(xs, ys, polygons)

        new_valid = np.column_stack([
            xs[is_outdoor], ys[is_outdoor], zs[is_outdoor],
        ])
        new_valid = place_above_terrain(new_valid, terrain)
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

    print(f"\nBox sampling (with polygon exclusion) complete:")
    print(f"  {target_n:,} valid from {total_tested:,} candidates")
    print(f"  Survival rate: {target_n/total_tested:.1%}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return positions, polygons


def sample_scene_rooftop(cfg, rng):
    """
    Box-bounded sampler with rooftop exclusion via direct PLY mesh reading.
    Height-aware: only roofs above z_query_m count as exclusion zones, so
    street-level points under elevated buildings are accepted.

    Reads *_rooftop.ply files from the resolved scene's mesh/ directory.

    No pre-extraction step required.
    """
    load_placement_terrain(cfg)
    dep = cfg.deployment
    scene_dir = os.path.dirname(resolve_scene_path(cfg))
    box = dep.tx_box
    num_samples = dep.num_samples

    z_query_m = float(dep.get("rooftop_mask_z_m", box.z_min))
    rooftop_triangles = load_rooftop_triangles(scene_dir, z_query_m)
    print(
        f"  Rejecting TX points inside {len(rooftop_triangles)} rooftop "
        f"triangles at z >= {z_query_m:.1f} m"
    )

    accepted = []
    total_draws = 0
    batch_size = max(num_samples, 1000)
    max_draws = int(dep.get("max_tx_sampling_draws", num_samples * 100))

    while sum(len(batch) for batch in accepted) < num_samples:
        if total_draws >= max_draws:
            num_accepted = sum(len(batch) for batch in accepted)
            raise RuntimeError(
                f"Only accepted {num_accepted}/{num_samples} TX points after "
                f"{total_draws} draws. Increase max_tx_sampling_draws or widen tx_box."
            )

        remaining_draws = max_draws - total_draws
        current_batch_size = min(batch_size, remaining_draws)

        # Use the seeded RNG for reproducibility
        rand_x = rng.uniform(box.x_min, box.x_max, current_batch_size)
        rand_y = rng.uniform(box.y_min, box.y_max, current_batch_size)
        rand_z = rng.uniform(box.z_min, box.z_max, current_batch_size)
        candidates = np.stack([rand_x, rand_y, rand_z], axis=1)
        total_draws += current_batch_size

        rejected = points_on_rooftop_mask(candidates[:, :2], rooftop_triangles)
        accepted.append(candidates[~rejected])

    tx_positions = np.concatenate(accepted, axis=0)[:num_samples]
    acceptance_rate = 100.0 * num_samples / max(total_draws, 1)
    print(
        f"  Accepted {num_samples} TX samples after {total_draws} draws "
        f"({acceptance_rate:.1f}% acceptance rate)"
    )
    return tx_positions


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

    elif mode == "box_polygon":
        print(f"\n[box_polygon mode] Targeting "
              f"{cfg.deployment.num_samples} valid TX samples for: {loc_id}")
        tx_positions, polygons = sample_box_polygon(cfg, rng)

    elif mode == "rooftop":
        print(f"\n[rooftop mode] Generating {cfg.deployment.num_samples} "
              f"TX samples with PLY-rooftop exclusion for: {loc_id}")
        tx_positions = sample_scene_rooftop(cfg, rng)

    else:
        raise ValueError(
            f"Unknown deployment.mode: '{mode}'. "
            f"Use 'box', 'polygon', 'box_polygon', or 'rooftop'."
        )

    N = len(tx_positions)

    # ---- Output directory ----
    output_dir = os.path.join(project_dir, "configs", "gps_coordinates")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Save deployment image (modes that use footprints) ----
    if mode in ("polygon", "box_polygon"):
        if not polygons:
            scene_dir = os.path.dirname(resolve_scene_path(cfg))
            polygons = load_footprints(scene_dir, buffer_m=footprint_buffer_m(cfg))

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