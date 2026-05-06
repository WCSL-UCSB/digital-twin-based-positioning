# =============================================================================
# tx_rx_location_csv_gen.py — TX/RX Position Layout Generator
# =============================================================================
#
# AIM:
#   Generates a CSV file defining transmitter (TX) and receiver (RX) positions
#   for a given deployment scenario. TX positions are randomly sampled within
#   a configured bounding box, while the RX (base station) is placed at a
#   fixed location. This layout is used by downstream modules (path_gen,
#   ray tracing) to simulate the wireless channel between each TX-RX pair.
#
# INPUTS:
#   - Hydra config (from configs/scenarios/*.yaml):
#       * selected_id              : location key (e.g. "nyc_high_rise")
#       * deployment.num_samples   : number of TX positions to generate
#       * deployment.tx_box        : bounding box for random TX placement
#                                    (x_min/max, y_min/max, z_min/max)
#       * deployment.base_stations : list of base stations, each with:
#                                      - name    : label for this BS (e.g. "bs_0")
#                                      - position: fixed RX coordinates [x, y, z]
#                                      - look_at : RX look-at direction [x, y, z]
#
# OUTPUTS:
#   - CSV file written to:
#       data/gps_coordinates/<selected_id>_tx_rx.csv
#     Columns: name, type, x, y, z, look_at_x, look_at_y, look_at_z
#     Contains N TX rows (tx1, tx2, ...) + one RX row per base station.
#
# CONNECTED DIRECTORIES:
#   - configs/scenarios/       — Hydra config files defining deployment params
#   - data/gps_coordinates/    — Output CSV consumed by path_gen.py and
#                                visualize_deployment.py in module_1
#
# USAGE:
#   python tx_rx_location_csv_gen.py --config-name=<scenario>
#   e.g.: python tx_rx_location_csv_gen.py --config-name=test0217
#
# =============================================================================

import os
import csv
import struct
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

load_dotenv()

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


def load_rooftop_triangles(project_dir, loc_id, z_query_m):
    """Load rooftop footprint triangles whose roof is above the TX height."""
    mesh_dir = os.path.join(project_dir, "data", "scenes", loc_id, "mesh")
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


def sample_uniform_tx_positions(box, num_samples):
    """Sample TX positions uniformly from the configured box."""
    rand_x = np.random.uniform(box.x_min, box.x_max, num_samples)
    rand_y = np.random.uniform(box.y_min, box.y_max, num_samples)
    rand_z = np.random.uniform(box.z_min, box.z_max, num_samples)
    return np.stack([rand_x, rand_y, rand_z], axis=1)


def sample_free_space_tx_positions(project_dir, loc_id, box, num_samples, deployment_cfg):
    """Keep sampling until exactly num_samples TX points are outside rooftops."""
    z_query_m = float(deployment_cfg.get("rooftop_mask_z_m", box.z_min))
    rooftop_triangles = load_rooftop_triangles(project_dir, loc_id, z_query_m)
    print(
        f"Rejecting TX points inside {len(rooftop_triangles)} rooftop footprint "
        f"triangles at z >= {z_query_m:.1f} m"
    )

    accepted = []
    total_draws = 0
    batch_size = max(num_samples, 1000)
    max_draws = int(deployment_cfg.get("max_tx_sampling_draws", num_samples * 100))

    while sum(len(batch) for batch in accepted) < num_samples:
        if total_draws >= max_draws:
            num_accepted = sum(len(batch) for batch in accepted)
            raise RuntimeError(
                f"Only accepted {num_accepted}/{num_samples} TX points after "
                f"{total_draws} draws. Increase max_tx_sampling_draws or widen tx_box."
            )

        remaining_draws = max_draws - total_draws
        current_batch_size = min(batch_size, remaining_draws)
        candidates = sample_uniform_tx_positions(box, current_batch_size)
        total_draws += current_batch_size

        rejected = points_on_rooftop_mask(candidates[:, :2], rooftop_triangles)
        accepted.append(candidates[~rejected])

    tx_positions = np.concatenate(accepted, axis=0)[:num_samples]
    acceptance_rate = 100.0 * num_samples / max(total_draws, 1)
    print(
        f"Accepted {num_samples} free-space TX samples after {total_draws} draws "
        f"({acceptance_rate:.1f}% acceptance rate)"
    )
    return tx_positions


@hydra.main(version_base=None, config_path="../../configs/scenarios")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    project_dir = cfg.project_dir
    loc_id = cfg.selected_id
    
    # Generate TX Positions in the box
    box = cfg.deployment.tx_box
    N = cfg.deployment.num_samples

    reject_on_rooftop = bool(cfg.deployment.get("reject_tx_on_rooftop_mask", False))
    if reject_on_rooftop:
        tx_positions = sample_free_space_tx_positions(project_dir, loc_id, box, N, cfg.deployment)
    else:
        tx_positions = sample_uniform_tx_positions(box, N)

    # Setup Output Path
    output_dir = os.path.join(project_dir, "configs", "gps_coordinates")
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f"{loc_id}_tx_rx.csv")

    # Write CSV
    fieldnames = ['name', 'type', 'x', 'y', 'z', 'look_at_x', 'look_at_y', 'look_at_z']
    
    print(f"Generating {N} TX samples for location: {loc_id}")
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Add the Randomized Transmitters (tx1, tx2, ...)
        for i, pos in enumerate(tx_positions):
            writer.writerow({
                'name': f'tx{i+1}',
                'type': 'tx',
                'x': f"{pos[0]:.3f}", 
                'y': f"{pos[1]:.3f}", 
                'z': f"{pos[2]:.3f}",
                'look_at_x': '', 'look_at_y': '', 'look_at_z': ''
            })

        # Add one row per base station (receiver)
        for bs in cfg.deployment.base_stations:
            bs_pos  = bs.position
            bs_look = bs.look_at
            writer.writerow({
                'name':       bs.name,
                'type':       'rx',
                'x':          f"{bs_pos[0]:.3f}",
                'y':          f"{bs_pos[1]:.3f}",
                'z':          f"{bs_pos[2]:.3f}",
                'look_at_x':  f"{bs_look[0]:.3f}",
                'look_at_y':  f"{bs_look[1]:.3f}",
                'look_at_z':  f"{bs_look[2]:.3f}",
            })

    num_bs = len(cfg.deployment.base_stations)
    print(f"SUCCESS! Generated layout with {num_bs} base station(s) saved to: {csv_filename}")

if __name__ == "__main__":
    main()