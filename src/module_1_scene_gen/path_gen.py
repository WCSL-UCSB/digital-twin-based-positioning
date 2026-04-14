# =============================================================================
# path_gen.py — Batched Ray-Tracing Path Generation
# =============================================================================
import os
import csv
import gc
import time
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

load_dotenv()

import sionna
from sionna.rt import (
    load_scene, Transmitter, Receiver, PlanarArray,
    RadioMaterial, PathSolver,
)


# ---------------------------------------------------------------------------
# CSV / materials helpers
# ---------------------------------------------------------------------------

def load_devices_from_csv(csv_path):
    """Reads TX/RX positions (and optional look_at) from a CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Device layout CSV not found at: {csv_path}")

    devices = []
    print(f"Loading devices from: {csv_path}")
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dev = {
                "name": row["name"],
                "type": row["type"].strip().lower(),
                "pos":  [float(row["x"]), float(row["y"]), float(row["z"])],
            }
            if row.get("look_at_x") and row.get("look_at_y") and row.get("look_at_z"):
                dev["look_at"] = [
                    float(row["look_at_x"]),
                    float(row["look_at_y"]),
                    float(row["look_at_z"]),
                ]
            else:
                dev["look_at"] = None
            devices.append(dev)
    return devices


def apply_materials(scene, mat_config):
    if isinstance(mat_config, str):
        if not os.path.exists(mat_config):
            print(f"[Warning] Material config file not found: {mat_config}")
            return
        cfg = OmegaConf.load(mat_config)
    else:
        cfg = mat_config

    if "overrides" not in cfg:
        return

    custom_mats = {}
    for key, props in cfg.overrides.items():
        mat_name = f"custom_{key}"
        new_mat = RadioMaterial(
            name=mat_name,
            relative_permittivity=props["relative_permittivity"],
            conductivity=props["conductivity"],
        )
        scene.add(new_mat)
        custom_mats[key] = new_mat

    count = 0
    for obj_name, scene_obj in scene.objects.items():
        lname = obj_name.lower()
        for key, new_mat in custom_mats.items():
            if key in lname:
                scene_obj.radio_material = new_mat
                count += 1
                break
    print(f"Applied custom materials to {count} objects.")


@hydra.main(version_base=None, config_path="../../configs/scenarios")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    loc_id      = cfg.selected_id
    project_dir = cfg.project_dir
    frequency   = float(cfg.frequency)

    # ---- GPS DB + CSV routing ----
    gps_path = cfg.gps_location_path
    if os.path.isdir(gps_path):
        gps_path = os.path.join(gps_path, "gps_locations.yaml")
    gps_db = OmegaConf.load(gps_path)

    if loc_id in gps_db and "coordinate_file" in gps_db[loc_id]:
        csv_path = gps_db[loc_id].coordinate_file.replace("${project_dir}", project_dir)
    else:
        print(f"[Info] '{loc_id}' not in GPS database. Using default CSV routing.")
        yaml_dir = os.path.dirname(gps_path)
        csv_path = os.path.join(yaml_dir, f"{loc_id}_tx_rx.csv")

    # ---- Scene path ----
    if "scene_xml_path" in cfg:
        raw_path = cfg.scene_xml_path.replace("${project_dir}", project_dir)
        scene_path = os.path.abspath(raw_path)
    else:
        scene_path = os.path.join(project_dir, "data", "scenes", loc_id, "scene.xml")
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Geometry not found: {scene_path}")

    # ---- Load scene ONCE ----
    print(f"Loading scene: {scene_path}")
    scene = load_scene(scene_path)
    scene.frequency = frequency
    if "material_config" in cfg:
        apply_materials(scene, cfg.material_config)

    # Iso V-pol single element at both ends — "trace iso, synthesize arrays later"
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

    # ---- Devices ----
    devices   = load_devices_from_csv(csv_path)
    tx_list   = [d for d in devices if d["type"] == "tx"]
    rx_list   = [d for d in devices if d["type"] == "rx"]
    num_tx    = len(tx_list)
    num_rx    = len(rx_list)

    print("\n--- Device Layout ---")
    print(f"TX: {num_tx}  |  RX: {num_rx}")

    # RXs are persistent — add them once
    for rx in rx_list:
        kwargs = {"name": rx["name"], "position": rx["pos"]}
        if rx["look_at"] is not None:
            kwargs["look_at"] = rx["look_at"]
        scene.add(Receiver(**kwargs))

    # ---- PathSolver kwargs from config ----
    rt_cfg = cfg.ray_tracing_params
    solver_kwargs = dict(
        max_depth           = int(rt_cfg.max_depth),
        synthetic_array     = bool(rt_cfg.synthetic_array),
        los                 = bool(rt_cfg.get("los", True)),
        specular_reflection = bool(rt_cfg.get("specular_reflection", True)),
        diffuse_reflection  = bool(rt_cfg.get("diffuse_reflection", False)),
        refraction          = bool(rt_cfg.get("refraction", False)),
    )
    if "samples_per_src" in rt_cfg:
        solver_kwargs["samples_per_src"] = int(rt_cfg.samples_per_src)
    if "max_num_paths_per_src" in rt_cfg:
        solver_kwargs["max_num_paths_per_src"] = int(rt_cfg.max_num_paths_per_src)
    if "seed" in rt_cfg:
        solver_kwargs["seed"] = int(rt_cfg.seed)

    # ---- Batched tracing ----
    batch_size  = min(int(rt_cfg.get("batch_size", 50)), num_tx)
    num_batches = int(np.ceil(num_tx / batch_size))
    print(f"\nTracing {num_tx} TXs in {num_batches} batches "
          f"(batch_size={batch_size}, max_depth={solver_kwargs['max_depth']})")

    batched = []  # list of dicts, each with shape (num_rx, num_tx_batch, num_paths_batch)

    for bi in range(num_batches):
        s = bi * batch_size
        e = min(s + batch_size, num_tx)
        chunk = tx_list[s:e]
        print(f"  -> Batch {bi+1}/{num_batches}: TX {s+1}..{e}")

        # Add this batch's TXs
        for tx in chunk:
            kwargs = {"name": tx["name"], "position": tx["pos"]}
            if tx["look_at"] is not None:
                kwargs["look_at"] = tx["look_at"]
            scene.add(Transmitter(**kwargs))

        solver = PathSolver()
        paths  = solver(scene=scene, **solver_kwargs)

        # Complex amplitudes: Sionna returns (real, imag) tuple
        # Shape: (num_rx, num_rx_ant, num_tx_batch, num_tx_ant, num_paths)
        a_re = paths.a[0].numpy()
        a_im = paths.a[1].numpy()
        assert a_re.shape[1] == 1 and a_re.shape[3] == 1, (
            f"Expected single-antenna iso trace, got shape {a_re.shape}. "
            "This script assumes num_rx_ant == num_tx_ant == 1."
        )
        a_complex = (a_re + 1j * a_im)[:, 0, :, 0, :]  # (num_rx, num_tx_batch, num_paths)

        # tau, angles with synthetic_array=True: (num_rx, num_tx_batch, num_paths)
        batched.append({
            "a":       a_complex,
            "tau":     paths.tau.numpy(),
            "theta_r": paths.theta_r.numpy(),
            "phi_r":   paths.phi_r.numpy(),
            "theta_t": paths.theta_t.numpy(),
            "phi_t":   paths.phi_t.numpy(),
        })

        # Remove this batch's TXs before adding the next
        for tx in chunk:
            scene.remove(tx["name"])

        del paths, solver
        gc.collect()

    # ---- Merge batches with padding on the path axis ----
    print("\nMerging batches...")
    max_paths_global = max(b["tau"].shape[-1] for b in batched)

    merged = {}
    for k in batched[0].keys():
        pieces = []
        for b in batched:
            v = b[k]
            pad = max_paths_global - v.shape[-1]
            if pad > 0:
                pad_tuple = [(0, 0)] * (v.ndim - 1) + [(0, pad)]
                fill = 0j if k == "a" else (np.nan if k == "tau" else 0.0)
                v = np.pad(v, pad_tuple, constant_values=fill)
            pieces.append(v)
        # concat along tx axis (axis=1 since shape is (num_rx, num_tx_batch, num_paths))
        merged[k] = np.concatenate(pieces, axis=1) if len(pieces) > 1 else pieces[0]

    assert merged["tau"].shape == (num_rx, num_tx, max_paths_global), (
        f"Merged shape mismatch: got {merged['tau'].shape}, "
        f"expected ({num_rx}, {num_tx}, {max_paths_global})"
    )

    # Convert tau to nanoseconds, keep NaN as invalid sentinel
    tau_ns = merged["tau"] * 1e9
    valid  = ~np.isnan(tau_ns)
    num_valid_paths = valid.sum(axis=-1).astype(np.int32)  # (num_rx, num_tx)

    # Replace NaN / pad values with neutral fill so downstream can ignore via num_valid_paths
    tau_ns = np.where(valid, tau_ns, 0.0)
    for k in ("theta_r", "phi_r", "theta_t", "phi_t"):
        merged[k] = np.where(valid, merged[k], 0.0)
    merged["a"] = np.where(valid, merged["a"], 0 + 0j)

    # ---- Device metadata as plain arrays ----
    def _pack(devs):
        names   = np.array([d["name"] for d in devs])
        pos     = np.array([d["pos"] for d in devs], dtype=np.float64)
        look_at = np.array(
            [d["look_at"] if d["look_at"] is not None else [np.nan]*3 for d in devs],
            dtype=np.float64,
        )
        return names, pos, look_at

    tx_names, tx_pos, tx_look_at = _pack(tx_list)
    rx_names, rx_pos, rx_look_at = _pack(rx_list)

    # ---- Save ----
    output_dir = os.path.join(project_dir, "data", "path_data")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(
        output_dir,
        f"{time.strftime('%Y%m%d_%H%M%S')}_{cfg.scenario_name}_paths.npz",
    )

    metadata = {
        "scenario_name":      cfg.scenario_name,
        "selected_id":        loc_id,
        "scene_path":         scene_path,
        "frequency_hz":       frequency,
        "polarization_traced": "V",
        "num_rx":             int(num_rx),
        "num_tx":             int(num_tx),
        "max_paths":          int(max_paths_global),
        "solver_kwargs":      solver_kwargs,
        "sionna_version":     getattr(sionna, "__version__", "unknown"),
        "config_snapshot":    OmegaConf.to_container(cfg, resolve=True),
    }

    print(f"\nSaving to {filename}")
    np.savez(
        filename,
        # per-link path data, shape (num_rx, num_tx, max_paths)
        a               = merged["a"],
        tau_ns          = tau_ns,
        theta_r         = merged["theta_r"],
        phi_r           = merged["phi_r"],
        theta_t         = merged["theta_t"],
        phi_t           = merged["phi_t"],
        num_valid_paths = num_valid_paths,
        # device tables
        tx_names        = tx_names,
        tx_pos          = tx_pos,
        tx_look_at      = tx_look_at,
        rx_names        = rx_names,
        rx_pos          = rx_pos,
        rx_look_at      = rx_look_at,
        # metadata (single pickled object)
        metadata        = metadata,
    )
    print("SUCCESS")


if __name__ == "__main__":
    main()