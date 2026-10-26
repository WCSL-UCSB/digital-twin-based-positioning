# =============================================================================
# filter_and_adjust_paths.py — Post-process traced paths for a specific array
# =============================================================================
#
# Reads the shape-preserving .npz from path_gen.py, transforms arrival angles
# into each RX's local coordinate system, applies separate azimuth/elevation
# FOV cuts, normalizes delays, and precomputes per-element differential delays
# and inter-element phase shifts for the configured UPA.
#
# Convention: TRUE SPHERICAL angles in the LCS.
#   theta_az = atan2(k_y, k_x)   (azimuth from broadside, rad)
#   theta_el = arcsin(k_z)       (elevation from array plane, rad)
#
# Steering phases (per path):
#   omega_h = 2*pi * d_h_wl * cos(theta_el) * sin(theta_az)
#   omega_v = 2*pi * d_v_wl * sin(theta_el)
#
# Per-element delay offset (row m, column n, centered indices):
#   delta_tau = -(n*d_h_phys*cos(theta_el)*sin(theta_az)
#               + m*d_v_phys*sin(theta_el)) / c
# Arrival directions point toward the source/last interaction: elements
# displaced toward it receive the wave earlier (negative delay offset).
# =============================================================================

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.dataset_io import resolve_input, pinned, write_sidecar  # noqa: E402
from src.utils.path_classification import LOS_CLASSIFICATION, los_from_interactions
import time
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
from scipy.constants import speed_of_light

load_dotenv()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def build_rotation_matrix(pos, look_at):
    """
    Build the 3x3 rotation matrix R mapping LCS -> GCS.

        +x_local = broadside (from pos toward look_at)
        +y_local = array horizontal axis (column direction)
        +z_local = array vertical axis (row direction)

    To go GCS -> LCS use R.T (orthonormal: R_inv = R.T).

    Uses an explicit Gram-Schmidt step so the basis is orthonormal by
    construction, including when x_local is parallel to z_world
    (look straight up/down).
    """
    pos = np.asarray(pos, dtype=float)
    look_at = np.asarray(look_at, dtype=float)
    if (pos.shape != (3,) or look_at.shape != (3,)
            or not np.isfinite(pos).all() or not np.isfinite(look_at).all()):
        raise ValueError("Receiver position and look_at must be finite 3D points")

    x_local = look_at - pos
    norm = np.linalg.norm(x_local)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("Receiver look_at must differ from its position")
    x_local = x_local / norm

    z_world = np.array([0.0, 0.0, 1.0])
    y_local = np.cross(z_world, x_local)

    if np.linalg.norm(y_local) == 0.0:
        # Degenerate: x_local is (anti)parallel to world-z. Pick any seed
        # and Gram-Schmidt it against x_local. For any nonzero horizontal
        # projection, retain its azimuth, matching the tracer's look_at yaw.
        y_local = np.array([0.0, 1.0, 0.0])

    # Gram-Schmidt: remove component along x_local, then normalize.
    y_local = y_local - np.dot(y_local, x_local) * x_local
    y_local = y_local / np.linalg.norm(y_local)

    z_local = np.cross(x_local, y_local)
    z_local = z_local / np.linalg.norm(z_local)

    return np.column_stack([x_local, y_local, z_local])


def gcs_angles_to_lcs_cartesian(theta_gcs, phi_gcs, R):
    """
    Convert GCS spherical angles to LCS Cartesian unit vectors.

    GCS convention (Sionna): theta from +z, phi from +x in xy-plane.

    Returns
    -------
    k_lcs : ndarray (..., 3) unit vectors in the device LCS.
    """
    k_gcs = np.stack([
        np.sin(theta_gcs) * np.cos(phi_gcs),
        np.sin(theta_gcs) * np.sin(phi_gcs),
        np.cos(theta_gcs),
    ], axis=-1)
    # R maps LCS->GCS, so R.T maps GCS->LCS.
    # For a row-vector convention, k_lcs = k_gcs @ R (== k_gcs @ (R.T).T).
    return k_gcs @ R


def lcs_to_spherical_broadside(k_lcs):
    """
    True-spherical broadside-referenced angles from LCS unit vectors.

        theta_az = atan2(k_y, k_x)   -- azimuth from broadside (+x_local)
        theta_el = arcsin(k_z)       -- elevation from array plane (xy_local)

    Both in radians.
    """
    kx = k_lcs[..., 0]
    ky = k_lcs[..., 1]
    kz = k_lcs[..., 2]
    theta_az = np.arctan2(ky, kx)
    theta_el = np.arcsin(np.clip(kz, -1.0, 1.0))
    return theta_az, theta_el


def compute_element_delays_ns(theta_az, theta_el, M, N, d_h_wl, d_v_wl, frequency):
    """
    Per-element differential delays for a UPA, in NANOSECONDS.

    Centered element indices:
        n = column index, 0..N-1, centered so n' = n - (N-1)/2
        m = row index,    0..M-1, centered so m' = m - (M-1)/2

    delta_tau[m, n, p] = -(n' * d_h_phys * cos(theta_el_p) * sin(theta_az_p)
                         + m' * d_v_phys * sin(theta_el_p)) / c

    The arrival unit vector points from the RX toward the source/last
    interaction. In the plane-wave approximation, moving an element toward
    that direction shortens its path: delta_tau = -dot(offset, direction)/c.
    Thus exp(-j*2*pi*f*delta_tau) agrees with the positive spatial steering
    phase used downstream. Offsets are relative to the array center and
    may be negative, including for the earliest surviving path.

    Returns
    -------
    delta_tau_ns : ndarray (M, N, num_paths) in nanoseconds.
    """
    wavelength = speed_of_light / frequency
    d_h_phys = d_h_wl * wavelength
    d_v_phys = d_v_wl * wavelength

    n_idx = np.arange(N, dtype=float) - (N - 1) / 2.0
    m_idx = np.arange(M, dtype=float) - (M - 1) / 2.0

    cos_el = np.cos(theta_el)
    sin_az = np.sin(theta_az)
    sin_el = np.sin(theta_el)

    horiz = d_h_phys * np.outer(n_idx, cos_el * sin_az)  # (N, P)
    vert = d_v_phys * np.outer(m_idx, sin_el)            # (M, P)

    delta_tau_s = -(vert[:, np.newaxis, :] + horiz[np.newaxis, :, :]) / speed_of_light
    return delta_tau_s * 1e9  # -> ns


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def resolve_frequency(cfg, meta_frequency_hz):
    """
    Source-of-truth: the .npz metadata. If the config also specifies a
    frequency, assert it matches (within 1 Hz). Drift here is a silent
    killer of downstream phase math, so we fail loudly.
    """
    meta_frequency_hz = float(meta_frequency_hz)
    if not np.isfinite(meta_frequency_hz) or meta_frequency_hz <= 0:
        raise ValueError("Raw-path frequency must be finite and positive")
    if "frequency" in cfg:
        cfg_freq = float(cfg.frequency)
        if not np.isfinite(cfg_freq) or cfg_freq <= 0:
            raise ValueError("Config frequency must be finite and positive")
        if abs(cfg_freq - meta_frequency_hz) > 1.0:
            raise ValueError(
                f"Frequency mismatch: config says {cfg_freq:.6g} Hz, "
                f".npz metadata says {meta_frequency_hz:.6g} Hz. "
                "Regenerate paths or fix the config."
            )
    return float(meta_frequency_hz)


def validate_array_geometry(arr):
    """Check panel dimensions, wavelength spacings and full FoV widths."""
    for key in ("num_rows", "num_cols"):
        value = float(arr[key])
        if not np.isfinite(value) or value < 1 or not value.is_integer():
            raise ValueError(f"rx_array_geometry.{key} must be a positive integer")
    for key in ("horizontal_spacing", "vertical_spacing"):
        value = float(arr.get(key, 0.5))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"rx_array_geometry.{key} must be finite and positive")
    for key, maximum in (("fov_az_deg", 360), ("fov_el_deg", 180)):
        value = float(arr[key])
        if not np.isfinite(value) or not 0 < value <= maximum:
            raise ValueError(f"rx_array_geometry.{key} must be in (0, {maximum}] degrees")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../configs/scenarios")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    project_dir = cfg.project_dir

    # ---- Array geometry from config ----
    validate_array_geometry(cfg.rx_array_geometry)
    M = int(cfg.rx_array_geometry.num_rows)
    N = int(cfg.rx_array_geometry.num_cols)
    d_h_wl = float(cfg.rx_array_geometry.get("horizontal_spacing", 0.5))
    d_v_wl = float(cfg.rx_array_geometry.get("vertical_spacing", 0.5))

    fov_az_deg = float(cfg.rx_array_geometry.fov_az_deg)
    fov_el_deg = float(cfg.rx_array_geometry.fov_el_deg)
    fov_az_rad = np.radians(fov_az_deg)
    fov_el_rad = np.radians(fov_el_deg)

    array_type = "UPA" if M > 1 and N > 1 else "ULA"

    # ---- Load traced paths ----
    path_dir = os.path.join(project_dir, "data", "path_data")
    npz_file = resolve_input(
        os.path.join(path_dir, f"*_{cfg.scenario_name}_paths.npz"),
        pinned(cfg, "paths_npz"), label="raw paths")

    data = np.load(npz_file, allow_pickle=True)
    metadata = data["metadata"].item()

    frequency = resolve_frequency(cfg, metadata["frequency_hz"])
    wavelength = speed_of_light / frequency

    print(f"\n--- RX Array ({array_type}) ---")
    print(f"Layout   : {M} rows x {N} cols = {M*N} elements")
    print(f"Spacing  : h={d_h_wl} lambda ({d_h_wl*wavelength*1e3:.2f} mm), "
          f"v={d_v_wl} lambda ({d_v_wl*wavelength*1e3:.2f} mm)")
    print(f"Frequency: {frequency/1e9:.3f} GHz (lambda={wavelength*1e3:.2f} mm)")
    print(f"FOV      : az=+-{fov_az_deg/2:.1f} deg, el=+-{fov_el_deg/2:.1f} deg")

    # ---- Shape-preserving arrays from path_gen.py ----
    a_3d       = data["a"]              # (num_rx, num_tx, max_paths)
    tau_ns_3d  = data["tau_ns"]
    theta_r_3d = data["theta_r"]
    phi_r_3d   = data["phi_r"]
    theta_t_3d = data["theta_t"]
    phi_t_3d   = data["phi_t"]
    num_valid  = data["num_valid_paths"]  # (num_rx, num_tx) int

    # Required to distinguish unobstructed direct paths from transmissions
    # and short reflections, which can share the straight-line travel time.
    inter_4d = data["interactions"] if "interactions" in data.files else None
    los_from_interactions(inter_4d, num_valid)  # validate before filtering

    rx_names   = data["rx_names"]
    rx_pos     = data["rx_pos"]
    rx_look_at = data["rx_look_at"]
    tx_names   = data["tx_names"]

    num_rx, num_tx, max_paths_in = tau_ns_3d.shape
    if not num_rx or not num_tx:
        raise ValueError("Raw paths must contain at least one RX and one TX")
    for key, values in (("a", a_3d), ("theta_r", theta_r_3d), ("phi_r", phi_r_3d),
                        ("theta_t", theta_t_3d), ("phi_t", phi_t_3d)):
        if values.shape != tau_ns_3d.shape:
            raise ValueError(f"Raw {key} shape must match tau_ns")
    if (num_valid.shape != (num_rx, num_tx)
            or inter_4d.shape[1:] != tau_ns_3d.shape):
        raise ValueError("Raw counts/interactions must match the delay tensor")
    for prefix, count in (("rx", num_rx), ("tx", num_tx)):
        if (data[f"{prefix}_names"].shape != (count,)
                or data[f"{prefix}_pos"].shape != (count, 3)
                or data[f"{prefix}_look_at"].shape != (count, 3)):
            raise ValueError(f"Raw {prefix} device tables do not match the path axes")
        if not np.isfinite(data[f"{prefix}_pos"]).all():
            raise ValueError(f"Raw {prefix} positions must be finite")

    n_links_traced    = int((num_valid > 0).sum())
    n_total_links     = num_rx * num_tx
    print(f"Links with any traced paths:     {n_links_traced} / {n_total_links} "
        f"({100*n_links_traced/n_total_links:.1f}%)")
    mean_paths = float(num_valid[num_valid > 0].mean()) if n_links_traced else 0.0
    print(f"Mean paths per traced link:      {mean_paths:.2f}")
    print(f"Max paths on any link:           {int(num_valid.max())}")

    # ---- Output accumulators (shape-preserving with a new path axis) ----
    # We don't know the post-filter max_paths yet, so first pass collects
    # per-link kept indices, then we allocate once we know the ceiling.
    kept_indices = np.empty((num_rx, num_tx), dtype=object)
    num_kept     = np.zeros((num_rx, num_tx), dtype=np.int32)

    # LCS-referenced spherical angles per kept path
    theta_az_ragged = np.empty((num_rx, num_tx), dtype=object)
    theta_el_ragged = np.empty((num_rx, num_tx), dtype=object)

    # Per-RX rotation matrices (store for downstream too)
    R_rx_all = np.zeros((num_rx, 3, 3), dtype=np.float64)

    print(f"\n--- Filtering (front-hemisphere + FOV az/el) ---")

    # Stage-by-stage link survival counters
    stage_counts = {
        "total":  num_rx * num_tx,  # every possible link
        "traced": 0,                # path_gen returned at least one valid path
        "front":  0,                # at least one path in front hemisphere
        "az":     0,                # at least one path passed front + az
        "el":     0,                # at least one path passed front + az + el
        "final":  0,                # same as el, kept separate for clarity
    }

    for r in range(num_rx):
        look = rx_look_at[r]
        if not np.isfinite(look).all():
            raise ValueError(
                f"BS/RX '{rx_names[r]}' requires an explicit, finite look_at target. "
                "Update the deployment CSV and regenerate raw paths.")
        R = build_rotation_matrix(rx_pos[r], look)
        R_rx_all[r] = R
        bs_gcs = R[:, 0]
        print(f"\n  RX '{rx_names[r]}' broadside GCS: "
              f"[{bs_gcs[0]:+.3f}, {bs_gcs[1]:+.3f}, {bs_gcs[2]:+.3f}]")

        for t in range(num_tx):
            nv = int(num_valid[r, t])

            # Initialize empty defaults so the object arrays always have something
            kept_indices[r, t]    = np.empty(0, dtype=np.int64)
            theta_az_ragged[r, t] = np.empty(0, dtype=np.float64)
            theta_el_ragged[r, t] = np.empty(0, dtype=np.float64)

            if nv == 0:
                continue
            for values in (a_3d, tau_ns_3d, theta_r_3d, phi_r_3d, theta_t_3d, phi_t_3d):
                if not np.isfinite(values[r, t, :nv]).all():
                    raise ValueError(f"Nonfinite valid path data at RX {r}, TX {t}")
            if np.any(tau_ns_3d[r, t, :nv] < 0):
                raise ValueError(f"Negative raw delay at RX {r}, TX {t}")
            stage_counts["traced"] += 1

            # Only the first nv entries along the path axis are meaningful.
            idx_valid = np.arange(nv)
            theta_gcs = theta_r_3d[r, t, :nv]
            phi_gcs   = phi_r_3d[r, t, :nv]

            # GCS -> LCS Cartesian
            k_lcs = gcs_angles_to_lcs_cartesian(theta_gcs, phi_gcs, R)

            # Front-hemisphere precondition
            front = k_lcs[..., 0] > 0.0
            if not np.any(front):
                continue
            stage_counts["front"] += 1

            # True-spherical broadside angles
            theta_az_all, theta_el_all = lcs_to_spherical_broadside(k_lcs)

            # Independent az / el FOV cuts
            az_ok = np.abs(theta_az_all) <= (fov_az_rad / 2.0)
            el_ok = np.abs(theta_el_all) <= (fov_el_rad / 2.0)

            if not np.any(front & az_ok):
                continue
            stage_counts["az"] += 1

            if not np.any(front & az_ok & el_ok):
                continue
            stage_counts["el"] += 1

            keep = front & az_ok & el_ok
            kept_local = idx_valid[keep]
            kept_indices[r, t]    = kept_local
            num_kept[r, t]        = kept_local.size
            theta_az_ragged[r, t] = theta_az_all[keep]
            theta_el_ragged[r, t] = theta_el_all[keep]
            stage_counts["final"] += 1

    # ---- Stage-by-stage survival report ----
    print("\n--- Link survival by stage ---")
    total = stage_counts["total"]
    for key in ["traced", "front", "az", "el", "final"]:
        v = stage_counts[key]
        pct = 100.0 * v / total if total > 0 else 0.0
        print(f"  {key:8s}: {v:6d} / {total} ({pct:5.1f}%)")

    # Extra diagnostics when something drops a path
    if stage_counts["traced"] > 0:
        nv_flat = num_valid[num_valid > 0]
        print(f"\n  Of traced links ({stage_counts['traced']}):")
        print(f"    mean paths/link : {nv_flat.mean():.2f}")
        print(f"    median          : {np.median(nv_flat):.1f}")
        print(f"    max             : {int(nv_flat.max())}")
        print(f"    min             : {int(nv_flat.min())}")

    # ---- Second pass: allocate padded arrays and fill ----
    max_paths_out = int(num_kept.max()) if num_kept.size > 0 else 0
    print(f"\nMax surviving paths per link: {max_paths_out}")

    if max_paths_out == 0:
        raise ValueError("No paths survived filtering for any link; no refined dataset written")

    shape_3d = (num_rx, num_tx, max_paths_out)
    tau_ns_out        = np.zeros(shape_3d, dtype=np.float64)
    theta_az_out      = np.zeros(shape_3d, dtype=np.float64)
    theta_el_out      = np.zeros(shape_3d, dtype=np.float64)
    omega_h_out       = np.zeros(shape_3d, dtype=np.float64)
    omega_v_out       = np.zeros(shape_3d, dtype=np.float64)
    a_out             = np.zeros(shape_3d, dtype=np.complex128)
    theta_t_out       = np.zeros(shape_3d, dtype=np.float64)
    phi_t_out         = np.zeros(shape_3d, dtype=np.float64)
    min_delay_ns_out  = np.zeros((num_rx, num_tx), dtype=np.float64)

    # Per-element delay tensor: (num_rx, num_tx, M, N, max_paths_out)
    tau_per_element_ns_out = np.zeros(
        (num_rx, num_tx, M, N, max_paths_out), dtype=np.float64
    )

    # Preserve interactions for every surviving path.
    max_depth = inter_4d.shape[0]
    interactions_out = np.zeros(
        (max_depth, num_rx, num_tx, max_paths_out), dtype=np.int16
    )

    for r in range(num_rx):
        for t in range(num_tx):
            k = int(num_kept[r, t])
            if k == 0:
                continue

            idx = kept_indices[r, t]
            tau_raw_ns = tau_ns_3d[r, t, idx]
            min_delay_ns = float(np.min(tau_raw_ns))
            tau_norm_ns = tau_raw_ns - min_delay_ns

            th_az = theta_az_ragged[r, t]
            th_el = theta_el_ragged[r, t]

            # Per-element differential delays in ns
            delta_tau_ns = compute_element_delays_ns(
                th_az, th_el, M, N, d_h_wl, d_v_wl, frequency
            )  # (M, N, k)
            tau_per_el_ns = tau_norm_ns[np.newaxis, np.newaxis, :] + delta_tau_ns

            # Inter-element phase shifts (true-spherical steering)
            cos_el = np.cos(th_el)
            omega_h = 2.0 * np.pi * d_h_wl * cos_el * np.sin(th_az)
            omega_v = 2.0 * np.pi * d_v_wl * np.sin(th_el)

            tau_ns_out[r, t, :k]   = tau_norm_ns
            theta_az_out[r, t, :k] = th_az
            theta_el_out[r, t, :k] = th_el
            omega_h_out[r, t, :k]  = omega_h
            omega_v_out[r, t, :k]  = omega_v
            a_out[r, t, :k]        = a_3d[r, t, idx]
            theta_t_out[r, t, :k]  = theta_t_3d[r, t, idx]
            phi_t_out[r, t, :k]    = phi_t_3d[r, t, idx]
            min_delay_ns_out[r, t] = min_delay_ns
            tau_per_element_ns_out[r, t, :, :, :k] = tau_per_el_ns
            interactions_out[:, r, t, :k] = inter_4d[:, r, t, idx]

    # The label describes paths surviving the RX field-of-view filter.
    is_los_kept = los_from_interactions(interactions_out, num_kept)

    # ---- Verification print ----
    print("\n" + "=" * 70)
    print("  VERIFICATION")
    print("=" * 70)
    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    verif_shown = 0
    for r in range(num_rx):
        for t in range(num_tx):
            k = int(num_kept[r, t])
            if k == 0 or verif_shown >= 3:
                continue
            print(f"\nRX '{rx_names[r]}' -> TX '{tx_names[t]}' | paths={k}")
            print(f"  tau (center)      [ns] : {tau_ns_out[r, t, :k]}")
            print(f"  theta_az          [deg]: {np.rad2deg(theta_az_out[r, t, :k])}")
            print(f"  theta_el          [deg]: {np.rad2deg(theta_el_out[r, t, :k])}")
            print(f"  omega_h           [rad]: {omega_h_out[r, t, :k]}")
            if M > 1:
                print(f"  omega_v           [rad]: {omega_v_out[r, t, :k]}")
            print(f"  min delay offset  [ns] : {min_delay_ns_out[r, t]:.3f}")
            dt = tau_per_element_ns_out[r, t, :, :, 0]
            spread_ps = (dt.max() - dt.min()) * 1e3
            print(f"  path-0 spatial spread [ps]: {spread_ps:.2f}")
            verif_shown += 1
    np.set_printoptions()
    print("\n" + "=" * 70)

    # ---- Save as .npz (consistent with path_gen.py) ----
    output_dir = os.path.join(project_dir, "data", "path_data", "refined_paths")
    os.makedirs(output_dir, exist_ok=True)
    stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
    out_file = os.path.join(
        output_dir, f"{stamp}_filtered_{cfg.scenario_name}.npz"
    )

    filter_metadata = {
        "source_npz":       os.path.basename(npz_file),
        "source_metadata":  metadata,
        "frequency_hz":     frequency,
        "rx_array": {
            "num_rows": M, "num_cols": N,
            "horizontal_spacing_wl": d_h_wl,
            "vertical_spacing_wl":   d_v_wl,
            "fov_az_deg": fov_az_deg,
            "fov_el_deg": fov_el_deg,
        },
        "angle_convention": "true_spherical_broadside",
        "delay_reference": "earliest_surviving_path_at_array_center",
        "rx_orientation_policy": "explicit_look_at_required",
        "link_survival_counts": stage_counts,
        "los_classification": LOS_CLASSIFICATION,
        "element_delay_convention": "delta_tau=-dot(element_offset,arrival_direction)/c",
        "units": {
            "tau_ns": "ns",
            "tau_per_element_ns": "ns",
            "theta_az": "rad",
            "theta_el": "rad",
            "omega_h": "rad",
            "omega_v": "rad",
        },
        "polarization_traced": metadata.get("rx_array", {}).get(
            "polarization", metadata.get("polarization_traced", "unknown")),
    }

    print(f"\nSaving to {out_file}")
    np.savez(
        out_file,
        # per-link path data, (num_rx, num_tx, max_paths_out)
        a                   = a_out,
        tau_ns              = tau_ns_out,
        theta_az            = theta_az_out,
        theta_el            = theta_el_out,
        omega_h             = omega_h_out,
        omega_v             = omega_v_out,
        theta_t             = theta_t_out,
        phi_t               = phi_t_out,
        num_paths_after_filter = num_kept,
        min_delay_ns        = min_delay_ns_out,
        # per-element delays, (num_rx, num_tx, M, N, max_paths_out)
        tau_per_element_ns  = tau_per_element_ns_out,
        # device + geometry tables
        rx_names            = rx_names,
        rx_pos              = rx_pos,
        rx_look_at          = rx_look_at,
        R_rx                = R_rx_all,
        tx_names            = tx_names,
        tx_pos              = data["tx_pos"],
        tx_look_at          = data["tx_look_at"],
        # metadata
        metadata            = filter_metadata,
        interactions        = interactions_out,
        is_los_kept         = is_los_kept,
    )
    write_sidecar(out_file, stage="filter_and_adjust_paths", cfg=cfg,
                  inputs={"paths_npz": npz_file}, summary=filter_metadata)
    print(f"Saved {int(num_kept.sum())} total surviving paths across "
          f"{int((num_kept > 0).sum())} links.")


if __name__ == "__main__":
    main()
