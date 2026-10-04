# =============================================================================
# channel_taps_gen.py — Convert calibrated paths to first-arrival-referenced taps
# =============================================================================
#
# Consumes the calibrated .npz from apply_link_budget.py and produces a
# complex channel-tap tensor per link, sampled on a uniform time grid at
# `dt` spacing with tau = 0 aligned to the EARLIEST SURVIVING PATH of the
# link (the first arrival kept after filtering/adjusting), for LoS and NLoS
# links alike.
#
# Time reference convention:
#     tau = 0  ≡  arrival time of the earliest surviving path of the link
#
# This models ideal alignment to the first retained arrival; synchronization
# and detectability of that path are not simulated here. A geometric-LoS
# reference (tau_los = ||tx-rx||/c) uses true TX/RX positions, so using that
# reference for the tap axis would introduce unavailable position information.
# We keep the filter's min-normalized frame as-is.
#
# The per-link geometric LoS time and the NLoS first-arrival excess
# (shift_applied = min_delay - tau_los, snapped/clamped) are still COMPUTED and
# SAVED as diagnostics. The upstream interaction-based `is_los_kept[r, t]`
# distinguishes the two regimes — but the excess is not added to the
# taps.
# =============================================================================

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.dataset_io import resolve_input, pinned, write_sidecar  # noqa: E402
from src.utils.radio_power import positive_float, positive_int
from src.utils.path_classification import LOS_CLASSIFICATION, los_from_interactions
import time
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
from scipy.constants import speed_of_light

load_dotenv()

# Tolerance only for diagnosing arrivals earlier than straight-line travel.
# LoS labels come from surviving path interactions, never from this tolerance.
DELAY_TOL_NS = 0.5


# ---------------------------------------------------------------------------
# Dimensionless pulse shapes, with singularities tested in normalized time
# ---------------------------------------------------------------------------

def sinc_pulse(t, Tsym):
    return np.sinc(t / Tsym)


def rrc_pulse(t, Tsym, beta):
    u = np.asarray(t, dtype=float) / Tsym
    if beta == 0:
        return np.sinc(u)

    x = np.pi * u
    fourbt = 4.0 * beta * u
    num = np.sin((1.0 - beta) * x) + fourbt * np.cos((1.0 + beta) * x)
    den = x * (1.0 - fourbt**2)
    at_zero = np.isclose(u, 0.0, rtol=0, atol=1e-10)
    at_pole = np.isclose(np.abs(u), 1.0 / (4.0 * beta), rtol=0, atol=1e-10)
    p = num / np.where(at_zero | at_pole, 1.0, den)
    pole_value = (beta / np.sqrt(2.0)) * (
        (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta)) +
        (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
    )
    return np.where(at_zero, 1.0 + beta * (4.0 / np.pi - 1.0),
                    np.where(at_pole, pole_value, p))


def rc_pulse(t, Tsym, beta):
    u = np.asarray(t, dtype=float) / Tsym
    if beta == 0:
        return np.sinc(u)

    denom = 1.0 - (2.0 * beta * u)**2
    at_pole = np.isclose(np.abs(u), 1.0 / (2.0 * beta), rtol=0, atol=1e-10)
    p = np.sinc(u) * np.cos(np.pi * beta * u) / np.where(at_pole, 1.0, denom)
    pole_value = (np.pi / 4.0) * np.sinc(1.0 / (2.0 * beta))
    return np.where(at_pole, pole_value, p)


def get_pulse_fn(pulse_type):
    pt = pulse_type.strip().lower()
    if pt == "sinc":
        def _sinc(t, Tsym, beta):
            return sinc_pulse(t, Tsym)
        return _sinc
    if pt == "rrc":
        return rrc_pulse
    if pt == "rc":
        return rc_pulse
    raise ValueError(f"Unknown pulse type '{pulse_type}'. Supported: sinc, rrc, rc")


# ---------------------------------------------------------------------------
# Single-link tap generator
# ---------------------------------------------------------------------------

def channel_gen_link(
    a,                    # (k,) complex, in sqrt(W) per element
    tau_center_ns,        # (k,) delays at array center, already first-arrival ref
    omega_h,              # (k,) inter-element phase, horizontal (rad)
    omega_v,              # (k,) inter-element phase, vertical (rad)
    tau_per_elem_ns,      # (M, N, k) per-element delays, already first-arrival ref
    f_carrier,
    M, N,
    dt, T_fixed,
    pulse_fn, pulse_Tsym, pulse_beta,
    include_true_time_delay,
    include_spatial_phase,
):
    """
    Generate (M, N, T_fixed) complex channel taps for one link.

    All delays passed in must already be referenced to the link's time origin
    (tau = 0 at the earliest surviving path). This function does not rebase
    delays; the caller is responsible for that.
    """
    k = a.size
    if k == 0:
        return np.zeros((M, N, T_fixed), dtype=np.complex128)

    tau_center_s = tau_center_ns * 1e-9
    t_grid_s = np.arange(T_fixed, dtype=float) * dt

    # Carrier phase per path, referenced to the earliest surviving path.
    # Note: tau_center is first-arrival-referenced, so this encodes phase
    # relative to the first arrival, not absolute propagation phase. (For the
    # log-magnitude BDP used in training, a per-link constant phase is moot.)
    carrier = np.exp(-1j * 2.0 * np.pi * f_carrier * tau_center_s)

    # Spatial phase (M, N, k)
    if include_spatial_phase:
        n_idx = np.arange(N, dtype=float) - (N - 1) / 2.0
        m_idx = np.arange(M, dtype=float) - (M - 1) / 2.0
        phase = (
            m_idx[:, np.newaxis, np.newaxis] * omega_v[np.newaxis, np.newaxis, :]
            + n_idx[np.newaxis, :, np.newaxis] * omega_h[np.newaxis, np.newaxis, :]
        )
        spatial = np.exp(1j * phase)
    else:
        spatial = np.ones((M, N, k), dtype=np.complex128)

    # Per-element effective delay
    if include_true_time_delay:
        tau_eff_s = tau_per_elem_ns * 1e-9
    else:
        tau_eff_s = np.broadcast_to(
            tau_center_s[np.newaxis, np.newaxis, :], (M, N, k)
        )

    # Sample pulse on (t_grid - tau_eff) -> (M, N, k, T_fixed)
    pulse_arg = (
        t_grid_s[np.newaxis, np.newaxis, np.newaxis, :]
        - tau_eff_s[:, :, :, np.newaxis]
    )
    P = pulse_fn(pulse_arg, pulse_Tsym, pulse_beta)

    gain = a * carrier  # (k,)
    H_per_path = (
        gain[np.newaxis, np.newaxis, :, np.newaxis]
        * spatial[:, :, :, np.newaxis]
        * P
    )
    H = H_per_path.sum(axis=2)  # (M, N, T_fixed)
    return H.astype(np.complex128)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../configs/scenarios")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    project_dir = cfg.project_dir

    # ---- Load calibrated paths ----
    in_dir = os.path.join(project_dir, "data", "path_data", "calibrated_paths")
    in_file = resolve_input(
        os.path.join(in_dir, f"*_calibrated_{cfg.scenario_name}.npz"),
        pinned(cfg, "calibrated_npz"), label="calibrated paths")
    data = np.load(in_file, allow_pickle=True)
    calib_meta = data["metadata"].item()

    # Frequency from source-of-truth metadata chain
    source_meta = calib_meta.get("source_metadata", {})
    f_carrier = positive_float(source_meta.get("frequency_hz", cfg.get("frequency")),
                               "upstream frequency_hz")
    if cfg.get("frequency") is not None:
        configured_frequency = positive_float(cfg.frequency, "frequency")
        if not np.isclose(configured_frequency, f_carrier, rtol=0, atol=1.0):
            raise ValueError("frequency mismatch with calibrated paths; regenerate refinement or fix config")
    lam = speed_of_light / f_carrier

    # Array geometry (sanity-check against config)
    rx_array_meta = source_meta.get("rx_array", {})
    M = positive_int(rx_array_meta.get("num_rows", cfg.rx_array_geometry.num_rows), "array rows")
    N = positive_int(rx_array_meta.get("num_cols", cfg.rx_array_geometry.num_cols), "array columns")
    d_h_wl = positive_float(rx_array_meta.get("horizontal_spacing_wl",
                             cfg.rx_array_geometry.horizontal_spacing), "horizontal spacing")
    d_v_wl = positive_float(rx_array_meta.get("vertical_spacing_wl",
                             cfg.rx_array_geometry.vertical_spacing), "vertical spacing")

    if (positive_int(cfg.rx_array_geometry.num_rows, "configured rows") != M
            or positive_int(cfg.rx_array_geometry.num_cols, "configured columns") != N):
        raise ValueError(
            f"rx_array mismatch: config {cfg.rx_array_geometry.num_rows}x"
            f"{cfg.rx_array_geometry.num_cols}, upstream {M}x{N}. "
            "Regenerate upstream data or fix the config."
        )
    for field, actual in (("horizontal_spacing", d_h_wl), ("vertical_spacing", d_v_wl)):
        configured = positive_float(cfg.rx_array_geometry[field], field)
        if not np.isclose(configured, actual, rtol=0, atol=1e-12):
            raise ValueError(f"rx_array {field} mismatch; regenerate refinement or fix config")

    # ---- Pulse + tap parameters ----
    pulse_type = str(cfg.channel_params.pulse.type).strip().lower()
    pulse_Tsym = positive_float(cfg.channel_params.pulse.Tsym, "channel_params.pulse.Tsym")
    pulse_beta = float(cfg.channel_params.pulse.get("beta", 0.0))
    if not np.isfinite(pulse_beta) or not 0 <= pulse_beta <= 1:
        raise ValueError("channel_params.pulse.beta must be finite and in [0, 1]")
    pulse_fn = get_pulse_fn(pulse_type)

    dt = positive_float(cfg.channel_params.dt, "channel_params.dt")
    include_ttd = bool(cfg.channel_params.get("include_true_time_delay", True))
    include_sp  = bool(cfg.channel_params.get("include_spatial_phase", True))

    # This setting defines the physical channel window. BDP generation uses
    # the resulting saved tap count and checks that its config still agrees.
    configured_taps = (cfg.get("training_sequence") or {}).get("max_delay_bins")
    if configured_taps is not None:
        T_fixed = positive_int(configured_taps, "training_sequence.max_delay_bins")
    else:
        max_delay_ns = float(cfg.channel_params.get("max_delay_ns", 500.0))
        if not np.isfinite(max_delay_ns) or max_delay_ns < 0:
            raise ValueError("channel_params.max_delay_ns must be finite and nonnegative")
        T_fixed = int(np.floor(max_delay_ns * 1e-9 / dt)) + 1

    window_ns = T_fixed * dt * 1e9

    # ---- Pull arrays ----
    a_all          = data["a"]                          # (R, T, P) complex
    tau_ns_all     = data["tau_ns"]                     # already min-normalized
    omega_h_all    = data["omega_h"]
    omega_v_all    = data["omega_v"]
    tau_per_el_all = data["tau_per_element_ns"]         # (R, T, M, N, P)
    n_kept_all     = data["num_paths_after_filter"]     # (R, T)
    min_delay_all  = data["min_delay_ns"]               # (R, T)
    rx_names       = data["rx_names"]
    tx_names       = data["tx_names"]
    rx_pos         = data["rx_pos"]                     # (R, 3)
    tx_pos         = data["tx_pos"]                     # (T, 3)

    if a_all.ndim != 3 or not all(a_all.shape[:2]):
        raise ValueError("Calibrated amplitudes must have shape (RX, TX, path) with nonempty device axes")
    num_rx, num_tx, P_max = a_all.shape
    if (n_kept_all.shape != (num_rx, num_tx)
            or not np.issubdtype(n_kept_all.dtype, np.integer)
            or np.any(n_kept_all < 0) or np.any(n_kept_all > P_max)):
        raise ValueError("Path counts must be (RX, TX) integers within the stored path axis")
    valid = np.arange(P_max) < n_kept_all[..., None]
    for name, value in (("a", a_all), ("tau_ns", tau_ns_all),
                        ("omega_h", omega_h_all), ("omega_v", omega_v_all)):
        if value.shape != a_all.shape or not np.isfinite(value[valid]).all():
            raise ValueError(f"{name} must have shape {a_all.shape} and finite valid entries")
    if tau_per_el_all.shape != (num_rx, num_tx, M, N, P_max):
        raise ValueError("tau_per_element_ns shape must agree with devices, panel dimensions and paths")
    for name, value, shape in (("min_delay_ns", min_delay_all, (num_rx, num_tx)),
                               ("rx_pos", rx_pos, (num_rx, 3)), ("tx_pos", tx_pos, (num_tx, 3))):
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        relevant = value[n_kept_all > 0] if name == "min_delay_ns" else value
        if not np.isfinite(relevant).all():
            raise ValueError(f"{name} must contain finite valid values")
    if rx_names.shape != (num_rx,) or tx_names.shape != (num_tx,):
        raise ValueError("Device names must agree with the path tensor axes")

    # ---- Diagnostics only: geometric LoS delay and NLoS first-arrival excess
    # These are NOT applied to the taps (delays stay first-arrival-referenced);
    # they record the excess delay for downstream metadata / error breakdowns.
    # They do not determine whether an unobstructed direct path survived.
    # tau_los_ns[r, t] = 1e9 * ||tx_pos[t] - rx_pos[r]|| / c
    dxyz = tx_pos[np.newaxis, :, :] - rx_pos[:, np.newaxis, :]   # (R, T, 3)
    d_geom = np.linalg.norm(dxyz, axis=-1)                       # (R, T) meters
    tau_los_ns_all = 1e9 * d_geom / speed_of_light               # (R, T) ns

    # filter's min_delay is the raw delay of the earliest surviving path (ns).
    # shift_ns = min_delay - tau_los is how much later the first kept arrival
    # is than the straight-line LoS instant (the NLoS excess for NLoS links).
    shift_ns_all = min_delay_all - tau_los_ns_all  # (R, T)

    # Read the post-filter label. Older calibrated files can derive it from
    # their surviving interactions; missing evidence must not become a guess.
    has_paths_link = n_kept_all > 0
    if "is_los_kept" in data.files:
        is_los_kept = np.asarray(data["is_los_kept"])
        if is_los_kept.shape != n_kept_all.shape or is_los_kept.dtype != np.bool_:
            raise ValueError("is_los_kept must be a boolean (RX, TX) array")
        if np.any(is_los_kept & ~has_paths_link):
            raise ValueError("An empty link cannot be marked is_los_kept")
        los_label_source = "upstream_is_los_kept"
    else:
        is_los_kept = los_from_interactions(
            data["interactions"] if "interactions" in data.files else None,
            n_kept_all)
        los_label_source = "legacy_surviving_interactions"
    pathological = (shift_ns_all < -DELAY_TOL_NS) & has_paths_link
    n_pathological = int(pathological.sum())

    # Recorded NLoS excess (diagnostic, saved as "shift_applied" for backward
    # compatibility with downstream readers): 0 for LoS-kept, pathological, and
    # empty links; the positive first-arrival-over-LoS offset for NLoS-only
    # links. This is NOT added to the taps.
    shift_applied = np.maximum(shift_ns_all, 0.0)
    shift_applied[is_los_kept] = 0.0
    shift_applied[pathological] = 0.0
    shift_applied[~has_paths_link] = 0.0   # empty links carry no time reference

    # ---- Diagnostic summary ----
    array_type = "UPA" if M > 1 else "ULA"
    print(f"\n--- Channel Tap Generation ({array_type}) ---")
    print(f"Array    : {M} x {N} = {M*N} elements "
          f"(h={d_h_wl}λ, v={d_v_wl}λ)")
    print(f"fc       : {f_carrier/1e9:.3f} GHz | λ = {lam*1e3:.2f} mm")
    print(f"Tap grid : dt={dt*1e9:.1f} ns, T_fixed={T_fixed}, "
          f"window={window_ns:.1f} ns")
    print(f"Pulse    : {pulse_type}, Tsym={pulse_Tsym*1e9:.1f} ns, beta={pulse_beta}")
    print(f"TTD      : {include_ttd}  |  SpatialPhase: {include_sp}")
    print(f"tau = 0    : earliest surviving path (first arrival, per link)")

    has_paths = n_kept_all > 0
    n_non_empty = int(has_paths.sum())
    n_los_kept  = int((is_los_kept & has_paths).sum())
    n_nlos_only = int((~is_los_kept & has_paths).sum())

    print(f"\n--- LoS diagnostic ---")
    print(f"  Links with paths   : {n_non_empty} / {num_rx*num_tx}")
    print(f"  LoS-kept           : {n_los_kept} "
          f"({100*n_los_kept/max(n_non_empty,1):.1f}% of non-empty)")
    print(f"  NLoS-only          : {n_nlos_only} "
          f"({100*n_nlos_only/max(n_non_empty,1):.1f}% of non-empty)")
    if n_pathological > 0:
        print(f"  [WARN] pathological (shift < -{DELAY_TOL_NS} ns): "
              f"{n_pathological} — clamped to 0.")

    if n_non_empty > 0:
        nlos_shifts = shift_applied[~is_los_kept & ~pathological & has_paths]
        if nlos_shifts.size > 0:
            print(f"\n  NLoS-only shift distribution (ns):")
            for p, q in zip(
                [5, 50, 95],
                np.percentile(nlos_shifts, [5, 50, 95]),
            ):
                print(f"    p{p:02d}: {q:+8.2f}")
            print(f"    max: {nlos_shifts.max():+8.2f}")

    # ---- Allocate output ----
    H_all = np.zeros((num_rx, num_tx, M, N, T_fixed), dtype=np.complex64)
    n_used_all = np.zeros((num_rx, num_tx), dtype=np.int32)

    skipped_empty  = 0
    skipped_window = 0
    generated      = 0

    for r in range(num_rx):
        for t in range(num_tx):
            k = int(n_kept_all[r, t])
            if k == 0:
                skipped_empty += 1
                continue

            # Reference delays to the earliest surviving path. The filter
            # already min-normalized tau_ns so the earliest kept path sits at 0,
            # for LoS and NLoS links alike. We deliberately do NOT rebase to
            # the geometric-LoS instant: it is computed from the (unknown at
            # inference) TX/RX positions, so baking it into the delay axis is
            # non-realizable and, for NLoS links, leaks the straight-line
            # range. shift_applied[r, t] is still recorded as a per-link
            # diagnostic but is no longer added to the taps.
            tau_ns_rel = tau_ns_all[r, t, :k]
            tpe_ns_rel = tau_per_el_all[r, t, :, :, :k]
            if not np.isfinite(tpe_ns_rel).all():
                raise ValueError(f"Nonfinite valid element delays at RX[{r}], TX[{t}]")

            # Drop paths that fall outside the tap window.
            # Delays are min-normalized (>= 0); the lower guard is defensive.
            in_window = (tau_ns_rel >= -dt * 1e9) & (tau_ns_rel < window_ns)
            if not np.any(in_window):
                skipped_window += 1
                continue

            k_in      = int(in_window.sum())
            a_k       = a_all[r, t, :k][in_window]
            tau_k_ns  = tau_ns_rel[in_window]
            omh_k     = omega_h_all[r, t, :k][in_window]
            omv_k     = omega_v_all[r, t, :k][in_window]
            tpe_k_ns  = tpe_ns_rel[:, :, in_window]

            H = channel_gen_link(
                a=a_k, tau_center_ns=tau_k_ns,
                omega_h=omh_k, omega_v=omv_k,
                tau_per_elem_ns=tpe_k_ns,
                f_carrier=f_carrier,
                M=M, N=N,
                dt=dt, T_fixed=T_fixed,
                pulse_fn=pulse_fn,
                pulse_Tsym=pulse_Tsym,
                pulse_beta=pulse_beta,
                include_true_time_delay=include_ttd,
                include_spatial_phase=include_sp,
            )
            with np.errstate(over="ignore", invalid="ignore"):
                H_all[r, t] = H.astype(np.complex64)
            if not np.isfinite(H_all[r, t]).all():
                raise ValueError(f"Nonfinite channel taps at RX[{r}], TX[{t}]; check amplitudes and pulse settings")
            n_used_all[r, t] = k_in
            generated += 1

    # ---- Sanity check: one link, in detail ----
    if generated > 0:
        rr, tt = map(int, np.argwhere(n_used_all > 0)[0])
        H_sample = H_all[rr, tt]
        mag = np.abs(H_sample)
        peak_elem = int(np.argmax(mag.max(axis=-1)))
        peak_tap = int(np.argmax(mag.reshape(M*N, T_fixed)[peak_elem]))
        peak_val = float(mag.reshape(M*N, T_fixed)[peak_elem, peak_tap])
        loc = (f"row {peak_elem // N}, col {peak_elem % N}"
               if M > 1 else f"col {peak_elem}")

        print(f"\n--- Sanity check: RX[{rr}] -> TX[{tt}] ---")
        print(f"  paths used       : {int(n_used_all[rr, tt])}")
        print(f"  |H| peak         : {peak_val:.4e} at {loc}, tap {peak_tap}")
        print(f"  peak tap time    : {peak_tap * dt * 1e9:.2f} ns")
        print(f"  geometric LoS    : {tau_los_ns_all[rr, tt]:.2f} ns (abs, diag)")
        print(f"  LoS kept?        : {bool(is_los_kept[rr, tt])}")
        print(f"  NLoS excess      : {shift_applied[rr, tt]:+.3f} ns "
              f"(recorded, NOT applied)")

    # ---- Save ----
    out_dir = os.path.join(project_dir, "data", "channel_data")
    os.makedirs(out_dir, exist_ok=True)
    stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
    out_file = os.path.join(out_dir, f"{stamp}_channels_{cfg.scenario_name}.npz")

    channel_meta = {
        "source_calibrated_npz": os.path.basename(in_file),
        "source_metadata":       calib_meta,
        "f_carrier_hz":          f_carrier,
        "dt_s":                  dt,
        "T_fixed":               T_fixed,
        "window_ns":             window_ns,
        "last_sample_ns":        (T_fixed - 1) * dt * 1e9,
        "path_window_policy":    "center delay in [-dt, T_fixed*dt); pulse tails outside saved grid truncated",
        "pulse_normalization":   "dimensionless analytic pulse; no discrete energy normalization",
        "carrier_phase_reference": "exp(-j*2*pi*fc*tau_center_relative) times centered spatial phase",
        "units": {"H": "sqrt(W), sampled effective channel per element", "dt_s": "s"},
        "array":                 {"M": M, "N": N,
                                  "d_h_wl": d_h_wl, "d_v_wl": d_v_wl},
        "pulse":                 {"type": pulse_type,
                                  "Tsym_s": pulse_Tsym,
                                  "beta": pulse_beta},
        "include_true_time_delay": include_ttd,
        "include_spatial_phase":   include_sp,
        "time_reference":          "earliest_surviving_path_per_link",
        "los_classification":      LOS_CLASSIFICATION,
        "los_label_source":        los_label_source,
        "delay_tolerance_ns":      DELAY_TOL_NS,
        "los_kept_count":          n_los_kept,
        "nlos_only_count":         n_nlos_only,
        "pathological_count":      n_pathological,
    }

    # Keep SNR diagnostics with the exact channels they describe, so BDP
    # generation never needs an independently selected calibration file.
    diagnostics = {}
    for key in ("snr_raw_db", "snr_bf_db", "snr_detector_db", "p_rx_total_dbm"):
        if key in data.files:
            value = data[key]
            if value.shape != (num_rx, num_tx):
                raise ValueError(f"Invalid calibrated diagnostic shape for {key}: {value.shape}")
            diagnostics[key] = value

    print(f"\nSaving to {out_file}")
    np.savez(
        out_file,
        H              = H_all,              # (R, T, M, N, T_fixed) complex64
        num_paths      = n_used_all,         # (R, T) int32
        is_los_kept    = is_los_kept,        # (R, T) bool
        tau_los_ns     = tau_los_ns_all,     # (R, T) float64
        shift_applied  = shift_applied,      # (R, T) float64
        rx_names       = rx_names,
        tx_names       = tx_names,
        rx_pos         = rx_pos,
        tx_pos         = tx_pos,
        rx_look_at     = data["rx_look_at"],
        metadata       = channel_meta,
        **diagnostics,
    )
    write_sidecar(out_file, stage="channel_taps_gen", cfg=cfg,
                  inputs={"calibrated_npz": in_file}, summary=channel_meta)
    print(f"Generated {generated} links | "
          f"{skipped_empty} empty, {skipped_window} out-of-window")


if __name__ == "__main__":
    main()
