# =============================================================================
# apply_link_budget.py — Calibrate path amplitudes to absolute physical units
# =============================================================================
#
# Reads the filtered .npz from filter_and_adjust_paths.py and produces a
# calibrated version with:
#   - complex path amplitudes rescaled so |a_p|^2 is received power (W)
#     per path at a single antenna element, before beamforming / matched
#     filtering
#   - per-link aggregate received power (dBm)
#   - per-link SNR at three stages:
#       * raw       : per-element, pre-correlation, pre-beamforming
#       * bf        : ideal coherent-array SNR estimate, pre-correlation
#       * detector  : ideal array + matched-filter SNR estimate
#
# NO thresholding is applied. All SNR quantities are stored as diagnostics
# so downstream code can decide what to do with weak links. Detection,
# if needed, should happen after the matched filter has actually been
# applied, not on a predicted quantity here.
# =============================================================================

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.dataset_io import resolve_input, pinned, write_sidecar  # noqa: E402
from src.utils.radio_power import diagnostic_gains, positive_float, positive_int
import time
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

load_dotenv()

K_BOLTZMANN = 1.380649e-23  # J/K


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def db_to_lin(db):
    return 10.0 ** (db / 10.0)


def lin_to_db(x):
    """Convert nonnegative finite power ratios; zero power is -inf dB."""
    x = np.asarray(x, dtype=np.float64)
    if np.any(~np.isfinite(x)) or np.any(x < 0):
        raise ValueError("Linear power must be finite and nonnegative")
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(x)


def watts_to_dbm(p_w):
    return lin_to_db(p_w) + 30.0


def noise_power_watts(bandwidth_hz, noise_figure_db, temperature_k):
    """
    Thermal noise power at the receiver input (linear, watts).

        N = k * T * B * F
    """
    bandwidth_hz = positive_float(bandwidth_hz, "link_budget.bandwidth_hz")
    temperature_k = positive_float(temperature_k, "link_budget.temperature_k")
    if not np.isfinite(noise_figure_db) or noise_figure_db < 0:
        raise ValueError("link_budget.noise_figure_db must be finite and nonnegative")
    nf_lin = db_to_lin(noise_figure_db)
    return positive_float(K_BOLTZMANN * temperature_k * bandwidth_hz * nf_lin,
                          "receiver noise power")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def print_snr_distribution(label, snr_db, mask):
    """Pretty-print a percentile summary of an SNR tensor over a mask."""
    if not np.any(mask):
        print(f"  {label}: (no data)")
        return
    vals = snr_db[mask]
    finite = np.isfinite(vals)
    if not np.all(finite):
        print(f"  {label}: {int((~finite).sum())} zero-power link(s), SNR = -inf")
        vals = vals[finite]
    if vals.size == 0:
        return
    pcts = [5, 25, 50, 75, 95]
    qs = np.percentile(vals, pcts)
    print(f"  {label}")
    for p, q in zip(pcts, qs):
        print(f"    p{p:02d}  : {q:+7.2f} dB")
    print(f"    mean: {vals.mean():+7.2f} dB")
    print(f"    min : {vals.min():+7.2f} dB")
    print(f"    max : {vals.max():+7.2f} dB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../configs/scenarios")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    project_dir = cfg.project_dir

    # ---- Load filtered paths ----
    in_dir = os.path.join(project_dir, "data", "path_data", "refined_paths")
    in_file = resolve_input(
        os.path.join(in_dir, f"*_filtered_{cfg.scenario_name}.npz"),
        pinned(cfg, "filtered_npz"), label="filtered paths")
    data = np.load(in_file, allow_pickle=True)
    filter_meta = data["metadata"].item()

    # ---- Read link budget parameters ----
    lb = cfg.link_budget
    p_tx_dbm      = float(lb.tx_power_dbm)
    g_tx_db       = float(lb.get("tx_antenna_gain_db", 0.0))
    g_rx_db       = float(lb.get("rx_antenna_gain_db", 0.0))
    noise_fig_db  = float(lb.noise_figure_db)
    bandwidth_hz  = float(lb.bandwidth_hz)
    temperature_k = float(lb.get("temperature_k", 290.0))

    # Derive ideal diagnostic gains from the actual filtered array and pilot.
    # Old YAML array_gain_db/processing_gain_db values are deliberately unused.
    array_meta = filter_meta.get("rx_array")
    if not array_meta:
        raise ValueError("Filtered data lacks RX array metadata; regenerate refinement")
    rows = positive_int(array_meta["num_rows"], "filtered array rows")
    columns = positive_int(array_meta["num_cols"], "filtered array columns")
    sequence_length = positive_int(cfg.training_sequence.length, "pilot length")
    array_gain_db, processing_gain_db = diagnostic_gains(rows, columns, sequence_length)
    for name, value in (("tx_power_dbm", p_tx_dbm), ("tx_antenna_gain_db", g_tx_db),
                        ("rx_antenna_gain_db", g_rx_db)):
        if not np.isfinite(value):
            raise ValueError(f"link_budget.{name} must be finite")

    # ---- Calibration scalar for amplitudes ----
    # |a_calibrated_p|^2 = P_tx_lin * G_tx_lin * G_rx_lin * |a_p|^2  (watts)
    # Note: array_gain and processing_gain are NOT multiplied into `a`.
    # They belong to downstream processing (beamforming, matched filter)
    # and applying them here would double-count when those stages run.
    p_tx_w   = db_to_lin(p_tx_dbm - 30.0)
    g_tx_lin = db_to_lin(g_tx_db)
    g_rx_lin = db_to_lin(g_rx_db)
    amp_scale = positive_float(np.sqrt(p_tx_w * g_tx_lin * g_rx_lin), "amplitude scale")

    # ---- Noise floor (scalar, same for every link) ----
    n_power_w   = noise_power_watts(bandwidth_hz, noise_fig_db, temperature_k)
    n_power_dbm = watts_to_dbm(n_power_w)

    # ---- Print header ----
    print("\n--- Link Budget (calibration) ---")
    print(f"  TX power            : {p_tx_dbm:+.1f} dBm")
    print(f"  TX antenna gain     : {g_tx_db:+.1f} dB")
    print(f"  RX antenna gain     : {g_rx_db:+.1f} dB")
    print(f"  Bandwidth           : {bandwidth_hz/1e6:.2f} MHz")
    print(f"  Noise figure        : {noise_fig_db:+.1f} dB")
    print(f"  Temperature         : {temperature_k:.1f} K")
    print(f"  Noise power         : {n_power_dbm:+.2f} dBm")

    print("\n--- Downstream gains (diagnostic only, not in `a`) ---")
    print(f"  Array gain          : {array_gain_db:+.1f} dB "
          f"(ideal coherent-array estimate)")
    print(f"  Processing gain     : {processing_gain_db:+.1f} dB "
          f"(ideal matched-filter estimate)")
    print(f"  Total downstream    : {array_gain_db + processing_gain_db:+.1f} dB")

    # ---- Load path data ----
    a_all    = data["a"]                        # (num_rx, num_tx, P) complex
    num_kept = data["num_paths_after_filter"]   # (num_rx, num_tx) int
    if a_all.ndim != 3 or not all(a_all.shape[:2]):
        raise ValueError("Filtered amplitudes must have shape (RX, TX, path) with nonempty device axes")
    num_rx, num_tx, P_max = a_all.shape
    if (num_kept.shape != (num_rx, num_tx)
            or not np.issubdtype(num_kept.dtype, np.integer)
            or np.any(num_kept < 0) or np.any(num_kept > P_max)):
        raise ValueError("Filtered path counts must be (RX, TX) integers within the stored path axis")
    valid = np.arange(P_max) < num_kept[..., None]
    if not np.isfinite(a_all[valid]).all():
        raise ValueError("Valid filtered path coefficients must be finite")

    # ---- Calibrate amplitudes uniformly ----
    # Counts are authoritative; padding must never contribute to power.
    # Promote before squaring to retain very weak complex64 input paths.
    a_calibrated = np.where(valid, a_all, 0).astype(np.complex128) * amp_scale

    # ---- Per-link aggregate received power (watts, then dBm) ----
    # Incoherent sum: interference between paths is not included here.
    power_per_path_w = np.abs(a_calibrated) ** 2      # (num_rx, num_tx, P)
    p_rx_total_w     = power_per_path_w.sum(axis=-1)  # (num_rx, num_tx)
    if not np.isfinite(p_rx_total_w).all():
        raise ValueError("Calibrated received power overflowed; check amplitudes and link-budget settings")
    has_paths        = num_kept > 0

    p_rx_total_dbm = np.where(
        has_paths, watts_to_dbm(p_rx_total_w), -np.inf
    )

    # ---- Three-stage SNR diagnostics ----
    # Stage 1: per-element, pre-correlation, pre-beamforming
    snr_raw_db = np.where(
        has_paths, p_rx_total_dbm - n_power_dbm, -np.inf
    )
    # Stage 2: ideal coherent-array gain, not a measured beam output
    snr_bf_db = snr_raw_db + array_gain_db
    # Stage 3: ideal array + matched-filter gain, not realized detector SNR
    snr_detector_db = snr_bf_db + processing_gain_db

    # ---- Diagnostic distribution report ----
    n_links_total = int(num_rx * num_tx)
    n_links_kept  = int(has_paths.sum())

    print(f"\n--- Links ---")
    print(f"  Total links         : {n_links_total}")
    print(f"  Links with paths    : {n_links_kept}  "
          f"({100.0*n_links_kept/n_links_total:.1f}%)")

    print(f"\n--- SNR distributions (kept links only) ---")
    print_snr_distribution(
        "Stage 1: per-element, pre-correlation", snr_raw_db, has_paths
    )
    print_snr_distribution(
        "Stage 2: ideal array estimate, pre-correlation", snr_bf_db, has_paths
    )
    print_snr_distribution(
        "Stage 3: ideal array + matched-filter estimate", snr_detector_db, has_paths
    )

    # ---- Sanity check: one link in full detail ----
    if n_links_kept > 0:
        r, t = map(int, np.argwhere(has_paths)[0])
        k = int(num_kept[r, t])
        paths_p_w   = power_per_path_w[r, t, :k]
        paths_p_dbm = watts_to_dbm(paths_p_w)
        print(f"\n--- Sanity check: RX[{r}] -> TX[{t}] ---")
        print(f"  Surviving paths     : {k}")
        print(f"  Per-path P_rx (dBm) : "
              + ", ".join(f"{x:+.1f}" for x in paths_p_dbm))
        print(f"  Aggregate P_rx      : {p_rx_total_dbm[r, t]:+.2f} dBm")
        print(f"  Noise power         : {n_power_dbm:+.2f} dBm")
        print(f"  SNR stage 1 (raw)   : {snr_raw_db[r, t]:+.2f} dB")
        print(f"  SNR stage 2 (+BF)   : {snr_bf_db[r, t]:+.2f} dB")
        print(f"  SNR stage 3 (+MF)   : {snr_detector_db[r, t]:+.2f} dB")

    # ---- Save ----
    out_dir = os.path.join(project_dir, "data", "path_data", "calibrated_paths")
    os.makedirs(out_dir, exist_ok=True)
    stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
    out_file = os.path.join(
        out_dir, f"{stamp}_calibrated_{cfg.scenario_name}.npz"
    )

    lb_metadata = {
        "source_filtered_npz": os.path.basename(in_file),
        "source_metadata":     filter_meta,
        "link_budget": {
            "tx_power_dbm":       p_tx_dbm,
            "tx_antenna_gain_db": g_tx_db,
            "rx_antenna_gain_db": g_rx_db,
            "bandwidth_hz":       bandwidth_hz,
            "noise_figure_db":    noise_fig_db,
            "temperature_k":      temperature_k,
            "noise_power_w":      float(n_power_w),
            "noise_power_dbm":    float(n_power_dbm),
            "amplitude_scale_lin": float(amp_scale),
            # Diagnostic gains — recorded but not applied to `a`
            "array_gain_db":      array_gain_db,
            "processing_gain_db": processing_gain_db,
            "array_num_rows":     rows,
            "array_num_cols":     columns,
            "pilot_length":       sequence_length,
            "gain_source":        "array dimensions and pilot length",
        },
        "units": {
            "a":               "sqrt(W), per element, pre-beamforming",
            "p_rx_total_dbm":  "dBm, incoherent sum of valid path powers per element",
            "snr_raw_db":      "dB, per-element pre-correlation",
            "snr_bf_db":       "dB, ideal array-gain estimate, pre-correlation",
            "snr_detector_db": "dB, ideal array-plus-pilot-gain estimate",
        },
        "validity_source": "num_paths_after_filter",
        "zero_power_convention": "-inf dBm and -inf SNR, including valid zero-amplitude paths",
        "noise_model": "k_B * temperature_k * bandwidth_hz * 10**(noise_figure_db/10)",
        "snr_kind": "incoherent_path_power_with_ideal_processing_gains",
        "thresholding_note": (
            "No thresholding is applied. All SNR quantities are diagnostic. "
            "If detection is needed downstream, apply it on the empirical "
            "correlator output after channel_tap_gen + matched filter, not "
            "on the predicted values stored here."
        ),
        "calibration_note": (
            "Amplitudes rescaled uniformly by sqrt(P_tx * G_tx * G_rx). "
            "Phases unchanged. Array gain and processing gain are NOT "
            "multiplied into `a` — they belong to downstream stages and "
            "are recorded here as diagnostic metadata only."
        ),
    }

    # Replace `a` and `metadata`, pass everything else through unchanged.
    save_dict = {
        "a":                a_calibrated,
        "p_rx_total_dbm":   p_rx_total_dbm.astype(np.float64),
        "snr_raw_db":       snr_raw_db.astype(np.float64),
        "snr_bf_db":        snr_bf_db.astype(np.float64),
        "snr_detector_db":  snr_detector_db.astype(np.float64),
        "metadata":         lb_metadata,
    }
    for key in data.files:
        if key in save_dict or key == "metadata":
            continue
        save_dict[key] = data[key]

    print(f"\nSaving calibrated paths to {out_file}")
    np.savez(out_file, **save_dict)
    write_sidecar(out_file, stage="apply_link_budget", cfg=cfg,
                  inputs={"filtered_npz": in_file}, summary=lb_metadata)
    print("Done.")


if __name__ == "__main__":
    main()
