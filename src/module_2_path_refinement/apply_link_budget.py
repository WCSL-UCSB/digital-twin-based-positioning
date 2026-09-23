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
#       * bf        : post-beamforming, pre-correlation
#       * detector  : post-beamforming, post-matched-filter (what a real
#                     detector would see)
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
    """Safe dB with a floor to avoid log(0)."""
    return 10.0 * np.log10(np.maximum(x, 1e-300))


def watts_to_dbm(p_w):
    return lin_to_db(p_w) + 30.0


def noise_power_watts(bandwidth_hz, noise_figure_db, temperature_k):
    """
    Thermal noise power at the receiver input (linear, watts).

        N = k * T * B * F
    """
    nf_lin = db_to_lin(noise_figure_db)
    return K_BOLTZMANN * temperature_k * bandwidth_hz * nf_lin


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def print_snr_distribution(label, snr_db, mask):
    """Pretty-print a percentile summary of an SNR tensor over a mask."""
    if not np.any(mask):
        print(f"  {label}: (no data)")
        return
    vals = snr_db[mask]
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

    # Diagnostic gain stages (not multiplied into `a`)
    array_gain_db      = float(lb.get("array_gain_db", 0.0))
    processing_gain_db = float(lb.get("processing_gain_db", 0.0))

    # ---- Calibration scalar for amplitudes ----
    # |a_calibrated_p|^2 = P_tx_lin * G_tx_lin * G_rx_lin * |a_p|^2  (watts)
    # Note: array_gain and processing_gain are NOT multiplied into `a`.
    # They belong to downstream processing (beamforming, matched filter)
    # and applying them here would double-count when those stages run.
    p_tx_w   = db_to_lin(p_tx_dbm - 30.0)
    g_tx_lin = db_to_lin(g_tx_db)
    g_rx_lin = db_to_lin(g_rx_db)
    amp_scale = np.sqrt(p_tx_w * g_tx_lin * g_rx_lin)

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
          f"(applied during beamforming)")
    print(f"  Processing gain     : {processing_gain_db:+.1f} dB "
          f"(applied during matched filter)")
    print(f"  Total downstream    : {array_gain_db + processing_gain_db:+.1f} dB")

    # ---- Load path data ----
    a_all    = data["a"]                        # (num_rx, num_tx, P) complex
    num_kept = data["num_paths_after_filter"]   # (num_rx, num_tx) int
    num_rx, num_tx, P_max = a_all.shape

    # ---- Calibrate amplitudes uniformly ----
    a_calibrated = a_all * amp_scale  # phases unchanged

    # ---- Per-link aggregate received power (watts, then dBm) ----
    # Pad entries beyond num_kept are zero and contribute nothing.
    power_per_path_w = np.abs(a_calibrated) ** 2      # (num_rx, num_tx, P)
    p_rx_total_w     = power_per_path_w.sum(axis=-1)  # (num_rx, num_tx)
    has_paths        = num_kept > 0

    p_rx_total_dbm = np.where(
        has_paths, watts_to_dbm(p_rx_total_w), -np.inf
    )

    # ---- Three-stage SNR diagnostics ----
    # Stage 1: per-element, pre-correlation, pre-beamforming
    snr_raw_db = np.where(
        has_paths, p_rx_total_dbm - n_power_dbm, -np.inf
    )
    # Stage 2: post-beamforming, pre-correlation
    snr_bf_db = snr_raw_db + array_gain_db
    # Stage 3: post-beamforming, post-matched-filter (what a detector sees)
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
        "Stage 2: post-beamforming, pre-correlation", snr_bf_db, has_paths
    )
    print_snr_distribution(
        "Stage 3: post-beamforming, post-matched-filter", snr_detector_db, has_paths
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
    stamp = time.strftime("%Y%m%d_%H%M%S")
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
            "noise_power_dbm":    float(n_power_dbm),
            "amplitude_scale_lin": float(amp_scale),
            # Diagnostic gains — recorded but not applied to `a`
            "array_gain_db":      array_gain_db,
            "processing_gain_db": processing_gain_db,
        },
        "units": {
            "a":               "sqrt(W), per element, pre-beamforming",
            "p_rx_total_dbm":  "dBm, aggregate per link",
            "snr_raw_db":      "dB, per-element pre-correlation",
            "snr_bf_db":       "dB, post-beamforming pre-correlation",
            "snr_detector_db": "dB, post-beamforming post-matched-filter",
        },
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