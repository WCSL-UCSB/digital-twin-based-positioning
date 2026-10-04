# =============================================================================
# beam_delay_gen.py — Beam-Delay Profile generation for CNN training
# =============================================================================
#
# Consumes the consolidated channel-taps .npz from channel_taps_gen.py and
# produces complex beam-delay profiles (BDPs) per link plus a single
# consolidated training dataset.
#
# Multi-BS support:
#   When num_rx > 1, the output dataset groups BDPs by TX position and
#   stacks them along a new BS-channel axis:
#     ULA: Z shape = (num_positions, N_beams, Lh, K)
#     UPA: Z shape = (num_positions, M, N_beams, Lh, K)
#   When num_rx == 1 (single-BS), the output is unchanged:
#     ULA: Z shape = (num_positions, N_beams, Lh)
#     UPA: Z shape = (num_positions, M, N_beams, Lh)
#   A BS without surviving paths contributes zeros, regardless of the noise setting.
#
# Pipeline per link with surviving paths:
#   1. Cyclic-prefixed Zadoff-Chu pilot is transmitted through each element.
#   2. Add receiver noise, then beamspace DFT across columns and rows.
#   3. Per-beam frequency-domain matched filter against ZC conjugate.
#   4. Truncate to Lh delay bins → complex beam-delay profile.
#
# Canonical dataset is COMPLEX. Log-magnitude conversion, dynamic-range
# clipping, and normalization are training-time preprocessing decisions
# and belong in the data loader, not here. The sanity plot uses a default
# display range just for visualization.
#
# Filtering policy: only "link has at least one surviving path." All other
# per-sample metadata (is_los_kept, shift_ns, snr_*, p_rx_total_dbm) is
# carried through so downstream code can mask or weight without regenerating.
# =============================================================================

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.dataset_io import resolve_input, pinned, write_sidecar  # noqa: E402
from src.utils.radio_power import diagnostic_gains, positive_int
from src.utils.receiver_noise import recorded_noise_budget, link_noise_rng, complex_receiver_noise
import json
import math
import time
import uuid
from datetime import datetime
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

load_dotenv()


# ---------------------------------------------------------------------------
# Training sequence
# ---------------------------------------------------------------------------

def gen_zadoff_chu(length, root):
    """Generate a Zadoff-Chu sequence of given length and root index."""
    length = positive_int(length, "pilot length")
    root = positive_int(root, "Zadoff-Chu root")
    if length < 2 or root >= length or math.gcd(root, length) != 1:
        raise ValueError("Zadoff-Chu root must be smaller than and coprime to its length")
    n = np.arange(length)
    return np.exp(-1j * np.pi * root * n * (n + length % 2) / length)


# ---------------------------------------------------------------------------
# BDP core (vectorized across elements)
# ---------------------------------------------------------------------------

def bdp_from_H(H, s_cp, S_conj, Lh, Lcp, Nseq, *, noise_power_w=0.0,
               rng=None, return_diagnostics=False):
    """
    Convert a channel-taps tensor to a complex beam-delay profile.

    Parameters
    ----------
    H      : (M, N, T_fixed) complex — per-element channel taps
    s_cp   : (Lcp + Nseq,) complex — cyclic-prefixed ZC sequence
    S_conj : (Nseq,) complex — conj(FFT(s))
    Lh     : int — output delay bins, must equal the saved channel tap count
    Lcp    : int — cyclic prefix length
    Nseq   : int — training sequence length
    noise_power_w : float — receiver noise power per complex antenna sample
    rng : numpy.random.Generator — independent stream for this link
    return_diagnostics : bool — also return mean per-element pilot/noise powers

    Returns
    -------
    Z : (M, N_beams, Lh) complex. For a ULA (M=1) the caller squeezes.
    """
    if H.ndim != 3 or not all(H.shape) or not np.isfinite(H).all():
        raise ValueError("H must be a nonempty, finite (M, N, taps) tensor")
    M, N, T_fixed = H.shape
    if not 1 <= Lh <= Nseq or Lcp < Lh - 1 or len(s_cp) != Lcp + Nseq:
        raise ValueError("Require 1 <= delay bins <= pilot length and a sufficient cyclic prefix")
    if np.shape(S_conj) != (Nseq,):
        raise ValueError("Pilot spectrum length does not match Nseq")

    # A BDP uses the complete saved channel; it cannot extend its physical
    # delay window or silently discard echoes before pilot propagation.
    if Lh != T_fixed:
        raise ValueError(
            f"BDP delay bins ({Lh}) must equal the channel tap count ({T_fixed}); "
            "regenerate channel taps to change the delay window.")

    # ---- Step 1: FFT-based convolution across all elements at once ----
    conv_len = len(s_cp) + Lh - 1
    n_fft = 1 << int(np.ceil(np.log2(conv_len)))

    S_cp_f = np.fft.fft(s_cp, n=n_fft)
    H_f    = np.fft.fft(H, n=n_fft, axis=-1)
    Y_full = np.fft.ifft(H_f * S_cp_f[np.newaxis, np.newaxis, :],
                         n=n_fft, axis=-1)

    # Signal-settled window
    Y_ant = Y_full[:, :, Lcp:Lcp + Nseq]
    signal_power_w = float(np.mean(np.abs(Y_ant)**2)) if return_diagnostics else None
    measured_noise_w = 0.0
    if noise_power_w != 0:
        noise = complex_receiver_noise(Y_ant.shape, noise_power_w, rng)
        if return_diagnostics:
            measured_noise_w = float(np.mean(np.abs(noise)**2))
        Y_ant = Y_ant + noise

    # ---- Step 2: beamspace DFT ----
    Y_beam = np.fft.fftshift(
        np.fft.fft(Y_ant, n=N, axis=1), axes=1
    ) / np.sqrt(N)

    if M > 1:
        Y_beam = np.fft.fftshift(
            np.fft.fft(Y_beam, n=M, axis=0), axes=0
        ) / np.sqrt(M)

    # ---- Step 3: per-beam frequency-domain matched filter ----
    Y_f = np.fft.fft(Y_beam, n=Nseq, axis=-1)
    Z   = np.fft.ifft(Y_f * S_conj[np.newaxis, np.newaxis, :],
                      n=Nseq, axis=-1)

    # ---- Step 4: truncate to Lh delay bins ----
    result = Z[:, :, :Lh]
    if return_diagnostics:
        return result, {"signal_power_w": signal_power_w, "noise_power_w": measured_noise_w}
    return result


def _source_snr(ch_data, channel_meta, cfg):
    """Use embedded diagnostics or the exact recorded calibration, never latest."""
    required = ("snr_raw_db", "p_rx_total_dbm")
    recorded = channel_meta.get("source_calibrated_npz")
    explicit = pinned(cfg, "calibrated_npz")
    if explicit and (not recorded or os.path.basename(explicit) != os.path.basename(recorded)):
        raise ValueError("inputs.calibrated_npz does not match the channel's recorded calibration")
    if all(key in ch_data.files for key in required):
        return {key: ch_data[key] for key in required}, None, "embedded_channel_diagnostics"
    if not recorded:
        print("[Info] Channel has no recorded calibration; predicted SNR fields omitted.")
        return {}, None, "unavailable"
    path = explicit or os.path.join(str(cfg.project_dir), "data", "path_data",
                                   "calibrated_paths", os.path.basename(recorded))
    if not os.path.isfile(path):
        if explicit:
            raise FileNotFoundError(f"Pinned source calibration not found: {path}")
        print(f"[Info] Recorded calibration unavailable: {path}; predicted SNR fields omitted.")
        return {}, None, "unavailable"
    with np.load(path, allow_pickle=True) as calibrated:
        expected_meta = channel_meta.get("source_metadata")
        if expected_meta is None or json.dumps(calibrated["metadata"].item(), sort_keys=True,
                                               default=str) != json.dumps(expected_meta, sort_keys=True,
                                                                          default=str):
            raise ValueError("Recorded calibration metadata does not match the channel source")
        for key in ("tx_names", "rx_names", "tx_pos", "rx_pos"):
            if key not in calibrated.files or not np.array_equal(calibrated[key], ch_data[key]):
                raise ValueError(f"Recorded calibration {key} does not match channel device order/layout")
        return {key: calibrated[key] for key in required}, path, "recorded_calibration"


# ---------------------------------------------------------------------------
# Display-only log conversion (for sanity plot, not saved to dataset)
# ---------------------------------------------------------------------------

def _complex_to_display_log(Z, display_range_db=50.0):
    """
    Convert complex BDP to a normalized [0, 1] log-magnitude image FOR
    DISPLAY ONLY. Not saved to disk.
    """
    mag = np.abs(Z)
    peak = mag.max() + 1e-12
    mag_norm = mag / peak
    X = 20.0 * np.log10(mag_norm + 1e-12)
    X = np.maximum(X, -display_range_db)
    X = (X + display_range_db) / display_range_db
    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sanity plot
# ---------------------------------------------------------------------------

def save_sanity_plot(Z_samples, titles, out_path, M, N, K,
                     display_range_db=50.0):
    """
    Grid plot of BDPs with labels. For multi-BS, shows one row per sample
    with columns for each BS.
    """
    n = len(Z_samples)
    if n == 0:
        return

    if K > 1:
        # Multi-BS: one row per sample, one column per BS
        rows = min(n, 4)
        cols = K
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        axes = np.atleast_2d(axes)

        for row in range(rows):
            Z = Z_samples[row]  # (..., Lh, K) or (M, ..., Lh, K)
            for k in range(K):
                ax = axes[row, k]
                if M > 1:
                    Z_k = Z[:, :, :, k]  # (M, N, Lh)
                else:
                    Z_k = Z[:, :, k]     # (N, Lh)
                img = _complex_to_display_log(Z_k, display_range_db)
                if img.ndim == 3:
                    peak_el = int(np.argmax(img.max(axis=(1, 2))))
                    img = img[peak_el]
                ax.imshow(img, aspect="auto", origin="lower", cmap="inferno",
                          vmin=0, vmax=1)
                if row == 0:
                    ax.set_title(f"BS {k}", fontsize=9)
                if k == 0:
                    ax.set_ylabel(titles[row], fontsize=8)
                ax.set_xlabel("Delay bin")
    else:
        # Single-BS: original layout
        cols = 3
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        axes = np.atleast_2d(axes).ravel()

        for ax, Z, title in zip(axes, Z_samples, titles):
            img = _complex_to_display_log(Z, display_range_db)
            if img.ndim == 3:
                peak_el = int(np.argmax(img.max(axis=(1, 2))))
                img = img[peak_el]
            ax.imshow(img, aspect="auto", origin="lower", cmap="inferno",
                      vmin=0, vmax=1)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Delay bin")
            ax.set_ylabel("Az beam")
        for ax in axes[n:]:
            ax.axis("off")

    plt.suptitle(f"Display log range: {display_range_db:.0f} dB "
                 f"(for visualization only)", fontsize=8, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../configs/scenarios")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    project_dir = cfg.project_dir

    # ---- Load channels ----
    in_dir = os.path.join(project_dir, "data", "channel_data")
    ch_file = resolve_input(
        os.path.join(in_dir, f"*_channels_{cfg.scenario_name}.npz"),
        pinned(cfg, "channels_npz"), label="channels")
    ch_data = np.load(ch_file, allow_pickle=True)
    channel_meta = ch_data["metadata"].item()

    H_all       = ch_data["H"]               # (R, T, M, N, T_fixed) complex64
    num_paths   = ch_data["num_paths"]       # (R, T) int32
    is_los_kept = ch_data["is_los_kept"]     # (R, T) bool
    shift_ns    = ch_data["shift_applied"]   # (R, T) float64
    tx_pos      = ch_data["tx_pos"]          # (T, 3)
    rx_pos      = ch_data["rx_pos"]          # (R, 3)
    tx_names    = ch_data["tx_names"]
    rx_names    = ch_data["rx_names"]

    if H_all.ndim != 5 or not all(H_all.shape):
        raise ValueError("H must have nonempty (RX, TX, M, N, taps) axes")
    num_rx, num_tx, M, N, T_fixed = H_all.shape
    link_shape = (num_rx, num_tx)
    if (num_paths.shape != link_shape or not np.issubdtype(num_paths.dtype, np.integer)
            or np.any(num_paths < 0)):
        raise ValueError("num_paths must be nonnegative integers with shape (RX, TX)")
    if is_los_kept.shape != link_shape or is_los_kept.dtype != np.bool_:
        raise ValueError("is_los_kept must be a boolean (RX, TX) array")
    if np.any(is_los_kept & (num_paths == 0)):
        raise ValueError("An empty link cannot be marked is_los_kept")
    for name, value, shape in (("shift_applied", shift_ns, link_shape),
                               ("tx_pos", tx_pos, (num_tx, 3)), ("rx_pos", rx_pos, (num_rx, 3))):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name} must have shape {shape} and finite values")
    if tx_names.shape != (num_tx,) or rx_names.shape != (num_rx,):
        raise ValueError("Device names must agree with the channel tensor axes")
    K = num_rx  # number of base stations

    # The saved channel determines both the BDP length and cyclic prefix.
    # A current config value is a consistency check, not a resizing request.
    Lh = positive_int(T_fixed, "saved channel tap count")
    configured_taps = cfg.training_sequence.get("max_delay_bins")
    if configured_taps is not None:
        configured_taps = positive_int(configured_taps, "training_sequence.max_delay_bins")
        if configured_taps != Lh:
            raise ValueError(
                f"training_sequence.max_delay_bins={configured_taps}, but the saved "
                f"channel has {Lh} taps. Restore max_delay_bins={Lh} or regenerate "
                "channel taps from calibrated paths before BDP generation.")

    # Original noise budget belongs to H; current YAML cannot silently replace it.
    noise_cfg = cfg.get("receiver_noise", {}) or {}
    noise_enabled = bool(noise_cfg.get("enabled", True))
    noise_seed = noise_cfg.get("seed", 71)
    link_noise_rng(noise_seed, "validation", "validation")  # validate even for empty datasets
    if noise_enabled:
        noise_power_w, noise_bandwidth, noise_dt = recorded_noise_budget(channel_meta)
        if len(set(map(str, tx_names))) != num_tx or len(set(map(str, rx_names))) != num_rx:
            raise ValueError("Unique TX/RX names are required for independent noise streams")
    else:
        noise_power_w, noise_bandwidth, noise_dt = 0.0, None, channel_meta.get("dt_s")

    # Prefer diagnostics saved with H; legacy files follow their exact source.
    snr_data, calib_file, snr_source = _source_snr(ch_data, channel_meta, cfg)
    has_snr = bool(snr_data)
    if has_snr:
        for key, value in snr_data.items():
            if value.shape != (num_rx, num_tx):
                raise ValueError(f"{key} shape does not match channel device layout")
        snr_raw_db = snr_data["snr_raw_db"]
        p_rx_total_dbm = snr_data["p_rx_total_dbm"]

    # ---- Training sequence ----
    seq_type = str(cfg.training_sequence.type).strip().lower()
    Nseq = positive_int(cfg.training_sequence.length, "pilot length")
    if Lh > Nseq:
        raise ValueError("Saved channel tap count must not exceed pilot length")
    Lcp = Lh - 1
    array_gain_db, processing_gain_db = diagnostic_gains(M, N, Nseq)
    if has_snr:
        snr_bf_db = snr_raw_db + array_gain_db
        snr_detector_db = snr_bf_db + processing_gain_db

    if seq_type == "zadoff_chu":
        root = positive_int(cfg.training_sequence.root_index, "Zadoff-Chu root")
        s = gen_zadoff_chu(Nseq, root)
    else:
        raise ValueError(f"Unknown training sequence type: '{seq_type}'")

    s_cp   = np.concatenate([s[-Lcp:], s]) if Lcp else s.copy()
    S_conj = np.conj(np.fft.fft(s, n=Nseq))

    # ---- Labels ----
    label_dim = positive_int(cfg.training_sequence.get("label_dim", 2), "label_dim")
    if label_dim not in (2, 3):
        raise ValueError(f"label_dim must be 2 or 3, got {label_dim}")

    # ---- Summary ----
    array_type = "UPA" if M > 1 else "ULA"
    multi_bs = K > 1
    print(f"\n--- BDP Generation ({array_type}) ---")
    print(f"Array        : {M} x {N} = {M*N} elements")
    print(f"Base stations: {K}  {'(multi-BS)' if multi_bs else '(single-BS)'}")
    print(f"Pilot        : {seq_type} length={Nseq} root={root}")
    print(f"Delay bins   : Lh={Lh}, CP={Lcp}")
    print(f"Total links  : {num_rx * num_tx}")
    print(f"Label dim    : {label_dim}D")
    print(f"Output dtype : complex64 (log conversion deferred to training)")
    print(f"Receiver noise: {'enabled' if noise_enabled else 'disabled'}, "
          f"power={noise_power_w:.6g} W/element, seed={noise_seed}")

    # ================================================================
    #  Coverage analysis & dataset allocation
    # ================================================================
    # Single BDP shape (after M==1 squeeze): (N, Lh)  or (M, N, Lh) for UPA
    bdp_shape_single = (N, Lh) if M == 1 else (M, N, Lh)

    # For each TX position, we need at least one BS to have paths.
    # has_any_bs[t] = True if any BS sees this TX
    has_any_bs = np.any(num_paths > 0, axis=0)  # (T,) bool
    n_positions = int(has_any_bs.sum())

    # Coverage statistics per BS
    print(f"\n--- Multi-BS Coverage ---")
    for r in range(K):
        bs_coverage = int((num_paths[r] > 0).sum())
        print(f"  {rx_names[r]:12s}: {bs_coverage}/{num_tx} positions "
              f"({100*bs_coverage/num_tx:.1f}%)")
    print(f"  Union (any BS) : {n_positions}/{num_tx} positions "
          f"({100*n_positions/num_tx:.1f}%)")

    if multi_bs:
        for r in range(K):
            n_zeropad = int(has_any_bs.sum() - (num_paths[r] > 0).sum())
            print(f"  {rx_names[r]:12s}: {n_zeropad} positions will be zero-filled")

    # Allocate output arrays
    if multi_bs:
        # Stack K BS channels along the last axis
        # ULA: (n_positions, N, Lh, K)
        # UPA: (n_positions, M, N, Lh, K)
        Z_shape = (n_positions,) + bdp_shape_single + (K,)
    else:
        # Single-BS: no extra axis
        # ULA: (n_positions, N, Lh)
        # UPA: (n_positions, M, N, Lh)
        Z_shape = (n_positions,) + bdp_shape_single

    Z_all = np.zeros(Z_shape, dtype=np.complex64)
    Y_all = np.zeros((n_positions, label_dim), dtype=np.float32)

    # Per-sample metadata: per-BS info stored as (n_positions, K) or scalar
    is_los_out    = np.zeros((n_positions, K), dtype=bool)
    shift_ns_out  = np.zeros((n_positions, K), dtype=np.float32)
    has_paths_out = np.zeros((n_positions, K), dtype=bool)
    pilot_signal_power_w = np.zeros((n_positions, K), dtype=np.float64)
    realized_noise_power_w = np.zeros((n_positions, K), dtype=np.float64)

    if has_snr:
        snr_raw_out = np.zeros((n_positions, K), dtype=np.float32)
        snr_bf_out  = np.zeros((n_positions, K), dtype=np.float32)
        snr_det_out = np.zeros((n_positions, K), dtype=np.float32)
        p_rx_out    = np.zeros((n_positions, K), dtype=np.float32)
    else:
        snr_raw_out = snr_bf_out = snr_det_out = p_rx_out = None

    # TX indices that made it into the dataset (for reference)
    tx_idx_out = np.zeros(n_positions, dtype=np.int32)

    # ================================================================
    #  Compute BDPs directly into Z_all (no per-link intermediate buffer)
    # ================================================================
    # For every retained position, a BS with no channel paths keeps its zero
    # profile and power diagnostics. Noise is generated only for covered links.
    n_generated = 0
    t0 = time.time()

    valid_tx_indices = np.where(has_any_bs)[0]

    out_idx = 0
    for t in valid_tx_indices:
        Y_all[out_idx] = tx_pos[t, :label_dim]
        tx_idx_out[out_idx] = t

        for r in range(K):
            has = num_paths[r, t] > 0
            has_paths_out[out_idx, r] = has
            is_los_out[out_idx, r]    = bool(is_los_kept[r, t])
            shift_ns_out[out_idx, r]  = float(shift_ns[r, t])

            if has_snr:
                snr_raw_out[out_idx, r] = float(snr_raw_db[r, t])
                snr_bf_out[out_idx, r]  = float(snr_bf_db[r, t])
                snr_det_out[out_idx, r] = float(snr_detector_db[r, t])
                p_rx_out[out_idx, r]    = float(p_rx_total_dbm[r, t])

            if not has:
                continue

            H = H_all[r, t].astype(np.complex128)
            rng = link_noise_rng(noise_seed, tx_names[t], rx_names[r]) if noise_enabled else None
            Z, powers = bdp_from_H(H, s_cp, S_conj, Lh, Lcp, Nseq,
                                  noise_power_w=noise_power_w, rng=rng,
                                  return_diagnostics=True)
            pilot_signal_power_w[out_idx, r] = powers["signal_power_w"]
            realized_noise_power_w[out_idx, r] = powers["noise_power_w"]
            if M == 1:
                Z = Z[0]  # (N, Lh)

            with np.errstate(over="ignore", invalid="ignore"):
                Z_c64 = Z.astype(np.complex64)
            if not np.isfinite(Z_c64).all():
                raise ValueError(f"Nonfinite BDP at RX[{r}], TX[{t}]; check channel amplitudes")
            if multi_bs:
                Z_all[out_idx][..., r] = Z_c64
            else:
                Z_all[out_idx] = Z_c64
            n_generated += 1

            if n_generated % 500 == 0:
                rate = n_generated / max(time.time() - t0, 1e-6)
                print(f"  {n_generated} links processed  ({rate:.1f} links/s)")

        out_idx += 1

    assert out_idx == n_positions
    elapsed = time.time() - t0
    print(f"\nBDP generation complete: {n_generated} links in {elapsed:.1f}s")

    # Free the (R, T, M, N, T_fixed) channel tensor before the savez step,
    # which itself transiently buffers a copy of the largest output array.
    import gc
    del H_all
    gc.collect()

    print(f"\nDataset shapes:")
    print(f"  Z: {Z_all.shape}  dtype={Z_all.dtype}")
    print(f"  Y: {Y_all.shape}  dtype={Y_all.dtype}")
    if multi_bs:
        print(f"  Z axes: (num_positions, "
              + ("M, " if M > 1 else "")
              + f"N_beams={N}, Lh={Lh}, K={K})")
    else:
        print(f"  Z axes: (num_positions, "
              + ("M, " if M > 1 else "")
              + f"N_beams={N}, Lh={Lh})")

    # ---- Save ----
    out_dir = os.path.join(project_dir, "data", "ml_datasets", cfg.scenario_name)
    os.makedirs(out_dir, exist_ok=True)
    # Retain the timestamp prefix used by dataset discovery; give every run
    # its own dataset, sidecar, and plot names, including concurrent runs.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8]
    ds_file = os.path.join(out_dir, f"{stamp}_bdp_dataset_{cfg.scenario_name}.npz")

    # Z axes description
    if multi_bs:
        z_axes_str = (
            f"(num_positions, M, N_beams, Lh, K={K})" if M > 1
            else f"(num_positions, N_beams, Lh, K={K})"
        )
    else:
        z_axes_str = (
            "(num_positions, M, N_beams, Lh)" if M > 1
            else "(num_positions, N_beams, Lh)"
        )

    dataset_meta = {
        "source_channels_npz":   os.path.basename(ch_file),
        "source_channels_meta":  channel_meta,
        "source_calibrated_npz": channel_meta.get("source_calibrated_npz"),
        "snr_source": snr_source,
        "snr_kind": "incoherent_path_power_with_ideal_processing_gains",
        "diagnostic_gains": {"array_gain_db": array_gain_db,
                             "processing_gain_db": processing_gain_db},
        "beam_transform": "unitary negative-exponent DFT across columns and rows, fftshift",
        "correlation_normalization": "unnormalized circular pilot correlation; noiseless Z = pilot_length * beamspace(H)",
        "noise_power_w_per_complex_output_bin": Nseq * noise_power_w,
        "output_noise_power_scope": "links with has_paths=true; zero for missing links",
        "receiver_noise": {
            "enabled": noise_enabled, "model": "independent_complex_white_gaussian",
            "noise_power_w_per_element": noise_power_w,
            "bandwidth_hz": noise_bandwidth, "dt_s": noise_dt,
            "seed": int(noise_seed),
            "stream": "PCG64/SeedSequence(seed,SHA256(JSON[tx_name,rx_name]))",
            "injection_point": "received antenna pilot samples before beamforming",
            "missing_link": "zero",
            "applied_to": "links with num_paths > 0",
        },
        "power_diagnostics": {
            "pilot_signal_power_w": "clean pilot power averaged over samples and antenna elements",
            "realized_noise_power_w": "generated noise power averaged over samples and antenna elements; zero for missing links",
        },
        "training_sequence": {
            "type":   seq_type,
            "length": Nseq,
            "root":   root,
            "Lh":     Lh,
            "Lcp":    Lcp,
        },
        "label_dim":        label_dim,
        "array":            {"M": M, "N": N},
        "num_base_stations": K,
        "bs_names":         list(rx_names),
        "num_positions":    int(n_positions),
        "multi_bs":         multi_bs,
        "num_los_kept":     {str(rx_names[r]): int((is_los_out[:, r] &
                             has_paths_out[:, r]).sum()) for r in range(K)},
        "num_nlos_only":    {str(rx_names[r]): int((~is_los_out[:, r] &
                             has_paths_out[:, r]).sum()) for r in range(K)},
        "coverage":         {str(rx_names[r]): int(has_paths_out[:, r].sum())
                             for r in range(K)},
        "zero_padded":      {str(rx_names[r]): int((~has_paths_out[:, r]).sum())
                             for r in range(K)},
        "noise_only":       {str(rx_names[r]): 0
                             for r in range(K)},
        "Z_dtype":          "complex64",
        "Z_axes":           z_axes_str,
        "filtering_note": (
            "Only 'at least one BS has surviving paths' filter applied. "
            "A BS without surviving paths contributes zeros; no noise is generated "
            "for that link. Per-sample per-BS metadata lets "
            "training code filter or weight without regenerating."
        ),
        "preprocessing_note": (
            "Z is the CANONICAL complex beam-delay profile. Log-magnitude "
            "conversion, dynamic-range clipping, and normalization are "
            "training-time decisions and must be applied in the data "
            "loader — they are NOT baked into this dataset."
        ),
    }

    save_dict = {
        "Z":             Z_all,
        "Y":             Y_all,
        "has_paths":     has_paths_out,       # (n_positions, K) bool
        "pilot_signal_power_w": pilot_signal_power_w,
        "realized_noise_power_w": realized_noise_power_w,
        "is_los_kept":   is_los_out,          # (n_positions, K) bool
        "shift_ns":      shift_ns_out,        # (n_positions, K) float32
        "tx_idx":        tx_idx_out,           # (n_positions,) int32
        "tx_names":      tx_names,
        "rx_names":      rx_names,
        "tx_pos":        tx_pos,
        "rx_pos":        rx_pos,
        "metadata":      dataset_meta,
    }
    if has_snr:
        save_dict["snr_raw_db"]      = snr_raw_out      # (n_positions, K)
        save_dict["snr_bf_db"]       = snr_bf_out        # (n_positions, K)
        save_dict["snr_detector_db"] = snr_det_out       # (n_positions, K)
        save_dict["p_rx_total_dbm"]  = p_rx_out          # (n_positions, K)

    # Exclusive creation prevents even an unlikely name collision overwriting
    # another run. Pass the handle so np.savez cannot reopen/truncate the path.
    with open(ds_file, "xb") as output:
        np.savez(output, **save_dict)
    print(f"\nSaved: {ds_file}")
    write_sidecar(ds_file, stage="beam_delay_gen", cfg=cfg,
                  inputs={"channels_npz": ch_file, "calibrated_npz": calib_file},
                  summary=dataset_meta)

    # ---- Sanity plot ----
    if n_positions > 0:
        picks_Z, titles = [], []

        # Pick a spread of samples: some LoS-kept (from any BS), some NLoS
        any_los = np.any(is_los_out & has_paths_out, axis=1)
        los_idx  = np.where(any_los)[0]
        nlos_idx = np.where(~any_los)[0]

        def pick_spread(arr, n):
            if len(arr) == 0:
                return []
            if len(arr) <= n:
                return list(arr)
            step = max(1, len(arr) // n)
            return list(arr[::step][:n])

        for i in pick_spread(los_idx, 3):
            label = f"pos {tx_idx_out[i]} LoS-kept"
            picks_Z.append(Z_all[i])
            titles.append(label)

        for i in pick_spread(nlos_idx, 3):
            label = f"pos {tx_idx_out[i]} NLoS"
            picks_Z.append(Z_all[i])
            titles.append(label)

        if picks_Z:
            fig_path = os.path.join(
                out_dir, f"{stamp}_bdp_sanity_{cfg.scenario_name}.png"
            )
            save_sanity_plot(picks_Z, titles, fig_path, M, N, K,
                             display_range_db=50.0)
            print(f"Sanity plot: {fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
