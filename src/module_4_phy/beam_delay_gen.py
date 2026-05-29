# =============================================================================
# bdp_gen.py — Beam-Delay Profile generation for CNN training
# =============================================================================
#
# Consumes the consolidated channel-taps .npz from channel_tap_gen.py and
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
#   Zero-padding is applied when a BS has no surviving paths to a position.
#
# Pipeline per link:
#   1. Cyclic-prefixed Zadoff-Chu pilot is transmitted through each element.
#   2. Beamspace DFT across N columns (azimuth) and, if M > 1, M rows (elev).
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
import glob
import time
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
    n = np.arange(length)
    return np.exp(-1j * np.pi * root * n * (n + 1) / length)


# ---------------------------------------------------------------------------
# BDP core (vectorized across elements)
# ---------------------------------------------------------------------------

def bdp_from_H(H, s_cp, S_conj, Lh, Lcp, Nseq):
    """
    Convert a channel-taps tensor to a complex beam-delay profile.

    Parameters
    ----------
    H      : (M, N, T_fixed) complex — per-element channel taps
    s_cp   : (Lcp + Nseq,) complex — cyclic-prefixed ZC sequence
    S_conj : (Nseq,) complex — conj(FFT(s))
    Lh     : int — output delay bins
    Lcp    : int — cyclic prefix length
    Nseq   : int — training sequence length

    Returns
    -------
    Z : (M, N_beams, Lh) complex. For a ULA (M=1) the caller squeezes.
    """
    M, N, T_fixed = H.shape

    # Pad / truncate the tap axis to Lh
    if T_fixed < Lh:
        H_work = np.pad(H, ((0, 0), (0, 0), (0, Lh - T_fixed)))
    elif T_fixed > Lh:
        H_work = H[:, :, :Lh]
    else:
        H_work = H

    # ---- Step 1: FFT-based convolution across all elements at once ----
    conv_len = len(s_cp) + Lh - 1
    n_fft = 1 << int(np.ceil(np.log2(conv_len)))

    S_cp_f = np.fft.fft(s_cp, n=n_fft)
    H_f    = np.fft.fft(H_work, n=n_fft, axis=-1)
    Y_full = np.fft.ifft(H_f * S_cp_f[np.newaxis, np.newaxis, :],
                         n=n_fft, axis=-1)

    # Signal-settled window
    Y_ant = Y_full[:, :, Lcp:Lcp + Nseq]

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
    return Z[:, :, :Lh]


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

def get_latest_channels_npz(path_dir, scenario_name):
    pattern = os.path.join(path_dir, f"*_channels_{scenario_name}.npz")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No channels .npz matching: {pattern}")
    return max(files, key=os.path.getctime)


def get_latest_calibrated_npz(path_dir, scenario_name):
    pattern = os.path.join(path_dir, f"*_calibrated_{scenario_name}.npz")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)


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
    ch_file = get_latest_channels_npz(in_dir, cfg.scenario_name)
    print(f"Loading channels: {os.path.basename(ch_file)}")
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

    num_rx, num_tx, M, N, T_fixed = H_all.shape
    K = num_rx  # number of base stations

    # ---- Optional SNR pass-through ----
    calib_dir = os.path.join(project_dir, "data", "path_data", "calibrated_paths")
    calib_file = get_latest_calibrated_npz(calib_dir, cfg.scenario_name)
    if calib_file is not None:
        print(f"Loading SNR from: {os.path.basename(calib_file)}")
        calib_data = np.load(calib_file, allow_pickle=True)
        snr_raw_db      = calib_data["snr_raw_db"]       # (R, T)
        snr_bf_db       = calib_data["snr_bf_db"]         # (R, T)
        snr_detector_db = calib_data["snr_detector_db"]   # (R, T)
        p_rx_total_dbm  = calib_data["p_rx_total_dbm"]    # (R, T)
        has_snr = True
    else:
        print("[Info] No calibrated .npz found; SNR fields will be absent.")
        snr_raw_db = snr_bf_db = snr_detector_db = p_rx_total_dbm = None
        has_snr = False

    # ---- Training sequence ----
    seq_type = str(cfg.training_sequence.type).strip().lower()
    Nseq = int(cfg.training_sequence.length)
    Lh   = int(cfg.training_sequence.max_delay_bins)
    Lcp = Lh - 1

    if seq_type == "zadoff_chu":
        root = int(cfg.training_sequence.root_index)
        s = gen_zadoff_chu(Nseq, root)
    else:
        raise ValueError(f"Unknown training sequence type: '{seq_type}'")

    s_cp   = np.concatenate([s[-Lcp:], s])
    S_conj = np.conj(np.fft.fft(s, n=Nseq))

    # ---- Labels ----
    label_dim = int(cfg.training_sequence.get("label_dim", 2))
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
            print(f"  {rx_names[r]:12s}: {n_zeropad} positions will be zero-padded")

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
    # Zero-padding for "this BS does not see this position" is free: Z_all
    # is already zero-initialized, so we just skip those (r, t) pairs.
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
                continue  # zero-fill already done at Z_all allocation

            H = H_all[r, t].astype(np.complex128)
            Z = bdp_from_H(H, s_cp, S_conj, Lh, Lcp, Nseq)
            if M == 1:
                Z = Z[0]  # (N, Lh)

            Z_c64 = Z.astype(np.complex64)
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
    stamp = time.strftime("%Y%m%d_%H%M%S")
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
        "source_calibrated_npz": os.path.basename(calib_file) if calib_file else None,
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
        "Z_dtype":          "complex64",
        "Z_axes":           z_axes_str,
        "filtering_note": (
            "Only 'at least one BS has surviving paths' filter applied. "
            "Positions not visible from a BS are zero-padded in that BS's "
            "channel. Per-sample per-BS metadata is stored so downstream "
            "training code can filter or weight without regenerating."
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

    np.savez(ds_file, **save_dict)
    print(f"\nSaved: {ds_file}")

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