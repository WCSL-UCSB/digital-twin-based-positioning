# =============================================================================
# bdp_gen.py — Beam-Delay Profile generation for CNN training
# =============================================================================
#
# Consumes the consolidated channel-taps .npz from channel_tap_gen.py and
# produces complex beam-delay profiles (BDPs) per link plus a single
# consolidated training dataset.
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
#
# Note on pulse shape stacking: channel_tap_gen.py samples paths through a
# sinc pulse; this script applies a ZC pilot on top. See tracker T14 for
# the sanity check that this stacking is negligible for current parameters.
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
    DISPLAY ONLY. Not saved to disk. Training-time code should implement
    its own preprocessing on the complex data.
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
# Sanity plot (converts to display log on the fly)
# ---------------------------------------------------------------------------

def save_sanity_plot(Z_all, indices, titles, out_path, M, N,
                     display_range_db=50.0):
    """Grid plot of up to 9 BDPs with labels. Log conversion is display-only."""
    n = len(indices)
    if n == 0:
        return
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = np.atleast_2d(axes).ravel()

    for ax, idx, title in zip(axes, indices, titles):
        Z = Z_all[idx]
        img = _complex_to_display_log(Z, display_range_db)
        if img.ndim == 3:  # UPA: take peak elevation beam
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

    # ---- Optional SNR pass-through ----
    calib_dir = os.path.join(project_dir, "data", "path_data", "calibrated_paths")
    calib_file = get_latest_calibrated_npz(calib_dir, cfg.scenario_name)
    if calib_file is not None:
        print(f"Loading SNR from: {os.path.basename(calib_file)}")
        calib_data = np.load(calib_file, allow_pickle=True)
        snr_raw_db      = calib_data["snr_raw_db"]
        snr_bf_db       = calib_data["snr_bf_db"]
        snr_detector_db = calib_data["snr_detector_db"]
        p_rx_total_dbm  = calib_data["p_rx_total_dbm"]
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

    # ---- Process mask: only "link has any paths" ----
    process_mask = num_paths > 0
    n_process = int(process_mask.sum())

    # ---- Summary ----
    array_type = "UPA" if M > 1 else "ULA"
    print(f"\n--- BDP Generation ({array_type}) ---")
    print(f"Array        : {M} x {N} = {M*N} elements")
    print(f"Pilot        : {seq_type} length={Nseq} root={root}")
    print(f"Delay bins   : Lh={Lh}, CP={Lcp}")
    print(f"Total links  : {num_rx*num_tx}")
    print(f"Processing   : {n_process} links with surviving paths")
    print(f"  LoS-kept   : {int((is_los_kept & process_mask).sum())}")
    print(f"  NLoS-only  : {int((~is_los_kept & process_mask).sum())}")
    print(f"Label dim    : {label_dim}D")
    print(f"Output dtype : complex64 (log conversion deferred to training)")

    # ---- Allocate output (complex, canonical) ----
    Z_shape = (n_process, M, N, Lh) if M > 1 else (n_process, N, Lh)
    Z_all = np.zeros(Z_shape, dtype=np.complex64)
    Y_all = np.zeros((n_process, label_dim), dtype=np.float32)

    link_rx_idx  = np.zeros(n_process, dtype=np.int32)
    link_tx_idx  = np.zeros(n_process, dtype=np.int32)
    is_los_out   = np.zeros(n_process, dtype=bool)
    shift_ns_out = np.zeros(n_process, dtype=np.float32)
    snr_raw_out  = np.zeros(n_process, dtype=np.float32) if has_snr else None
    snr_bf_out   = np.zeros(n_process, dtype=np.float32) if has_snr else None
    snr_det_out  = np.zeros(n_process, dtype=np.float32) if has_snr else None
    p_rx_out     = np.zeros(n_process, dtype=np.float32) if has_snr else None

    # ---- Process ----
    t0 = time.time()
    out_idx = 0
    for r in range(num_rx):
        for t in range(num_tx):
            if not process_mask[r, t]:
                continue

            H = H_all[r, t].astype(np.complex128)
            Z = bdp_from_H(H, s_cp, S_conj, Lh, Lcp, Nseq)
            if M == 1:
                Z = Z[0]  # (N, Lh)

            Z_all[out_idx] = Z.astype(np.complex64)
            Y_all[out_idx] = tx_pos[t, :label_dim]
            link_rx_idx[out_idx]  = r
            link_tx_idx[out_idx]  = t
            is_los_out[out_idx]   = bool(is_los_kept[r, t])
            shift_ns_out[out_idx] = float(shift_ns[r, t])

            if has_snr:
                snr_raw_out[out_idx] = float(snr_raw_db[r, t])
                snr_bf_out[out_idx]  = float(snr_bf_db[r, t])
                snr_det_out[out_idx] = float(snr_detector_db[r, t])
                p_rx_out[out_idx]    = float(p_rx_total_dbm[r, t])

            out_idx += 1
            if out_idx % 500 == 0:
                rate = out_idx / max(time.time() - t0, 1e-6)
                print(f"  {out_idx}/{n_process}  ({rate:.1f} links/s)")

    elapsed = time.time() - t0
    print(f"\nProcessed {out_idx} links in {elapsed:.1f}s "
          f"({out_idx/max(elapsed,1e-6):.1f} links/s)")

    print(f"\nDataset shapes:")
    print(f"  Z: {Z_all.shape}  dtype={Z_all.dtype}")
    print(f"  Y: {Y_all.shape}  dtype={Y_all.dtype}")

    # ---- Save ----
    out_dir = os.path.join(project_dir, "data", "ml_datasets", cfg.scenario_name)
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ds_file = os.path.join(out_dir, f"{stamp}_bdp_dataset_{cfg.scenario_name}.npz")

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
        "label_dim":     label_dim,
        "array":         {"M": M, "N": N},
        "num_samples":   int(out_idx),
        "num_los_kept":  int(is_los_out.sum()),
        "num_nlos_only": int((~is_los_out).sum()),
        "Z_dtype":       "complex64",
        "Z_axes": (
            "(num_samples, M, N_beams, Lh)" if M > 1
            else "(num_samples, N_beams, Lh)"
        ),
        "filtering_note": (
            "Only 'link has surviving paths' filter applied. No SNR, "
            "delay, or LoS-based gating. Per-sample metadata is stored "
            "so downstream training code can filter or weight without "
            "regenerating the dataset."
        ),
        "preprocessing_note": (
            "Z is the CANONICAL complex beam-delay profile. Log-magnitude "
            "conversion, dynamic-range clipping, and normalization are "
            "training-time decisions and must be applied in the data "
            "loader — they are NOT baked into this dataset."
        ),
        "pilot_stacking_note": (
            "channel_tap_gen.py samples paths through a sinc pulse; this "
            "script applies a ZC pilot on top. See tracker T14."
        ),
    }

    save_dict = {
        "Z":            Z_all,          # canonical: complex BDP
        "Y":            Y_all,
        "link_rx_idx":  link_rx_idx,
        "link_tx_idx":  link_tx_idx,
        "is_los_kept":  is_los_out,
        "shift_ns":     shift_ns_out,
        "tx_names":     tx_names,
        "rx_names":     rx_names,
        "tx_pos":       tx_pos,
        "rx_pos":       rx_pos,
        "metadata":     dataset_meta,
    }
    if has_snr:
        save_dict["snr_raw_db"]      = snr_raw_out
        save_dict["snr_bf_db"]       = snr_bf_out
        save_dict["snr_detector_db"] = snr_det_out
        save_dict["p_rx_total_dbm"]  = p_rx_out

    np.savez(ds_file, **save_dict)
    print(f"\nSaved: {ds_file}")

    # ---- Sanity plot ----
    if out_idx > 0:
        picks, titles = [], []

        los_idx  = np.where(is_los_out)[0]
        nlos_idx = np.where(~is_los_out)[0]

        def pick_spread(arr, n):
            if len(arr) == 0:
                return []
            if len(arr) <= n:
                return list(arr)
            step = max(1, len(arr) // n)
            return list(arr[::step][:n])

        for i in pick_spread(los_idx, 3):
            label = f"LoS-kept  rx{link_rx_idx[i]}/tx{link_tx_idx[i]}"
            if has_snr:
                label += f"  SNR_det={snr_det_out[i]:+.0f}dB"
            picks.append(i)
            titles.append(label)

        if len(nlos_idx) > 0:
            nlos_sorted = nlos_idx[np.argsort(shift_ns_out[nlos_idx])]
            for i in pick_spread(nlos_sorted, 6):
                label = f"NLoS  Δτ={shift_ns_out[i]:.0f}ns"
                if has_snr:
                    label += f"  SNR_det={snr_det_out[i]:+.0f}dB"
                picks.append(i)
                titles.append(label)

        if picks:
            fig_path = os.path.join(
                out_dir, f"{stamp}_bdp_sanity_{cfg.scenario_name}.png"
            )
            save_sanity_plot(Z_all, picks, titles, fig_path, M, N,
                             display_range_db=50.0)
            print(f"Sanity plot: {fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()