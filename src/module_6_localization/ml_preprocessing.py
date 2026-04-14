# =============================================================================
# ml_preprocess.py — Convert complex BDPs to CNN-ready magnitude tensors
# =============================================================================
#
# Reads the canonical complex BDP dataset from bdp_gen.py and produces a
# CNN-ready preprocessed dataset with:
#   - log-magnitude image X (float32, ready for the training script)
#   - labels Y unchanged
#   - per-sample metadata (is_los_kept, shift_ns, SNR fields) carried
#     through for downstream evaluation analysis
#
# Optional filters (both default to off):
#   - min_snr_detector_db : drop samples with detector SNR below this
#   - los_kept_only       : drop NLoS-only samples
#
# This is non-destructive: the upstream complex BDP file is untouched, and
# rerunning with different parameters is cheap (no need to regenerate any
# upstream stage).
# =============================================================================

import os
import glob
import time
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Magnitude transform
# ---------------------------------------------------------------------------

def complex_to_cnn_input(Z, magnitude_mode, normalization, dynamic_range_db):
    """
    Convert a batch of complex BDPs to float32 CNN-input tensors.

    Parameters
    ----------
    Z : complex array of shape (num_samples, ..., Lh)
        For ULA: (num_samples, N_beams, Lh).
        For UPA: (num_samples, M_beams, N_beams, Lh).
    magnitude_mode  : "log" or "linear"
    normalization   : "per_sample" or "global"
    dynamic_range_db : float, used only when magnitude_mode == "log"

    Returns
    -------
    X : float32 array, same leading shape as Z, with a trailing channel
        axis appended, values in [0, 1].
    """
    mag = np.abs(Z).astype(np.float32)

    # Compute peak according to normalization mode
    if normalization == "per_sample":
        # Reduce over all axes except the leading sample axis
        reduce_axes = tuple(range(1, mag.ndim))
        peak = mag.max(axis=reduce_axes, keepdims=True) + 1e-12
    elif normalization == "global":
        peak = np.full(
            (1,) * mag.ndim, mag.max() + 1e-12, dtype=np.float32
        )
    else:
        raise ValueError(
            f"normalization must be 'per_sample' or 'global', got '{normalization}'"
        )

    mag_norm = mag / peak

    if magnitude_mode == "log":
        X = 20.0 * np.log10(mag_norm + 1e-12)
        X = np.maximum(X, -dynamic_range_db)
        X = (X + dynamic_range_db) / dynamic_range_db
    elif magnitude_mode == "linear":
        X = mag_norm
    else:
        raise ValueError(
            f"magnitude_mode must be 'log' or 'linear', got '{magnitude_mode}'"
        )

    # Append trailing channel axis for CNN
    return X[..., np.newaxis].astype(np.float32)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def get_latest_bdp_dataset(path_dir, scenario_name):
    pattern = os.path.join(path_dir, f"*_bdp_dataset_{scenario_name}.npz")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No BDP dataset matching: {pattern}")
    return max(files, key=os.path.getctime)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../configs/scenarios")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    project_dir = cfg.project_dir

    # ---- Load BDP dataset ----
    bdp_dir = os.path.join(project_dir, "data", "ml_datasets", cfg.scenario_name)
    bdp_file = get_latest_bdp_dataset(bdp_dir, cfg.scenario_name)
    print(f"Loading BDP dataset: {os.path.basename(bdp_file)}")
    data = np.load(bdp_file, allow_pickle=True)
    bdp_meta = data["metadata"].item()

    Z = data["Z"]
    Y = data["Y"].astype(np.float32)

    print(f"  Raw Z shape : {Z.shape}  dtype={Z.dtype}")
    print(f"  Y shape     : {Y.shape}")

    # ---- Read preprocessing config ----
    pp = cfg.get("ml_preprocessing", {})
    min_snr = pp.get("min_snr_detector_db", None)
    los_kept_only = bool(pp.get("los_kept_only", False))
    magnitude_mode = str(pp.get("magnitude_mode", "log")).lower()
    normalization = str(pp.get("normalization", "per_sample")).lower()
    dynamic_range_db = float(pp.get("dynamic_range_db", 50.0))

    # ---- Pull per-sample metadata if available ----
    meta_fields = {}
    for key in ("is_los_kept", "shift_ns",
                "snr_raw_db", "snr_bf_db", "snr_detector_db",
                "p_rx_total_dbm",
                "link_rx_idx", "link_tx_idx"):
        if key in data.files:
            meta_fields[key] = data[key]

    # ---- Build filter mask ----
    n_total = len(Y)
    keep_mask = np.ones(n_total, dtype=bool)

    if min_snr is not None:
        if "snr_detector_db" not in meta_fields:
            raise KeyError(
                "min_snr_detector_db is set but the BDP dataset does not "
                "contain snr_detector_db. Make sure apply_link_budget.py "
                "ran before bdp_gen.py."
            )
        snr_mask = meta_fields["snr_detector_db"] >= float(min_snr)
        keep_mask &= snr_mask

    if los_kept_only:
        if "is_los_kept" not in meta_fields:
            raise KeyError(
                "los_kept_only is true but the BDP dataset does not "
                "contain is_los_kept."
            )
        keep_mask &= meta_fields["is_los_kept"]

    n_kept = int(keep_mask.sum())

    # ---- Filter summary ----
    print(f"\n--- Filtering ---")
    print(f"  Total samples         : {n_total}")
    if min_snr is not None:
        n_pass_snr = int((meta_fields['snr_detector_db'] >= float(min_snr)).sum())
        print(f"  min_snr_detector_db   : {min_snr:+.1f} dB  "
              f"-> {n_pass_snr} pass")
    else:
        print(f"  min_snr_detector_db   : (off)")
    if los_kept_only:
        n_los = int(meta_fields["is_los_kept"].sum())
        print(f"  los_kept_only         : on   -> {n_los} pass")
    else:
        print(f"  los_kept_only         : off")
    print(f"  Combined kept         : {n_kept} "
          f"({100.0*n_kept/max(n_total,1):.1f}%)")

    if n_kept == 0:
        print("\n[ERROR] Filter rejected all samples. Loosen the criteria.")
        return

    # ---- Apply filter to Z, Y, and metadata ----
    Z_kept = Z[keep_mask]
    Y_kept = Y[keep_mask]
    meta_kept = {k: v[keep_mask] for k, v in meta_fields.items()}

    # ---- Magnitude transform ----
    print(f"\n--- Transform ---")
    print(f"  magnitude_mode  : {magnitude_mode}")
    print(f"  normalization   : {normalization}")
    if magnitude_mode == "log":
        print(f"  dynamic_range   : {dynamic_range_db:.1f} dB")

    X = complex_to_cnn_input(
        Z_kept,
        magnitude_mode=magnitude_mode,
        normalization=normalization,
        dynamic_range_db=dynamic_range_db,
    )
    print(f"  X shape         : {X.shape}  dtype={X.dtype}")

    # ---- Save ----
    out_dir = os.path.join(
        project_dir, "data", "preprocessed_datasets", cfg.scenario_name
    )
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(
        out_dir, f"{stamp}_preprocessed_{cfg.scenario_name}.npz"
    )

    pp_metadata = {
        "source_bdp_npz":      os.path.basename(bdp_file),
        "source_bdp_meta":     bdp_meta,
        "filters": {
            "min_snr_detector_db": min_snr,
            "los_kept_only":       los_kept_only,
        },
        "transform": {
            "magnitude_mode":   magnitude_mode,
            "normalization":    normalization,
            "dynamic_range_db": dynamic_range_db,
        },
        "n_total":  int(n_total),
        "n_kept":   int(n_kept),
        "n_los_kept": (int(meta_kept["is_los_kept"].sum())
                       if "is_los_kept" in meta_kept else None),
    }

    save_dict = {
        "X":        X,
        "Y":        Y_kept,
        "metadata": pp_metadata,
    }
    # Carry through per-sample metadata fields
    for k, v in meta_kept.items():
        save_dict[k] = v
    # Also carry through scenario-level fields so downstream can plot maps
    for k in ("tx_pos", "rx_pos", "tx_names", "rx_names"):
        if k in data.files:
            save_dict[k] = data[k]

    np.savez(out_file, **save_dict)
    print(f"\nSaved: {out_file}")
    print(f"  Samples         : {n_kept}")
    print(f"  Input shape     : {X.shape}")
    print(f"  Label shape     : {Y_kept.shape}")
    print("\nDone.")


if __name__ == "__main__":
    main()