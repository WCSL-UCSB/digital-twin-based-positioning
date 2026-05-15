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
# Multi-BS support:
#   When Z has shape (N, N_beams, Lh, K) with K > 1, the trailing K axis
#   IS the channel axis — no extra dim is appended. Output X has the same
#   shape as Z (but float32 magnitude). The magnitude transform operates
#   independently on each BS channel.
#
#   When Z has shape (N, N_beams, Lh) with K = 1 (single-BS), a trailing
#   channel dim of 1 is appended: X shape = (N, N_beams, Lh, 1).
#
# Optional filters (both default to off):
#   - min_snr_detector_db : drop samples with detector SNR below this
#   - los_kept_only       : drop NLoS-only samples
#
# For multi-BS metadata (shape (N, K)), filters use:
#   - is_los_kept: any BS LoS → keep
#   - snr_detector_db: max across BSs → compare to threshold
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

def complex_to_cnn_input_batched(Z, magnitude_mode, normalization,
                                  dynamic_range_db, batch_size=512):
    """
    Convert complex BDP tensor to real-valued CNN input.

    For single-BS: Z shape (N, ...) → X shape (N, ..., 1)
    For multi-BS:  Z shape (N, ..., K) with K > 1 → X shape (N, ..., K)
                   K is already the channel axis, no extra dim added.
    """
    n = Z.shape[0]

    # Detect multi-BS: last axis > 1 means K BS channels already present
    multi_bs = Z.ndim >= 3 and Z.shape[-1] > 1

    if multi_bs:
        out_shape = Z.shape  # K is the channel axis
    else:
        out_shape = Z.shape + (1,)  # add channel dim for single-BS

    X = np.empty(out_shape, dtype=np.float32)

    # For global normalization, first pass to find peak
    if normalization == "global":
        global_peak = np.float32(0.0)
        for i in range(0, n, batch_size):
            chunk = np.abs(Z[i:i+batch_size]).astype(np.float32)
            global_peak = max(global_peak, chunk.max())
        global_peak += 1e-12

    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        mag = np.abs(Z[i:j]).astype(np.float32)

        if normalization == "per_sample":
            reduce_axes = tuple(range(1, mag.ndim))
            peak = mag.max(axis=reduce_axes, keepdims=True) + 1e-12
        else:
            peak = global_peak

        mag_norm = mag / peak

        if magnitude_mode == "log":
            chunk_out = 20.0 * np.log10(mag_norm + 1e-12)
            np.maximum(chunk_out, -dynamic_range_db, out=chunk_out)
            chunk_out = (chunk_out + dynamic_range_db) / dynamic_range_db
        else:
            chunk_out = mag_norm

        if multi_bs:
            X[i:j] = chunk_out
        else:
            X[i:j] = chunk_out[..., np.newaxis]

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {j}/{n} samples")

    return X


# ---------------------------------------------------------------------------
# Multi-BS metadata helpers
# ---------------------------------------------------------------------------

def _collapse_for_filter(arr, rule="any"):
    """
    Collapse a per-BS metadata field from (N, K) to (N,) for filtering.
    If already 1D, return as-is.
    """
    if arr.ndim < 2:
        return arr
    if rule == "any":
        return np.any(arr, axis=1)
    elif rule == "max":
        return np.max(arr, axis=1)
    elif rule == "min":
        return np.min(arr, axis=1)
    return arr


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

    # Detect multi-BS
    multi_bs = Z.ndim >= 3 and Z.shape[-1] > 1
    K = Z.shape[-1] if multi_bs else 1

    print(f"  Raw Z shape : {Z.shape}  dtype={Z.dtype}")
    print(f"  Y shape     : {Y.shape}")
    if multi_bs:
        print(f"  Multi-BS    : K={K} (last axis is channel, no extra dim)")
    else:
        print(f"  Single-BS   : trailing channel dim will be appended")

    # ---- Read preprocessing config ----
    pp = cfg.get("ml_preprocessing", {})
    min_snr = pp.get("min_snr_detector_db", None)
    los_kept_only = bool(pp.get("los_kept_only", False))
    magnitude_mode = str(pp.get("magnitude_mode", "log")).lower()
    normalization = str(pp.get("normalization", "per_sample")).lower()
    dynamic_range_db = float(pp.get("dynamic_range_db", 50.0))

    # ---- Pull per-sample metadata if available ----
    meta_fields = {}
    for key in ("is_los_kept", "has_paths", "shift_ns",
                "snr_raw_db", "snr_bf_db", "snr_detector_db",
                "p_rx_total_dbm",
                "link_rx_idx", "link_tx_idx", "tx_idx"):
        if key in data.files:
            meta_fields[key] = data[key]

    # ---- Build filter mask ----
    # For multi-BS metadata (N, K), collapse to (N,) for scalar comparison
    n_total = len(Y)
    keep_mask = np.ones(n_total, dtype=bool)

    if min_snr is not None:
        if "snr_detector_db" not in meta_fields:
            raise KeyError(
                "min_snr_detector_db is set but the BDP dataset does not "
                "contain snr_detector_db. Make sure apply_link_budget.py "
                "ran before bdp_gen.py."
            )
        # Use max-across-BSs: keep if the best BS exceeds threshold
        snr_scalar = _collapse_for_filter(
            meta_fields["snr_detector_db"], rule="max"
        )
        snr_mask = snr_scalar >= float(min_snr)
        keep_mask &= snr_mask

    if los_kept_only:
        if "is_los_kept" not in meta_fields:
            raise KeyError(
                "los_kept_only is true but the BDP dataset does not "
                "contain is_los_kept."
            )
        # Use any-across-BSs: keep if LoS from any BS
        los_scalar = _collapse_for_filter(
            meta_fields["is_los_kept"], rule="any"
        )
        keep_mask &= los_scalar

    n_kept = int(keep_mask.sum())

    # ---- Filter summary ----
    print(f"\n--- Filtering ---")
    print(f"  Total samples         : {n_total}")
    if min_snr is not None:
        n_pass_snr = int(snr_mask.sum())
        print(f"  min_snr_detector_db   : {min_snr:+.1f} dB  "
              f"-> {n_pass_snr} pass")
    else:
        print(f"  min_snr_detector_db   : (off)")
    if los_kept_only:
        n_los = int(los_scalar.sum())
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

    X = complex_to_cnn_input_batched(
        Z_kept,
        magnitude_mode=magnitude_mode,
        normalization=normalization,
        dynamic_range_db=dynamic_range_db,
        batch_size=512,
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
        "n_total":    int(n_total),
        "n_kept":     int(n_kept),
        "multi_bs":   multi_bs,
        "K":          K,
        "n_los_kept": (int(_collapse_for_filter(
                           meta_kept["is_los_kept"], "any").sum())
                       if "is_los_kept" in meta_kept else None),
    }

    save_dict = {
        "X":        X,
        "Y":        Y_kept,
        "metadata": pp_metadata,
    }
    # Carry through per-sample metadata fields (keep original shapes)
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