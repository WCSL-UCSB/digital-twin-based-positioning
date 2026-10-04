# =============================================================================
# ml_preprocessing.py — Convert complex BDPs to CNN-ready magnitude tensors
# =============================================================================
#
# Reads the canonical complex BDP dataset from beam_delay_gen.py and produces a
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
#   with either one shared peak per position (per_sample) or a separate
#   peak per BS observation at each position (per_bs).
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
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.dataset_io import resolve_input, pinned, write_sidecar  # noqa: E402
from src.utils.radio_power import positive_int
import uuid
from datetime import datetime
import numpy as np
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Magnitude transform
# ---------------------------------------------------------------------------

def validate_transform_settings(magnitude_mode, normalization, dynamic_range_db):
    """Reject unsupported transforms before loading or allocating dataset arrays."""
    magnitude_mode = str(magnitude_mode).strip().lower()
    normalization = str(normalization).strip().lower()
    if magnitude_mode not in ("log", "linear"):
        raise ValueError(
            "ml_preprocessing.magnitude_mode must be 'log' or 'linear', "
            f"got {magnitude_mode!r}")
    if normalization not in ("per_sample", "per_bs"):
        raise ValueError(
            "ml_preprocessing.normalization must be 'per_sample' or 'per_bs', "
            f"got {normalization!r}")
    if magnitude_mode == "log":
        message = "ml_preprocessing.dynamic_range_db must be finite and greater than zero for log processing"
        try:
            dynamic_range_db = float(dynamic_range_db)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc
        if not np.isfinite(dynamic_range_db) or dynamic_range_db <= 0:
            raise ValueError(message)
    return magnitude_mode, normalization, dynamic_range_db


def complex_to_cnn_input_batched(Z, magnitude_mode, normalization,
                                  dynamic_range_db, batch_size=512,
                                  channels_present=None,
                                  return_peaks=False):
    """
    Convert complex BDP tensor to real-valued CNN input.

    For single-BS: Z shape (N, ...) → X shape (N, ..., 1)
    For multi-BS:  Z shape (N, ..., K) with K > 1 → X shape (N, ..., K)
                   K is already the channel axis, no extra dim added.

    channels_present must explicitly identify whether the last axis is a BS
    axis. Shape alone is ambiguous between single-BS UPA and multi-BS ULA.
    A single-BS subset Z=(N,...,1) already has that axis and passes True.
    """
    magnitude_mode, normalization, dynamic_range_db = validate_transform_settings(
        magnitude_mode, normalization, dynamic_range_db)
    batch_size = positive_int(batch_size, "batch_size")
    if Z.ndim < 2 or not all(Z.shape):
        raise ValueError("Z must have nonempty sample and feature axes")
    n = Z.shape[0]

    if channels_present is None:
        raise ValueError("channels_present must be specified from the BDP layout metadata")
    multi_bs = bool(channels_present)

    if multi_bs:
        out_shape = Z.shape  # K is the channel axis
    else:
        out_shape = Z.shape + (1,)  # add channel dim for single-BS

    X = np.empty(out_shape, dtype=np.float32)
    K = Z.shape[-1] if multi_bs else 1
    peak_shape = (n, K) if normalization == "per_bs" else (n,)
    normalization_peaks = np.empty(peak_shape, dtype=np.float64)

    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        if not np.isfinite(Z[i:j]).all():
            raise ValueError("BDP values must be finite before magnitude processing")
        # Double precision also avoids magnitude overflow for finite complex64
        # components near their limit. X remains float32 for model input.
        mag = np.abs(Z[i:j].astype(np.complex128))
        if not np.isfinite(mag).all():
            raise ValueError("BDP magnitudes must be finite")

        # per_bs excludes the explicit BS axis, while both modes exclude the
        # sample axis. Native single-BS input has no BS axis yet.
        last_reduce_axis = mag.ndim - 1 if normalization == "per_bs" and multi_bs else mag.ndim
        reduce_axes = tuple(range(1, last_reduce_axis))
        peak = mag.max(axis=reduce_axes, keepdims=True)
        if normalization == "per_bs":
            normalization_peaks[i:j] = peak.reshape(j-i, K)
        else:
            normalization_peaks[i:j] = peak.reshape(-1)

        # All-zero samples stay zero; no absolute epsilon alters weak signals.
        mag_norm = np.divide(mag, peak, out=np.zeros_like(mag), where=np.asarray(peak) > 0)

        if magnitude_mode == "log":
            with np.errstate(divide="ignore"):
                chunk_out = 20.0 * np.log10(mag_norm)
            np.maximum(chunk_out, -dynamic_range_db, out=chunk_out)
            chunk_out = (chunk_out + dynamic_range_db) / dynamic_range_db
        else:
            chunk_out = mag_norm

        if multi_bs:
            X[i:j] = chunk_out
        else:
            X[i:j] = chunk_out[..., np.newaxis]
        if not np.isfinite(X[i:j]).all():
            raise ValueError("Normalized features must be finite; check input values and transform settings")

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {j}/{n} samples")

    return (X, normalization_peaks) if return_peaks else X


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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../configs/scenarios")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    project_dir = cfg.project_dir

    # ---- Read preprocessing config ----
    pp = cfg.get("ml_preprocessing", {})
    magnitude_mode, normalization, dynamic_range_db = validate_transform_settings(
        pp.get("magnitude_mode", "log"),
        pp.get("normalization", "per_sample"),
        pp.get("dynamic_range_db", 50.0))

    # ---- Load BDP dataset ----
    # `ml_preprocessing.source_scenario` lets a derived scenario (e.g. a BS
    # subset) read an EXISTING BDP while writing under its own scenario_name,
    # so we don't have to re-run the whole ray-trace pipeline to drop a BS.
    src_scn = str(pp.get("source_scenario", cfg.scenario_name))
    bdp_dir = os.path.join(project_dir, "data", "ml_datasets", src_scn)
    bdp_file = resolve_input(
        os.path.join(bdp_dir, f"*_bdp_dataset_{src_scn}.npz"),
        pinned(cfg, "bdp_npz"), label="BDP dataset")
    data = np.load(bdp_file, allow_pickle=True)
    bdp_meta = data["metadata"].item()

    Z = data["Z"]
    Y = data["Y"].astype(np.float32)
    if Y.ndim != 2 or Y.shape[1] not in (2, 3) or len(Y) != len(Z) or not np.isfinite(Y).all():
        raise ValueError("Y must be finite (samples, 2 or 3) coordinates matching Z")

    # A single-BS profile has no BS axis: its last axis is delay, not K.
    # Legacy datasets may identify K through their receiver table instead.
    recorded_k = bdp_meta.get("num_base_stations")
    receiver_k = len(data["rx_names"]) if "rx_names" in data.files else None
    if recorded_k is None:
        recorded_k = receiver_k
    if recorded_k is None:
        raise ValueError("BDP needs num_base_stations metadata or rx_names to identify its BS axis")
    K = positive_int(recorded_k, "BDP base-station count")
    if receiver_k is not None and receiver_k != K:
        raise ValueError("BDP base-station count disagrees with rx_names")
    multi_bs = K > 1
    channels_present = multi_bs
    expected_ranks = (4, 5) if multi_bs else (3, 4)
    array_rows = bdp_meta.get("array", {}).get("M")
    if array_rows is not None:
        expected_ranks = (3 + int(positive_int(array_rows, "BDP array rows") > 1)
                          + int(multi_bs),)
    if Z.ndim not in expected_ranks or (multi_bs and Z.shape[-1] != K):
        raise ValueError(f"BDP shape {Z.shape} disagrees with its array/BS layout metadata")

    print(f"  Raw Z shape : {Z.shape}  dtype={Z.dtype}")
    print(f"  Y shape     : {Y.shape}")
    if multi_bs:
        print(f"  Multi-BS    : K={K} (last axis is channel, no extra dim)")
    else:
        print(f"  Single-BS   : trailing channel dim will be appended")

    # ---- Read transform config ----
    min_snr = pp.get("min_snr_detector_db", None)
    if min_snr is not None:
        min_snr = float(min_snr)
        if not np.isfinite(min_snr):
            raise ValueError("ml_preprocessing.min_snr_detector_db must be finite or null")
    los_kept_only = bool(pp.get("los_kept_only", False))
    # Drop samples with zero coverage over the (possibly subset) BS set: a point
    # that reaches NO kept base station has zero-filled inputs in new BDPs.
    # Older BDPs may contain noise there. This is a coverage policy, not a detector.
    drop_uncovered = bool(pp.get("drop_uncovered", False))

    # ---- Pull per-sample metadata if available ----
    meta_fields = {}
    for key in ("is_los_kept", "has_paths", "shift_ns",
                "snr_raw_db", "snr_bf_db", "snr_detector_db",
                "p_rx_total_dbm", "pilot_signal_power_w", "realized_noise_power_w",
                "link_rx_idx", "link_tx_idx", "tx_idx"):
        if key in data.files:
            meta_fields[key] = data[key]
            value = meta_fields[key]
            expected = ((len(Y),),) if key in ("link_rx_idx", "link_tx_idx", "tx_idx") else (
                ((len(Y),), (len(Y), 1)) if K == 1 else ((len(Y), K),))
            if value.shape not in expected:
                raise ValueError(f"{key} shape {value.shape} does not match samples and BS count")
            if key in ("is_los_kept", "has_paths") and value.dtype != np.bool_:
                raise ValueError(f"{key} must be boolean")

    # Pull scenario-level fields now so we can close `data` before savez
    # and free its memory.
    carry_through = {}
    for k in ("tx_pos", "rx_pos", "tx_names", "rx_names"):
        if k in data.files:
            carry_through[k] = data[k]

    # ---- Optional BS subset (drop/keep specific base stations) ----
    # `ml_preprocessing.keep_bs: [i, j, ...]` selects a subset of the K base
    # stations BEFORE normalization, so the per-sample peak is recomputed over
    # only the kept BS. This yields a faithful K'<K deployment, not a post-hoc
    # channel drop (which would leave the peak set by an absent BS).
    keep_bs = pp.get("keep_bs", None)
    if keep_bs is not None:
        # Validate before indexing, including native single-BS datasets. Never
        # coerce fractional indices or duplicate a receiver into extra channels.
        if not isinstance(keep_bs, (list, tuple, ListConfig)) or not keep_bs:
            raise ValueError("ml_preprocessing.keep_bs must be a nonempty list of BS indices")
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
               for i in keep_bs):
            raise ValueError("ml_preprocessing.keep_bs indices must be integers")
        if any(i < 0 or i >= K for i in keep_bs):
            raise ValueError(f"ml_preprocessing.keep_bs indices must be in [0, {K})")
        if len(set(keep_bs)) != len(keep_bs):
            raise ValueError("ml_preprocessing.keep_bs indices must be unique")
        keep_bs = list(keep_bs)
    if keep_bs is not None and multi_bs:
        names = carry_through.get("rx_names")
        lbl = ", ".join(str(names[i]) for i in keep_bs) if names is not None else str(keep_bs)
        print(f"\n--- BS subset ---\n  keep_bs : {keep_bs}  ({lbl})  of K={K}")
        Z = Z[..., keep_bs]
        for key, v in list(meta_fields.items()):
            if getattr(v, "ndim", 1) >= 2 and v.shape[-1] == K:
                meta_fields[key] = v[..., keep_bs]
        for key in ("rx_pos", "rx_names"):
            if key in carry_through and len(carry_through[key]) == K:
                carry_through[key] = carry_through[key][keep_bs]
        K = len(keep_bs)
        multi_bs = K > 1
        print(f"  -> new K: {K}, Z {Z.shape}")

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

    if drop_uncovered:
        if "has_paths" not in meta_fields:
            raise KeyError(
                "drop_uncovered is true but the BDP dataset does not contain "
                "has_paths. Make sure the upstream stages carried it through."
            )
        # keep if ANY kept BS reaches the point (has_paths already sliced to the
        # kept subset above, so this is coverage over the subset deployment)
        cov_scalar = _collapse_for_filter(meta_fields["has_paths"], rule="any")
        keep_mask &= cov_scalar
        print(f"  drop_uncovered        : on   -> {int(cov_scalar.sum())} covered "
              f"({100.0*cov_scalar.mean():.1f}%)")

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
        data.close()
        raise ValueError(
            "No samples remain after preprocessing filters. No dataset was generated. "
            "Check the input dataset, BS selection, and preprocessing filters.")

    # ---- Apply filter to Z, Y, and metadata ----
    # When no filter is active (default), boolean indexing would force a
    # full copy of Z (~10 GB at testboston dims). Alias instead.
    if keep_mask.all():
        Z_kept = Z
        Y_kept = Y
        meta_kept = meta_fields
    else:
        Z_kept = Z[keep_mask]
        Y_kept = Y[keep_mask]
        meta_kept = {k: v[keep_mask] for k, v in meta_fields.items()}

    # ---- Magnitude transform ----
    print(f"\n--- Transform ---")
    print(f"  magnitude_mode  : {magnitude_mode}")
    print(f"  normalization   : {normalization}")
    if magnitude_mode == "log":
        print(f"  dynamic_range   : {dynamic_range_db:.1f} dB")

    X, normalization_peaks = complex_to_cnn_input_batched(
        Z_kept,
        magnitude_mode=magnitude_mode,
        normalization=normalization,
        dynamic_range_db=dynamic_range_db,
        batch_size=512,
        # After a keep_bs slice the last axis IS the BS-channel axis (even when
        # the subset is a single BS), so don't append a spurious channel dim.
        channels_present=channels_present,
        return_peaks=True,
    )
    print(f"  X shape         : {X.shape}  dtype={X.dtype}")

    # ---- Save ----
    out_dir = os.path.join(
        project_dir, "data", "preprocessed_datasets", cfg.scenario_name
    )
    os.makedirs(out_dir, exist_ok=True)
    # Use the same collision protection as BDP generation. The timestamp
    # prefix remains compatible with existing latest-dataset discovery.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8]
    out_file = os.path.join(
        out_dir, f"{stamp}_preprocessed_{cfg.scenario_name}.npz"
    )

    pp_metadata = {
        "source_bdp_npz":      os.path.basename(bdp_file),
        "source_bdp_meta":     bdp_meta,
        "filters": {
            "min_snr_detector_db": min_snr,
            "los_kept_only":       los_kept_only,
            "drop_uncovered":      drop_uncovered,
            "keep_bs":             keep_bs,
        },
        "transform": {
            "magnitude_mode":   magnitude_mode,
            "normalization":    normalization,
            "dynamic_range_db": dynamic_range_db,
        },
        "n_total":    int(n_total),
        "normalization_reference": {
            "scope": "sample_across_selected_bs" if normalization == "per_sample" else "sample_per_bs",
            "peak_axes": "(samples,)" if normalization == "per_sample" else "(samples, selected_bs)",
            "stored_field": "normalization_peak",
            "units": "magnitude of stored, unnormalized complex BDP Z",
        },
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
        "normalization_peak": normalization_peaks,
        "metadata": pp_metadata,
    }
    # Carry through per-sample metadata fields (keep original shapes)
    for k, v in meta_kept.items():
        save_dict[k] = v
    # Also carry through scenario-level fields so downstream can plot maps
    save_dict.update(carry_through)

    # Release the large complex BDP tensor(s) before savez. At testboston
    # dims Z is ~10 GB complex64; X (float32) is ~5 GB, and only X is
    # actually serialized as a heavy array from here on.
    import gc
    del Z_kept
    del Z
    data.close()
    gc.collect()

    # Exclusive creation prevents even a forced name collision from
    # overwriting an existing dataset or its associated provenance sidecar.
    with open(out_file, "xb") as output:
        np.savez(output, **save_dict)
    print(f"\nSaved: {out_file}")
    write_sidecar(out_file, stage="ml_preprocessing", cfg=cfg,
                  inputs={"bdp_npz": bdp_file}, summary=pp_metadata)
    print(f"  Samples         : {n_kept}")
    print(f"  Input shape     : {X.shape}")
    print(f"  Label shape     : {Y_kept.shape}")
    print("\nDone.")


if __name__ == "__main__":
    main()
