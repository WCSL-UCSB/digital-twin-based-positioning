"""Evaluation metrics for localization runs.

NumPy-only at import time, for use by baseline and learned positioning models.

compute_all_metrics() returns a JSON-ready dict with:
    core        : mean / rmse / p50 / p67 / p90 / p95 / cdf_at_m / n
    breakdowns  : LoS vs NLoS, SNR bins, number of BSs with paths
    calibration : NLL + highest-density-region coverage for the probabilistic
                  heads (heteroscedastic, mdn, uncertainty fusion)
    provenance  : scenario / run_tag / split params / sha1 of the test indices
                  (use this hash to check test-set identity across runs)

All NLLs include the 0.5*log(2*pi) constant per dimension so they are
comparable ACROSS heads even though heteroscedastic_loss omits the constant
during training.
"""

import hashlib
import json
import os
from datetime import datetime

import numpy as np

CDF_THRESHOLDS_M = (1.0, 2.0, 5.0, 10.0)
SNR_BIN_EDGES_DB = (-np.inf, 0.0, 10.0, 20.0, 30.0, np.inf)
COVERAGE_LEVELS = (0.5, 0.683, 0.9, 0.95)

_LOG_2PI = float(np.log(2.0 * np.pi))


def _chi2_2dof_ppf(q):
    """chi^2 quantile with 2 dof has the closed form -2*ln(1-q)."""
    return -2.0 * np.log(1.0 - q)


def core_error_metrics(errors):
    """Scalar error statistics for a 1-D array of Euclidean errors [m]."""
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0:
        return {"n": 0}
    return {
        "n": int(errors.size),
        "mean_m": float(errors.mean()),
        "rmse_m": float(np.sqrt((errors ** 2).mean())),
        "p50_m": float(np.percentile(errors, 50)),
        "p67_m": float(np.percentile(errors, 67)),
        "p90_m": float(np.percentile(errors, 90)),
        "p95_m": float(np.percentile(errors, 95)),
        "cdf_at_m": {f"{t:g}": float((errors <= t).mean())
                     for t in CDF_THRESHOLDS_M},
    }


def _subset_stats(errors, mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return None
    sub = errors[mask]
    return {
        "n": int(mask.sum()),
        "mean_m": float(sub.mean()),
        "median_m": float(np.median(sub)),
        "p90_m": float(np.percentile(sub, 90)),
    }


# ---------------------------------------------------------------------------
# Calibration helpers (one per probabilistic head)
# ---------------------------------------------------------------------------

def _gaussian_diag_calibration(Y, mu, log_var):
    """NLL + ellipse coverage for a per-sample diagonal Gaussian."""
    var = np.exp(log_var)
    sq = (Y - mu) ** 2
    nll = 0.5 * (_LOG_2PI + log_var + sq / var).sum(axis=1)
    maha2 = (sq / var).sum(axis=1)
    coverage = {f"{q:g}": float((maha2 <= _chi2_2dof_ppf(q)).mean())
                for q in COVERAGE_LEVELS}
    return {
        "nll_mean": float(nll.mean()),
        "hdr_coverage": coverage,
        "mean_sigma_m": [float(np.sqrt(var[:, 0]).mean()),
                         float(np.sqrt(var[:, 1]).mean())],
    }


def _mdn_log_density(points, logits, means, log_scales):
    """log p(points) under per-sample diagonal-Gaussian mixtures.

    points (N, S, 2) evaluated against mixtures with logits (N, M),
    means (N, M, 2), log_scales (N, M, 2). Returns (N, S).
    """
    log_pi = logits - np.log(np.exp(logits - logits.max(axis=1, keepdims=True))
                             .sum(axis=1, keepdims=True)) \
        - logits.max(axis=1, keepdims=True)
    diff = points[:, :, None, :] - means[:, None, :, :]      # (N, S, M, 2)
    inv_var = np.exp(-log_scales)[:, None, :, :]
    lp_dim = -0.5 * (_LOG_2PI + log_scales[:, None, :, :]
                     + diff ** 2 * inv_var)
    lp = lp_dim.sum(axis=-1) + log_pi[:, None, :]            # (N, S, M)
    m = lp.max(axis=-1, keepdims=True)
    return (m[..., 0] + np.log(np.exp(lp - m).sum(axis=-1)))


def _mdn_calibration(Y, raw_pred, n_mixtures, mc_samples=200, seed=0):
    """NLL (exact) + Monte-Carlo highest-density-region coverage for MDN."""
    M = int(n_mixtures)
    arr = np.asarray(raw_pred, dtype=np.float64)
    logits = arr[:, :M]
    means = arr[:, M:3 * M].reshape(-1, M, 2)
    log_scales = arr[:, 3 * M:5 * M].reshape(-1, M, 2)
    N = len(arr)

    nll = -_mdn_log_density(Y[:, None, :], logits, means, log_scales)[:, 0]

    # HDR coverage: y is inside the q-mass highest-density region iff fewer
    # than a q fraction of mixture samples have higher density than y.
    rng = np.random.default_rng(seed)
    pi = np.exp(logits - logits.max(axis=1, keepdims=True))
    pi /= pi.sum(axis=1, keepdims=True)
    comp = np.array([rng.choice(M, size=mc_samples, p=pi[i])
                     for i in range(N)])                      # (N, S)
    rows = np.arange(N)[:, None]
    mu_s = means[rows, comp]                                  # (N, S, 2)
    sd_s = np.exp(0.5 * log_scales[rows, comp])
    samples = mu_s + rng.standard_normal(mu_s.shape) * sd_s

    log_d_samples = _mdn_log_density(samples, logits, means, log_scales)
    rank = (log_d_samples > -nll[:, None]).mean(axis=1)       # (N,)
    coverage = {f"{q:g}": float((rank < q).mean()) for q in COVERAGE_LEVELS}

    return {"nll_mean": float(nll.mean()), "hdr_coverage": coverage,
            "mc_samples": mc_samples}


def _uncertainty_fusion_calibration(Y, ps):
    """Fused-Gaussian NLL/coverage + learned per-BS precision summary.

    ps: (N, K, 4) packed [p_k | s_k] from the uncertainty fusion head.
    Fused estimate: precision-weighted mean; fused variance per axis is
    1 / sum_k exp(-s_k) (independent-BS assumption — matches the fusion rule).
    """
    ps = np.asarray(ps, dtype=np.float64)
    p, s = ps[..., 0:2], ps[..., 2:4]
    w = np.exp(-s)                                            # (N, K, 2)
    fused_mu = (w * p).sum(axis=1) / (w.sum(axis=1) + 1e-8)
    fused_var = 1.0 / (w.sum(axis=1) + 1e-8)
    out = _gaussian_diag_calibration(Y, fused_mu, np.log(fused_var))
    out["per_bs_mean_precision"] = [float(x) for x in
                                    w.mean(axis=(0, 2))]      # (K,)
    return out


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def compute_all_metrics(predictions, Y_test, extras=None, test_idx=None,
                        raw_pred=None, output_mode=None, fusion_mode=None,
                        n_mixtures=5, provenance=None):
    predictions = np.asarray(predictions, dtype=np.float64)
    Y_test = np.asarray(Y_test, dtype=np.float64)
    errors = np.sqrt(((predictions - Y_test) ** 2).sum(axis=1))

    metrics = {"core": core_error_metrics(errors), "breakdowns": {}}

    extras = extras or {}
    if test_idx is not None:
        test_idx = np.asarray(test_idx)

        if "is_los_kept" in extras:
            is_los = np.asarray(extras["is_los_kept"])[test_idx].astype(bool)
            metrics["breakdowns"]["los"] = {
                "LoS-kept": _subset_stats(errors, is_los),
                "NLoS-only": _subset_stats(errors, ~is_los),
            }

        if "snr_detector_db" in extras:
            snr = np.asarray(extras["snr_detector_db"])[test_idx]
            bins = {}
            for lo, hi in zip(SNR_BIN_EDGES_DB[:-1], SNR_BIN_EDGES_DB[1:]):
                stats = _subset_stats(errors, (snr >= lo) & (snr < hi))
                if stats:
                    lo_s = f"{lo:+.0f}" if np.isfinite(lo) else "-inf"
                    hi_s = f"{hi:+.0f}" if np.isfinite(hi) else "+inf"
                    bins[f"[{lo_s},{hi_s})dB"] = stats
            metrics["breakdowns"]["snr"] = bins

        if "has_paths_per_bs" in extras:
            n_vis = np.asarray(extras["has_paths_per_bs"])[test_idx] \
                .astype(bool).sum(axis=1)
            metrics["breakdowns"]["n_visible_bs"] = {
                str(k): s for k in range(int(n_vis.max()) + 1)
                if (s := _subset_stats(errors, n_vis == k))
            }

    if raw_pred is not None and output_mode is not None:
        try:
            if output_mode == "heteroscedastic":
                arr = np.asarray(raw_pred, dtype=np.float64)
                metrics["calibration"] = _gaussian_diag_calibration(
                    Y_test, arr[:, 0:2], arr[:, 2:4])
            elif output_mode == "mdn":
                metrics["calibration"] = _mdn_calibration(
                    Y_test, raw_pred, n_mixtures)
            if fusion_mode == "uncertainty":
                ps = raw_pred.get("ps") if isinstance(raw_pred, dict) else None
                if ps is not None:
                    metrics["calibration_fused"] = \
                        _uncertainty_fusion_calibration(Y_test, ps)
        except Exception as e:  # calibration must never sink a finished run
            metrics["calibration_error"] = f"{type(e).__name__}: {e}"

    prov = dict(provenance or {})
    if test_idx is not None:
        prov["test_idx_sha1"] = hashlib.sha1(
            np.sort(np.asarray(test_idx, dtype=np.int64)).tobytes()
        ).hexdigest()
        prov["n_test"] = int(len(test_idx))
    prov["timestamp"] = datetime.now().isoformat(timespec="seconds")
    metrics["provenance"] = prov

    return metrics


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return _json_sanitize(obj.tolist())
    return obj


def save_metrics_json(out_dir, metrics):
    path = os.path.join(out_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(_json_sanitize(metrics), f, indent=2)
    print(f"Metrics saved: {path}")
    return path


def print_metrics(metrics, title=""):
    c = metrics["core"]
    print(f"\n--- Test Results {title} ---")
    print(f"Mean   : {c['mean_m']:.2f} m    RMSE: {c['rmse_m']:.2f} m")
    print(f"Median : {c['p50_m']:.2f} m    p67 : {c['p67_m']:.2f} m")
    print(f"p90    : {c['p90_m']:.2f} m    p95 : {c['p95_m']:.2f} m")
    cdf = "  ".join(f"≤{t}m: {v * 100:.1f}%" for t, v in c["cdf_at_m"].items())
    print(f"CDF    : {cdf}")

    for name, groups in metrics.get("breakdowns", {}).items():
        print(f"  by {name}:")
        for label, s in groups.items():
            if s:
                print(f"    {label:14s} n={s['n']:5d}  mean={s['mean_m']:.2f}  "
                      f"median={s['median_m']:.2f}  p90={s['p90_m']:.2f} m")

    for key in ("calibration", "calibration_fused"):
        if key in metrics:
            cal = metrics[key]
            cov = "  ".join(f"{q}: {v * 100:.1f}%"
                            for q, v in cal["hdr_coverage"].items())
            print(f"  {key}: NLL={cal['nll_mean']:.3f}  coverage[{cov}]")
            if "per_bs_mean_precision" in cal:
                print(f"    per-BS mean precision: "
                      f"{['%.3g' % p for p in cal['per_bs_mean_precision']]}")
