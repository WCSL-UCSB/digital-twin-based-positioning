# =============================================================================
# inspect_delays.py — Diagnostic for the delay chain across the pipeline
# =============================================================================
#
# Reads the latest filtered .npz and cross-checks the stored min_delay_ns
# and tau_ns values against geometric LoS delays computed from stored
# TX / RX positions. Useful for debugging any "shift looks wrong" symptom
# in channel_tap_gen.py.
#
# Usage:
#   python inspect_delays.py                   # auto-picks scenario from env
#   python inspect_delays.py --scenario NAME
#   python inspect_delays.py --scenario NAME --num-samples 10
# =============================================================================

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.dataset_io import resolve_input  # noqa: E402
import argparse
import numpy as np
from scipy.constants import speed_of_light
from dotenv import load_dotenv

load_dotenv()


def inspect_link(d, r, t):
    """Print a detailed report for one (rx, tx) link."""
    rx_pos       = d["rx_pos"]
    tx_pos       = d["tx_pos"]
    min_delay_ns = d["min_delay_ns"]
    tau_ns       = d["tau_ns"]
    num_kept     = d["num_paths_after_filter"]
    rx_names     = d["rx_names"]
    tx_names     = d["tx_names"]

    k = int(num_kept[r, t])
    dist_m = np.linalg.norm(tx_pos[t] - rx_pos[r])
    tau_los_geom_ns = 1e9 * dist_m / speed_of_light

    print(f"\nLink RX[{r}]='{rx_names[r]}' -> TX[{t}]='{tx_names[t]}'")
    print(f"  TX pos                       : {tx_pos[t]}")
    print(f"  RX pos                       : {rx_pos[r]}")
    print(f"  Distance                     : {dist_m:.2f} m")
    print(f"  Geometric LoS delay          : {tau_los_geom_ns:.2f} ns")
    print(f"  Stored min_delay_ns          : {min_delay_ns[r, t]:.2f} ns")
    print(f"  tau_ns[0..{k-1}]               : {tau_ns[r, t, :k]}")
    print(f"  tau_ns min / max             : "
          f"{tau_ns[r, t, :k].min():.2f} / {tau_ns[r, t, :k].max():.2f} ns")

    if tau_los_geom_ns > 0:
        ratio = min_delay_ns[r, t] / tau_los_geom_ns
        excess_ns = min_delay_ns[r, t] - tau_los_geom_ns
        excess_m = excess_ns * 1e-9 * speed_of_light
        print(f"  min_delay / geom_LoS ratio   : {ratio:.3f}")
        print(f"  min_delay - geom_LoS         : {excess_ns:+.2f} ns "
              f"(≈ {excess_m:+.1f} m of excess path length)")

        if abs(ratio - 1.0) < 0.01:
            verdict = "LoS-kept (min arrival matches geometric LoS within 1%)"
        elif ratio < 1.0:
            verdict = "⚠ IMPOSSIBLE — min arrival is earlier than geometric LoS"
        elif ratio < 2.0:
            verdict = "NLoS-only, short reflection (reasonable)"
        elif ratio < 5.0:
            verdict = "NLoS-only, long reflection (check scene extent)"
        else:
            verdict = "⚠ SUSPICIOUS — ratio > 5 suggests a unit or reference bug"
        print(f"  Verdict                      : {verdict}")


def summary_stats(d):
    """Aggregate ratio distribution across all links with paths."""
    rx_pos       = d["rx_pos"]
    tx_pos       = d["tx_pos"]
    min_delay_ns = d["min_delay_ns"]
    num_kept     = d["num_paths_after_filter"]
    has_paths    = num_kept > 0

    # Geometric LoS per link
    dxyz = tx_pos[np.newaxis, :, :] - rx_pos[:, np.newaxis, :]
    d_geom = np.linalg.norm(dxyz, axis=-1)
    tau_los_geom_ns = 1e9 * d_geom / speed_of_light

    # Mask to kept, non-degenerate links
    valid = has_paths & (tau_los_geom_ns > 1e-6)
    if not np.any(valid):
        print("\nNo valid links to summarize.")
        return

    ratio = min_delay_ns[valid] / tau_los_geom_ns[valid]
    excess = min_delay_ns[valid] - tau_los_geom_ns[valid]

    n_impossible = int((ratio < 0.99).sum())
    n_los_kept   = int((np.abs(ratio - 1.0) < 0.01).sum())
    n_short_nlos = int(((ratio >= 1.01) & (ratio < 2.0)).sum())
    n_long_nlos  = int(((ratio >= 2.0)  & (ratio < 5.0)).sum())
    n_suspicious = int((ratio >= 5.0).sum())
    n_total      = int(valid.sum())

    print("\n" + "=" * 70)
    print("  DELAY CHAIN SUMMARY (over links with surviving paths)")
    print("=" * 70)
    print(f"  Total links with paths   : {n_total}")
    print(f"  LoS-kept (ratio ≈ 1.00)  : {n_los_kept:6d} "
          f"({100*n_los_kept/n_total:.1f}%)")
    print(f"  NLoS, short (1.0-2.0x)   : {n_short_nlos:6d} "
          f"({100*n_short_nlos/n_total:.1f}%)")
    print(f"  NLoS, long  (2.0-5.0x)   : {n_long_nlos:6d} "
          f"({100*n_long_nlos/n_total:.1f}%)")
    print(f"  Suspicious  (>5.0x)      : {n_suspicious:6d} "
          f"({100*n_suspicious/n_total:.1f}%)")
    if n_impossible > 0:
        print(f"  ⚠ Impossible (<1.0x)     : {n_impossible:6d} "
              f"({100*n_impossible/n_total:.1f}%)")

    print("\n  Ratio (min_delay / geom_LoS) distribution:")
    for p, q in zip([5, 25, 50, 75, 95], np.percentile(ratio, [5, 25, 50, 75, 95])):
        print(f"    p{p:02d}: {q:6.2f}x")
    print(f"    min: {ratio.min():6.2f}x")
    print(f"    max: {ratio.max():6.2f}x")

    print("\n  Excess delay (min_delay - geom_LoS) distribution (ns):")
    for p, q in zip([5, 25, 50, 75, 95], np.percentile(excess, [5, 25, 50, 75, 95])):
        excess_m = q * 1e-9 * speed_of_light
        print(f"    p{p:02d}: {q:+8.2f} ns  (≈ {excess_m:+7.1f} m)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=str, default=None,
                    help="Scenario name (default: nyc_high_rise_test_ml)")
    ap.add_argument("--num-samples", type=int, default=5,
                    help="Number of individual links to inspect in detail")
    ap.add_argument("--project-dir", type=str,
                    default=os.environ.get("PROJECT_DIR", "."))
    ap.add_argument("--npz", type=str, default=None,
                    help="Use this file instead of the newest one for --scenario.")
    args = ap.parse_args()

    scenario = args.scenario or "nyc_high_rise_test_ml"

    fpath = resolve_input(
        os.path.join(args.project_dir, "data", "path_data", "refined_paths",
                     f"*_filtered_{scenario}.npz"),
        args.npz, label="filtered paths")
    d = np.load(fpath, allow_pickle=True)

    num_kept = d["num_paths_after_filter"]
    has = num_kept > 0
    valid_idx = np.argwhere(has)

    if valid_idx.size == 0:
        print("No links with surviving paths found.")
        return

    # Detailed inspection on a few sample links, spread across the dataset
    n_samples = min(args.num_samples, len(valid_idx))
    sample_spacing = max(1, len(valid_idx) // n_samples)
    sample_indices = valid_idx[::sample_spacing][:n_samples]

    for r, t in sample_indices:
        inspect_link(d, int(r), int(t))

    # Aggregate summary
    summary_stats(d)


if __name__ == "__main__":
    main()