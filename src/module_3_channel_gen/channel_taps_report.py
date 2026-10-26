# =============================================================================
# channel_taps_report.py — Stage diagnostic for channel_taps_gen.py
# =============================================================================
#
# Reads the latest *_channels_<scenario>.npz from channel_taps_gen.py and (when
# available) its source *_calibrated_<scenario>.npz, and verifies the channel-
# tap conventions:
#
#   1. tau = 0 is geometric LoS : LoS-kept links peak at tap 0; NLoS-only links
#      peak near shift/dt.
#   2. is_los_kept vs true LoS  : is_los_kept is DELAY-based, so a through-wall
#      refraction at the LoS delay can set it True with no real LoS path. We
#      cross-check against interaction types and flag the mismatches.
#   3. beamforming ramp         : the inter-column phase step of H at the LoS
#      tap should equal the LoS path's omega_h (end-to-end steering check).
#   4. omega_v inert (ULA)      : M==1 => the row index is 0 => omega_v never
#      enters. Confirmed structurally.
#   5. center LoS phase ~ 0     : at the center column / LoS tap, a real a_LoS
#      with carrier exp(-j2pi f_c * ~0) ~ 1 gives ~0 phase.
#
# Pure NumPy. Runs in any env on saved data only.
#
# Usage:
#   python src/module_3_channel_gen/channel_taps_report.py --scenario two_walls_toy_test
# =============================================================================

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.dataset_io import resolve_input  # noqa: E402
import argparse
import numpy as np
from dotenv import load_dotenv

load_dotenv()

NONE, SPECULAR, DIFFUSE, REFRACTION, DIFFRACTION = 0, 1, 2, 4, 8


def los_path_index(interactions, num_kept, r, t):
    if interactions is None:
        return None
    for p in range(int(num_kept[r, t])):
        if not np.any(interactions[:, r, t, p] != NONE):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=str, required=True)
    ap.add_argument("--project-dir", type=str, default=os.environ.get("PROJECT_DIR", "."))
    ap.add_argument("--num-links", type=int, default=8)
    ap.add_argument("--npz", type=str, default=None,
                    help="Use this file instead of the newest one for --scenario.")
    args = ap.parse_args()

    proj = args.project_dir
    ch_npz = resolve_input(
        os.path.join(proj, "data", "channel_data",
                     f"*_channels_{args.scenario}.npz"),
        args.npz, label="channels")
    ch = np.load(ch_npz, allow_pickle=True)
    meta = ch["metadata"].item()

    H = ch["H"]                          # (R, T, M, N, T_fixed) complex64
    is_los_kept = ch["is_los_kept"]
    shift_applied = ch["shift_applied"]
    tau_los = ch["tau_los_ns"]
    num_used = ch["num_paths"]
    rx_names, tx_names = ch["rx_names"], ch["tx_names"]
    R, T, M, N, T_fixed = H.shape

    dt_ns = float(meta["dt_s"]) * 1e9
    f_c = float(meta["f_carrier_hz"])
    print(f"Array {M}x{N}  dt={dt_ns:.1f} ns  T_fixed={T_fixed}  f_c={f_c/1e9:.4f} GHz  "
          f"TTD={meta.get('include_true_time_delay')}  spatial_phase={meta.get('include_spatial_phase')}")

    # cross-load the source calibrated npz for per-path truth (omega_h, types)
    cal_name = meta.get("source_calibrated_npz")
    cal = None
    if cal_name:
        cal_path = os.path.join(proj, "data", "path_data", "calibrated_paths", cal_name)
        if os.path.exists(cal_path):
            cal = np.load(cal_path, allow_pickle=True)
    if cal is None:
        print("[!] Source calibrated .npz not found — beamforming-ramp and true-LoS "
              "checks limited.")
    omega_h_all = cal["omega_h"] if cal is not None else None
    tau_norm_all = cal["tau_ns"] if cal is not None else None
    num_kept = cal["num_paths_after_filter"] if cal is not None else num_used
    interactions = cal["interactions"] if (cal is not None and "interactions" in cal.files) else None

    center_col = (N - 1) // 2
    valid_idx = np.argwhere(num_used > 0)

    # accumulators for summary
    peaktap_loskept, peaktap_nlos = [], []
    ramp_err = []           # |measured per-col step - omega_h_LoS|
    center_phase = []       # |angle(H center, LoS tap)|
    mismatch_links = []     # is_los_kept True but no true LoS path

    print("\n" + "=" * 104)
    print(f"  {'RX->TX':>8} {'losKept':>8} {'trueLoS':>8} {'shift[ns]':>10} "
          f"{'peakTap':>8} {'peak[ns]':>9} {'expTap':>7} {'cphase[deg]':>12} "
          f"{'rampMeas':>9} {'omH_LoS':>9}")
    n = min(args.num_links, len(valid_idx))
    spacing = max(1, len(valid_idx) // n)
    shown = set(map(tuple, valid_idx[::spacing][:n]))

    for rt in valid_idx:
        r, t = int(rt[0]), int(rt[1])
        Hl = H[r, t]                                   # (M, N, T_fixed)
        p_tap = np.abs(Hl).sum(axis=(0, 1)) ** 2       # energy per tap (proxy)
        peak_tap = int(np.argmax(p_tap))
        lk = bool(is_los_kept[r, t])
        sh = float(shift_applied[r, t])
        exp_tap = 0 if lk else int(round(sh / dt_ns))

        if lk:
            peaktap_loskept.append(peak_tap)
        else:
            peaktap_nlos.append((peak_tap, exp_tap))

        # true LoS path?
        p_los = los_path_index(interactions, num_kept, r, t)
        true_los = p_los is not None
        if lk and not true_los and interactions is not None:
            mismatch_links.append((r, t))

        cphase_s, ramp_s, omh_s = "", "", ""
        if p_los is not None and omega_h_all is not None:
            tau_los_rel = float(tau_norm_all[r, t, p_los] + sh)   # ~0 for LoS-kept
            los_tap = int(round(tau_los_rel / dt_ns))
            if 0 <= los_tap < T_fixed:
                row = Hl[0, :, los_tap]
                omh = float(omega_h_all[r, t, p_los])
                # inter-column phase step (robust to 2pi wrap): mean of
                # angle(H[n+1] * conj(H[n])) along the row-0 columns
                steps = np.angle(row[1:] * np.conj(row[:-1]))
                meas = float(np.mean(steps))
                ramp_err.append(abs(meas - omh))
                ramp_s = f"{meas:+9.3f}"
                omh_s = f"{omh:+9.3f}"
                # De-ramp by the LoS steering, then the coherent-sum phase is
                # the residual a_LoS*carrier phase (~0 for a LoS-kept link).
                # Using n_idx*omega_h removes the even-N half-element offset
                # that a single "center" column would otherwise carry.
                n_idx = np.arange(N) - (N - 1) / 2.0
                intercept = np.angle(np.sum(row * np.exp(-1j * n_idx * omh)))
                center_phase.append(abs(np.rad2deg(intercept)))
                cphase_s = f"{np.rad2deg(intercept):+12.1f}"

        if (r, t) in shown:
            print(f"  {f'{r}->{t}':>8} {str(lk):>8} {str(true_los):>8} {sh:>10.2f} "
                  f"{peak_tap:>8} {peak_tap*dt_ns:>9.1f} {exp_tap:>7} "
                  f"{cphase_s:>12} {ramp_s:>9} {omh_s:>9}")

    # ---------------- summary + checks ----------------
    print("\n" + "=" * 104)
    print("  STAGE-4 SUMMARY & CONVENTION CHECKS")
    print("=" * 104)
    n_links = int((num_used > 0).sum())
    n_lk = int((is_los_kept & (num_used > 0)).sum())
    print(f"  Links with taps     : {n_links}")
    print(f"  is_los_kept (delay) : {n_lk}")

    # Check 1: tau=0 geometric LoS
    if peaktap_loskept:
        pk = np.array(peaktap_loskept)
        frac0 = float(np.mean(pk == 0))
        ok1 = frac0 >= 0.5
        print(f"\n  [{'PASS' if ok1 else 'CHECK'}] tau=0 geometric LoS: "
              f"{100*frac0:.0f}% of LoS-kept links peak at tap 0 "
              f"(median peak tap {int(np.median(pk))})")
    if peaktap_nlos:
        errs = [abs(p - e) for p, e in peaktap_nlos]
        print(f"         NLoS-only links: peak vs expected(shift/dt) "
              f"median |err| = {int(np.median(errs))} taps over {len(peaktap_nlos)} links")

    # Check 2: is_los_kept vs true LoS
    if interactions is not None:
        print(f"\n  [INFO] is_los_kept vs true-LoS mismatch: {len(mismatch_links)} link(s) "
              f"are delay-LoS-kept but have NO true LoS path")
        if mismatch_links:
            tags = ", ".join(f"{rx_names[r]}->{tx_names[t]}" for r, t in mismatch_links[:6])
            print(f"         (refraction/diffraction at the LoS delay) e.g. {tags}")

    # Check 3: beamforming ramp
    if ramp_err:
        re_ = np.array(ramp_err)
        ok3 = np.median(re_) < 0.1
        print(f"\n  [{'PASS' if ok3 else 'CHECK'}] beamforming ramp == omega_h_LoS: "
              f"median |meas-omega_h| = {np.median(re_):.4f} rad/col "
              f"(p95 {np.percentile(re_,95):.4f}) over {len(ramp_err)} LoS taps")
        print("         (residual is leakage from other paths sharing the LoS tap)")

    # Check 4: omega_v inert
    print(f"\n  [{'PASS' if M == 1 else 'N/A'}] omega_v inert: M={M} "
          f"=> row index is {{0}} => omega_v never multiplies (ULA)")

    # Check 5: center LoS phase ~ 0
    if center_phase:
        cp = np.array(center_phase)
        ok5 = np.median(cp) < 10.0
        print(f"\n  [{'PASS' if ok5 else 'CHECK'}] LoS-tap carrier+a phase ~ 0 (de-ramped): "
              f"median |phase| = {np.median(cp):.1f} deg (p95 {np.percentile(cp,95):.1f})")
        print("         (de-ramped by omega_h_LoS; ~0 confirms a_LoS real & carrier(tau~0)~1)")


if __name__ == "__main__":
    main()
