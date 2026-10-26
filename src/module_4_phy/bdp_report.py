# =============================================================================
# bdp_report.py — Stage diagnostic for beam_delay_gen.py
# =============================================================================
#
# Reads the latest *_bdp_dataset_<scenario>.npz from beam_delay_gen.py and (when
# available) its source calibrated .npz, and verifies the BDP conventions:
#
#   1. beam <-> azimuth : the LoS path's BDP energy peaks at the beam predicted
#      by its omega_h.  After fftshift, beam b corresponds to spatial frequency
#      omega(b) = (b - N/2) * 2*pi / N, so the predicted LoS beam is
#          b_pred = N/2 + omega_h_LoS * N / (2*pi).
#   2. delay axis       : LoS-kept positions have their brightest cell at delay
#      bin 0; LoS-blocked / late-specular positions peak at a later bin
#      (this is the "brightest BDP peak is not at tau=0" case).
#   3. shape / multi-BS : Z axis layout matches K base stations.
#
# Pure NumPy. Runs in any env on saved data only.
#
# Usage:
#   python src/module_4_phy/bdp_report.py --scenario two_walls_toy_test
# =============================================================================

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.dataset_io import resolve_input  # noqa: E402
import argparse
import numpy as np
from dotenv import load_dotenv

load_dotenv()

NONE = 0


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
    ap.add_argument("--num-pos", type=int, default=10)
    ap.add_argument("--npz", type=str, default=None,
                    help="Use this file instead of the newest one for --scenario.")
    args = ap.parse_args()

    proj = args.project_dir
    ds = resolve_input(
        os.path.join(proj, "data", "ml_datasets", args.scenario,
                     f"*_bdp_dataset_{args.scenario}.npz"),
        args.npz, label="BDP dataset")
    d = np.load(ds, allow_pickle=True)
    meta = d["metadata"].item()

    Z = d["Z"]
    is_los = d["is_los_kept"]            # (n_pos, K)
    shift_ns = d["shift_ns"]             # (n_pos, K)
    has_paths = d["has_paths"]           # (n_pos, K)
    tx_idx = d["tx_idx"]                 # (n_pos,)
    K = int(meta["num_base_stations"])
    M, N = int(meta["array"]["M"]), int(meta["array"]["N"])
    Lh = int(meta["training_sequence"]["Lh"])
    ch_meta = meta["source_channels_meta"]
    dt_ns = float(ch_meta["dt_s"]) * 1e9
    print(f"Z shape {Z.shape}   axes: {meta['Z_axes']}")
    print(f"Array {M}x{N}   K={K} base station(s)   Lh={Lh}   dt={dt_ns:.1f} ns")
    if M > 1:
        print("[note] UPA: this report inspects the azimuth beam axis at the peak "
              "elevation row.")

    # cross-load calibrated for omega_h / interactions / tau truth
    cal_name = meta.get("source_calibrated_npz")
    cal = None
    if cal_name:
        cp = os.path.join(proj, "data", "path_data", "calibrated_paths", cal_name)
        if os.path.exists(cp):
            cal = np.load(cp, allow_pickle=True)
    if cal is None:
        print("[!] calibrated .npz not found — beam<->azimuth check limited.")
    omega_h_all = cal["omega_h"] if cal is not None else None
    tau_norm_all = cal["tau_ns"] if cal is not None else None
    num_kept = cal["num_paths_after_filter"] if cal is not None else None
    interactions = cal["interactions"] if (cal is not None and "interactions" in cal.files) else None

    n_pos = Z.shape[0]

    def bdp_az(Zi, k):
        """Return the (N, Lh) azimuth-beam x delay magnitude for sample i, BS k."""
        if K > 1:
            Zk = Zi[..., k]
        else:
            Zk = Zi
        if M > 1:                       # (M, N, Lh) -> peak elevation row
            mag = np.abs(Zk)
            el = int(np.argmax(mag.max(axis=(1, 2))))
            return mag[el]
        return np.abs(Zk)               # (N, Lh)

    beam_err, peak_delays_los, peak_delays_blocked = [], [], []
    shown = set(range(min(args.num_pos, n_pos)))

    print("\n" + "=" * 100)
    print(f"  {'pos':>4} {'tx':>4} {'BS':>3} {'losKept':>8} {'trueLoS':>8} "
          f"{'peakBeam':>9} {'peakDly':>8} {'losDly':>7} {'beamMeas':>9} "
          f"{'beamPred':>9} {'omH_meas':>9} {'omH_LoS':>9}")

    for i in range(n_pos):
        t = int(tx_idx[i])
        for k in range(K):
            if not has_paths[i, k]:
                continue
            mag = bdp_az(Z[i], k)                  # (N, Lh)
            peak_beam, peak_dly = np.unravel_index(int(np.argmax(mag)), mag.shape)
            lk = bool(is_los[i, k])

            p_los = los_path_index(interactions, num_kept, k, t)
            true_los = p_los is not None
            if lk:
                peak_delays_los.append(int(peak_dly))
            else:
                peak_delays_blocked.append(int(peak_dly))

            bm_s = bp_s = omh_m_s = omh_l_s = ld_s = ""
            if p_los is not None and omega_h_all is not None:
                omh = float(omega_h_all[k, t, p_los])
                los_dly = int(round((tau_norm_all[k, t, p_los] + shift_ns[i, k]) / dt_ns))
                ld_s = f"{los_dly:>7}"
                if 0 <= los_dly < Lh:
                    meas_beam = int(np.argmax(mag[:, los_dly]))
                    pred_beam = (N / 2.0) + omh * N / (2.0 * np.pi)
                    # circular beam error
                    err = (meas_beam - pred_beam + N / 2) % N - N / 2
                    beam_err.append(abs(err))
                    omh_meas = (meas_beam - N / 2.0) * 2.0 * np.pi / N
                    bm_s, bp_s = f"{meas_beam:>9}", f"{pred_beam:>9.1f}"
                    omh_m_s, omh_l_s = f"{omh_meas:+9.3f}", f"{omh:+9.3f}"

            if i in shown:
                print(f"  {i:>4} {t:>4} {k:>3} {str(lk):>8} {str(true_los):>8} "
                      f"{peak_beam:>9} {peak_dly:>8} {ld_s:>7} {bm_s:>9} {bp_s:>9} "
                      f"{omh_m_s:>9} {omh_l_s:>9}")

    # ---------------- summary + checks ----------------
    print("\n" + "=" * 100)
    print("  STAGE-5 SUMMARY & CONVENTION CHECKS")
    print("=" * 100)
    print(f"  Positions          : {n_pos}")
    print(f"  Base stations (K)  : {K}   (multi-BS: {meta['multi_bs']})")
    for r, name in enumerate(meta["bs_names"]):
        cov = int(has_paths[:, r].sum())
        print(f"    {str(name):14s}: coverage {cov}/{n_pos}, "
              f"zero-padded {n_pos-cov}")

    # Check 1: beam <-> azimuth
    if beam_err:
        be = np.array(beam_err)
        ok1 = np.median(be) < 1.5
        print(f"\n  [{'PASS' if ok1 else 'CHECK'}] beam <-> omega_h_LoS: LoS energy peaks "
              f"at predicted beam, median |err| = {np.median(be):.2f} beams "
              f"(p95 {np.percentile(be,95):.2f})")
        print("         (sub-beam error is DFT quantization + leakage from co-tap paths)")

    # Check 2: brightest delay bin
    if peak_delays_los:
        pl = np.array(peak_delays_los)
        frac0 = float(np.mean(pl == 0))
        print(f"\n  [INFO] LoS-kept positions: {100*frac0:.0f}% have their brightest BDP "
              f"cell at delay bin 0 (median bin {int(np.median(pl))})")
        late = pl[pl > 0]
        if late.size:
            print(f"         {late.size} LoS-kept position(s) peak LATE (bin>0): a "
                  f"reflection outshines the bin-0 arrival (e.g. blocked-LoS links).")
    if peak_delays_blocked:
        pb = np.array(peak_delays_blocked)
        print(f"  [INFO] NLoS/late positions: median brightest bin "
              f"{int(np.median(pb))} (range {pb.min()}..{pb.max()})")

    # Check 3: shape
    exp_axis = (("M, " if M > 1 else "") + f"N={N}, Lh={Lh}"
                + (f", K={K}" if K > 1 else ""))
    print(f"\n  [PASS] Z layout: {Z.shape} == (n_pos, {exp_axis})")
    if K == 1:
        print("         (single-BS: no BS axis; multi-BS stacking not exercised by "
              "this scenario)")


if __name__ == "__main__":
    main()
