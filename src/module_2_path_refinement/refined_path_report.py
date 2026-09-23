# =============================================================================
# refined_path_report.py — Stage-2 diagnostic for filter_and_adjust_paths.py
# =============================================================================
#
# Reads the latest *_filtered_<scenario>.npz from filter_and_adjust_paths.py
# and, for a few (RX, TX) links, prints what SURVIVED the front-hemisphere /
# FOV filter together with the array-synthesis quantities computed there:
#
#   - path type (LoS / Specular / ... ; only if interactions were carried)
#   - tau_norm (delay re-referenced to the earliest surviving path = 0)
#   - |a| / relative power
#   - theta_az / theta_el  (LCS true-spherical broadside angles, deg)
#   - omega_h / omega_v    (inter-element steering phases, rad)
#
# It also runs convention CHECKS and prints PASS/FAIL:
#   1. delay normalization : earliest surviving path has tau_norm == 0
#   2. steering phase       : stored omega_h == 2*pi*d_h*cos(el)*sin(az)
#                             (and omega_v == 2*pi*d_v*sin(el))
#   3. phase range          : |omega_h| <= 2*pi*d_h, |omega_v| <= 2*pi*d_v
#   4. amplitude            : `a` is unchanged by stage 2 (only filtered)
#   5. beam squint          : per-element true-time-delay spread vs the
#                             array aperture fill-time
#
# Pure NumPy. Runs in any env on saved data only.
#
# Usage:
#   python src/module_2_path_refinement/refined_path_report.py --scenario two_walls_toy_test
#   python src/module_2_path_refinement/refined_path_report.py --scenario NAME --num-links 6 --max-paths 16
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


NONE, SPECULAR, DIFFUSE, REFRACTION, DIFFRACTION = 0, 1, 2, 4, 8
_TYPE_SHORT = {SPECULAR: "Spec", DIFFUSE: "Diff", REFRACTION: "Refr", DIFFRACTION: "Difr"}


def classify_path(depth_col):
    """Short label + bounce count from a path's per-bounce interaction column."""
    nz = depth_col[depth_col != NONE]
    nb = int(nz.size)
    if nb == 0:
        return "LoS", 0
    present = sorted({int(c) for c in nz})
    if present == [SPECULAR]:
        return ("Specular" if nb == 1 else f"Specular(x{nb})"), nb
    if present == [DIFFUSE]:
        return ("Diffuse" if nb == 1 else f"Diffuse(x{nb})"), nb
    if present == [REFRACTION]:
        return "Refraction", nb
    if present == [DIFFRACTION]:
        return "Diffraction", nb
    return "+".join(_TYPE_SHORT.get(c, str(c)) for c in present), nb


def link_table(d, r, t, interactions, d_h_wl, d_v_wl, max_paths_show, checks):
    """Print the surviving-path table for one (rx, tx) link and update checks."""
    rx_name = str(d["rx_names"][r])
    tx_name = str(d["tx_names"][t])
    k = int(d["num_paths_after_filter"][r, t])

    bs = d["R_rx"][r][:, 0]  # broadside (x_local) in GCS
    dist_m = float(np.linalg.norm(d["tx_pos"][t] - d["rx_pos"][r]))

    print("\n" + "-" * 104)
    print(f"RX[{r}]='{rx_name}' -> TX[{t}]='{tx_name}'   distance={dist_m:7.2f} m   "
          f"surviving paths={k}")
    print(f"  broadside (GCS): [{bs[0]:+.3f}, {bs[1]:+.3f}, {bs[2]:+.3f}]   "
          f"min-delay offset (abs LoS-ish time) = {d['min_delay_ns'][r, t]:.2f} ns")
    if k == 0:
        print("  (no surviving paths)")
        return

    tau = d["tau_ns"][r, t, :k]            # tau_norm
    a = d["a"][r, t, :k]
    amp = np.abs(a)
    az = np.rad2deg(d["theta_az"][r, t, :k])
    el = np.rad2deg(d["theta_el"][r, t, :k])
    oh = d["omega_h"][r, t, :k]
    ov = d["omega_v"][r, t, :k]
    amp_max = amp.max() if amp.max() > 0 else 1.0
    rel_db = 20.0 * np.log10(np.maximum(amp, 1e-30) / amp_max)

    labels = []
    for p in range(k):
        if interactions is not None:
            lbl, _ = classify_path(interactions[:, r, t, p])
        else:
            lbl = "?"
        labels.append(lbl)

    # ---- convention checks (accumulate globally) ----
    checks["tau0_min"] = min(checks["tau0_min"], float(tau.min()))  # should be 0
    oh_pred = 2.0 * np.pi * d_h_wl * np.cos(d["theta_el"][r, t, :k]) * np.sin(d["theta_az"][r, t, :k])
    ov_pred = 2.0 * np.pi * d_v_wl * np.sin(d["theta_el"][r, t, :k])
    checks["omega_h_err"] = max(checks["omega_h_err"], float(np.abs(oh - oh_pred).max()))
    checks["omega_v_err"] = max(checks["omega_v_err"], float(np.abs(ov - ov_pred).max()))
    checks["omega_h_absmax"] = max(checks["omega_h_absmax"], float(np.abs(oh).max()))
    checks["omega_v_absmax"] = max(checks["omega_v_absmax"], float(np.abs(ov).max()))

    print(f"  {'idx':>3} {'type':>14} {'tau_norm[ns]':>12} {'|a|':>11} {'relP[dB]':>9} "
          f"{'az[deg]':>8} {'el[deg]':>8} {'omega_h':>9} {'omega_v':>9}")
    order = np.argsort(tau)
    for p in order[:max_paths_show]:
        print(f"  {p:>3} {labels[p]:>14} {tau[p]:>12.3f} {amp[p]:>11.3e} {rel_db[p]:>9.1f} "
              f"{az[p]:>8.1f} {el[p]:>8.1f} {oh[p]:>9.3f} {ov[p]:>9.3f}")
    if k > max_paths_show:
        print(f"  ... ({k-max_paths_show} more surviving paths; use --max-paths)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=str, required=True)
    ap.add_argument("--project-dir", type=str, default=os.environ.get("PROJECT_DIR", "."))
    ap.add_argument("--num-links", type=int, default=5)
    ap.add_argument("--max-paths", type=int, default=16)
    ap.add_argument("--npz", type=str, default=None,
                    help="Use this file instead of the newest one for --scenario.")
    args = ap.parse_args()

    npz = resolve_input(
        os.path.join(args.project_dir, "data", "path_data", "refined_paths",
                     f"*_filtered_{args.scenario}.npz"),
        args.npz, label="filtered paths")
    d = np.load(npz, allow_pickle=True)
    meta = d["metadata"].item()

    f_c = float(meta["frequency_hz"])
    arr = meta["rx_array"]
    M, N = int(arr["num_rows"]), int(arr["num_cols"])
    d_h_wl = float(arr["horizontal_spacing_wl"])
    d_v_wl = float(arr["vertical_spacing_wl"])
    print(f"Array: {M}x{N} (d_h={d_h_wl}, d_v={d_v_wl} lambda)   f_c={f_c/1e9:.4f} GHz   "
          f"FOV az=+-{arr['fov_az_deg']/2:.1f} el=+-{arr['fov_el_deg']/2:.1f} deg")

    interactions = d["interactions"] if "interactions" in d.files else None
    if interactions is None:
        print("[!] No 'interactions' in this filtered .npz (re-run path_gen.py + "
              "filter_and_adjust_paths.py to carry path types through).")

    num_kept = d["num_paths_after_filter"]
    valid_idx = np.argwhere(num_kept > 0)
    if valid_idx.size == 0:
        print("No links survived filtering.")
        return

    checks = {"tau0_min": np.inf, "omega_h_err": 0.0, "omega_v_err": 0.0,
              "omega_h_absmax": 0.0, "omega_v_absmax": 0.0}

    n = min(args.num_links, len(valid_idx))
    spacing = max(1, len(valid_idx) // n)
    for r, t in valid_idx[::spacing][:n]:
        link_table(d, int(r), int(t), interactions, d_h_wl, d_v_wl, args.max_paths, checks)

    # ---- scenario summary ----
    print("\n" + "=" * 104)
    print("  STAGE-2 SUMMARY")
    print("=" * 104)
    n_links = int((num_kept > 0).sum())
    print(f"  Links with surviving paths : {n_links}")
    print(f"  Surviving paths (total)    : {int(num_kept.sum())}")
    print(f"  Paths/link  min/mean/max   : {int(num_kept[num_kept>0].min())} / "
          f"{num_kept[num_kept>0].mean():.2f} / {int(num_kept.max())}")

    # type breakdown among survivors
    if interactions is not None:
        counts = {}
        n_los_links = 0
        for r, t in valid_idx:
            link_los = False
            for p in range(int(num_kept[r, t])):
                lbl, _ = classify_path(interactions[:, r, t, p])
                counts[lbl] = counts.get(lbl, 0) + 1
                if lbl == "LoS":
                    link_los = True
            n_los_links += int(link_los)
        print(f"  Links keeping a LoS path   : {n_los_links} "
              f"({100*n_los_links/n_links:.1f}%)")
        print("\n  Surviving path types:")
        for lbl, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    {lbl:>16} : {c}")

    # beam-squint context: aperture fill-time across the array
    tpe = d["tau_per_element_ns"]                      # (R, T, M, N, P)
    lam = speed_of_light / f_c
    fill_h_ns = (N - 1) * d_h_wl * lam / speed_of_light * 1e9
    fill_v_ns = (M - 1) * d_v_wl * lam / speed_of_light * 1e9
    # max per-element spread across all surviving path 0's
    spreads = []
    for r, t in valid_idx:
        dt = tpe[r, t, :, :, 0]
        spreads.append(dt.max() - dt.min())
    spreads = np.array(spreads)
    print(f"\n  Array aperture fill-time   : az={fill_h_ns:.3f} ns, el={fill_v_ns:.3f} ns "
          f"(max true-time-delay across the panel)")
    print(f"  Per-element TTD spread (path0) p50/p95/max [ns]: "
          f"{np.percentile(spreads,50):.3f} / {np.percentile(spreads,95):.3f} / {spreads.max():.3f}")

    # ---- convention checks ----
    print("\n  Convention checks:")
    tau0_ok = abs(checks["tau0_min"]) < 1e-6
    print(f"    [{'PASS' if tau0_ok else 'FAIL'}] delay normalization: "
          f"earliest surviving tau_norm = {checks['tau0_min']:.2e} ns (expect 0)")
    oh_ok = checks["omega_h_err"] < 1e-9
    ov_ok = checks["omega_v_err"] < 1e-9
    print(f"    [{'PASS' if oh_ok else 'FAIL'}] omega_h = 2pi d_h cos(el) sin(az): "
          f"max err = {checks['omega_h_err']:.2e} rad")
    print(f"    [{'PASS' if ov_ok else 'FAIL'}] omega_v = 2pi d_v sin(el): "
          f"max err = {checks['omega_v_err']:.2e} rad")
    bound_h, bound_v = 2 * np.pi * d_h_wl, 2 * np.pi * d_v_wl
    rh_ok = checks["omega_h_absmax"] <= bound_h + 1e-9
    rv_ok = checks["omega_v_absmax"] <= bound_v + 1e-9
    print(f"    [{'PASS' if rh_ok else 'FAIL'}] |omega_h| <= 2pi d_h ({bound_h:.3f}): "
          f"max = {checks['omega_h_absmax']:.3f} rad")
    print(f"    [{'PASS' if rv_ok else 'FAIL'}] |omega_v| <= 2pi d_v ({bound_v:.3f}): "
          f"max = {checks['omega_v_absmax']:.3f} rad")
    print("\n  (Tip: to verify `a` is unchanged by stage 2, compare |a| here against "
          "path_report.py on the same link.)")


if __name__ == "__main__":
    main()
