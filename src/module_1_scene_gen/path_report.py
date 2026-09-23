# =============================================================================
# path_report.py — Per-path type + property table for a path_gen.py output
# =============================================================================
#
# Reads the latest *_paths.npz written by path_gen.py and, for a few (RX, TX)
# links, prints a table classifying every traced path by interaction type
# (LoS / Specular / Diffuse / Refraction / Diffraction / mixtures) alongside
# its delay, excess delay vs the geometric LoS, amplitude, relative power, and
# the departure/arrival angles. A scenario-wide summary follows.
#
# Pure NumPy — no Sionna import — so it runs in any env on saved data only.
# Requires that path_gen.py was run AFTER the `interactions` field was added
# (older .npz files won't have it; the script says so and falls back to types
# inferred from delay only).
#
# Usage:
#   python src/module_1_scene_gen/path_report.py --scenario two_walls_toy_test
#   python src/module_1_scene_gen/path_report.py --scenario NAME --num-links 6 --max-paths 12
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


# InteractionType codes used by Sionna RT (see path_gen.py metadata).
NONE, SPECULAR, DIFFUSE, REFRACTION, DIFFRACTION = 0, 1, 2, 4, 8
_TYPE_SHORT = {SPECULAR: "Spec", DIFFUSE: "Diff", REFRACTION: "Refr", DIFFRACTION: "Difr"}


def wrap_deg(rad):
    """Wrap radians to (-180, 180] degrees."""
    return np.rad2deg((rad + np.pi) % (2.0 * np.pi) - np.pi)


def circ_std_rad(angles):
    """Circular standard deviation (rad) of a set of angles."""
    z = np.exp(1j * np.asarray(angles))
    R = np.abs(z.mean()) if z.size else 0.0
    R = min(max(R, 1e-12), 1.0)
    return float(np.sqrt(-2.0 * np.log(R)))


def classify_path(depth_col):
    """
    Map a path's per-bounce interaction column (length max_depth) to a short
    label and a bounce count. depth_col entries are InteractionType codes;
    NONE (0) entries are empty bounce slots.
    """
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
    # mixed interaction types along the path
    return "+".join(_TYPE_SHORT.get(c, str(c)) for c in present), nb


def link_table(d, r, t, interactions, max_paths_show, f_c):
    """Print the per-path table for one (rx, tx) link."""
    rx_pos = d["rx_pos"][r]
    tx_pos = d["tx_pos"][t]
    rx_name = str(d["rx_names"][r])
    tx_name = str(d["tx_names"][t])
    k = int(d["num_valid_paths"][r, t])

    dist_m = float(np.linalg.norm(tx_pos - rx_pos))
    tau_geom_ns = 1e9 * dist_m / speed_of_light

    print("\n" + "-" * 100)
    print(f"RX[{r}]='{rx_name}' {np.round(rx_pos,2)}  ->  TX[{t}]='{tx_name}' {np.round(tx_pos,2)}")
    print(f"  distance={dist_m:8.2f} m   geometric-LoS delay={tau_geom_ns:8.2f} ns   "
          f"valid paths={k}")

    if k == 0:
        print("  (no valid paths)")
        return

    tau = d["tau_ns"][r, t, :k]
    a = d["a"][r, t, :k]
    amp = np.abs(a)
    ang_deg = np.rad2deg(np.angle(a))                          # phase of a
    # residual: if `a` carries exp(-j2*pi*f_c*tau) this is ~constant across
    # paths/links; if `a` is phase-free, ang(a) itself is the constant one.
    resid_deg = wrap_deg(np.angle(a) + 2.0 * np.pi * f_c * tau * 1e-9)
    th_r = np.rad2deg(d["theta_r"][r, t, :k])
    ph_r = np.rad2deg(d["phi_r"][r, t, :k])
    th_t = np.rad2deg(d["theta_t"][r, t, :k])
    ph_t = np.rad2deg(d["phi_t"][r, t, :k])

    amp_max = amp.max() if amp.max() > 0 else 1.0
    rel_db = 20.0 * np.log10(np.maximum(amp, 1e-30) / amp_max)

    # Classify each path (if interactions available)
    labels, nbs, los_present = [], [], False
    for p in range(k):
        if interactions is not None:
            lbl, nb = classify_path(interactions[:, r, t, p])
        else:
            # Fallback: call it LoS if its delay matches the geometric LoS.
            is_los = abs(tau[p] - tau_geom_ns) < 0.5
            lbl, nb = ("LoS" if is_los else "?(reflect)"), (0 if is_los else 1)
        labels.append(lbl)
        nbs.append(nb)
        if lbl == "LoS":
            los_present = True

    print(f"  LoS path present: {'YES' if los_present else 'no'}")
    print(f"  {'idx':>3} {'type':>14} {'bnc':>3} {'tau[ns]':>10} {'excess[ns]':>11} "
          f"{'|a|':>11} {'relP[dB]':>9} {'ang(a)':>8} {'resid':>8} "
          f"{'AoD(th,ph)':>17} {'AoA(th,ph)':>17}")

    order = np.argsort(tau)  # earliest first
    shown = order[:max_paths_show]
    for p in shown:
        print(f"  {p:>3} {labels[p]:>14} {nbs[p]:>3} {tau[p]:>10.2f} "
              f"{tau[p]-tau_geom_ns:>+11.2f} {amp[p]:>11.3e} {rel_db[p]:>9.1f} "
              f"{ang_deg[p]:>8.1f} {resid_deg[p]:>8.1f} "
              f"{th_t[p]:>8.1f},{ph_t[p]:>7.1f} {th_r[p]:>8.1f},{ph_r[p]:>7.1f}")
    if len(order) > max_paths_show:
        print(f"  ... ({len(order)-max_paths_show} more paths not shown; "
              f"use --max-paths to see them)")


def scenario_summary(d, interactions, f_c):
    """Aggregate type counts and LoS coverage across all links with paths."""
    num_valid = d["num_valid_paths"]
    rx_pos, tx_pos = d["rx_pos"], d["tx_pos"]
    R, T = num_valid.shape

    type_counts = {}
    n_links_with_paths = 0
    n_links_with_los = 0
    excess_all = []
    los_ang, los_resid = [], []   # phase-convention test, LoS paths only

    dxyz = tx_pos[np.newaxis, :, :] - rx_pos[:, np.newaxis, :]
    tau_geom_ns = 1e9 * np.linalg.norm(dxyz, axis=-1) / speed_of_light

    for r in range(R):
        for t in range(T):
            k = int(num_valid[r, t])
            if k == 0:
                continue
            n_links_with_paths += 1
            link_has_los = False
            for p in range(k):
                if interactions is not None:
                    lbl, _ = classify_path(interactions[:, r, t, p])
                else:
                    is_los = abs(d["tau_ns"][r, t, p] - tau_geom_ns[r, t]) < 0.5
                    lbl = "LoS" if is_los else "?(reflect)"
                type_counts[lbl] = type_counts.get(lbl, 0) + 1
                if lbl == "LoS":
                    link_has_los = True
                    av = d["a"][r, t, p]
                    if np.isfinite(av) and av != 0 and f_c > 0:
                        tau_s = d["tau_ns"][r, t, p] * 1e-9
                        los_ang.append(np.angle(av))
                        los_resid.append(np.angle(av) + 2.0 * np.pi * f_c * tau_s)
            if link_has_los:
                n_links_with_los += 1
            excess_all.append(d["tau_ns"][r, t, :k].min() - tau_geom_ns[r, t])

    print("\n" + "=" * 100)
    print("  SCENARIO SUMMARY")
    print("=" * 100)
    print(f"  Links with surviving paths : {n_links_with_paths}")
    if n_links_with_paths:
        print(f"  Links with a LoS path      : {n_links_with_los} "
              f"({100*n_links_with_los/n_links_with_paths:.1f}%)")
    print("\n  Path count by type:")
    for lbl, c in sorted(type_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {lbl:>16} : {c}")

    if excess_all:
        ex = np.array(excess_all)
        print("\n  First-arrival excess delay over geometric LoS (ns), per link:")
        for p, q in zip([5, 50, 95], np.percentile(ex, [5, 50, 95])):
            print(f"    p{p:02d}: {q:+8.2f} ns")
        print(f"    min: {ex.min():+8.2f}   max: {ex.max():+8.2f}")

    # ── Phase-convention verdict (LoS paths only) ──
    # For a free-space LoS path, if `a` already contains exp(-j2*pi*f_c*tau) then
    # ang(a) sweeps with distance while the residual ang(a)+2*pi*f_c*tau stays
    # constant. If `a` is phase-free, ang(a) is the constant one instead.
    if len(los_ang) >= 2:
        std_ang = np.rad2deg(circ_std_rad(los_ang))
        std_res = np.rad2deg(circ_std_rad(los_resid))
        print(f"\n  Phase convention test on {len(los_ang)} LoS path(s) "
              f"(f_c = {f_c/1e9:.4f} GHz):")
        print(f"    circ-std[ ang(a) ]               : {std_ang:6.1f} deg")
        print(f"    circ-std[ ang(a) + 2pi f_c tau ] : {std_res:6.1f} deg")
        if std_ang < std_res - 30.0:
            print("    => ang(a) is the constant one: `a` is CARRIER-PHASE-FREE.")
            print("       channel_taps_gen.py correctly imposes exp(-j2pi f_c tau) itself.")
        elif std_res < std_ang - 30.0:
            print("    => residual is constant: `a` ALREADY carries exp(-j2pi f_c tau).")
            print("       channel_taps_gen.py's explicit carrier term would DOUBLE-COUNT.")
        else:
            print("    => inconclusive (need more / cleaner LoS links, or wider spread).")
    elif f_c > 0:
        print("\n  Phase convention test: skipped (fewer than 2 LoS paths).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=str, required=True)
    ap.add_argument("--project-dir", type=str, default=os.environ.get("PROJECT_DIR", "."))
    ap.add_argument("--num-links", type=int, default=5,
                    help="Number of (rx, tx) links to print in detail.")
    ap.add_argument("--max-paths", type=int, default=12,
                    help="Max paths shown per link table.")
    ap.add_argument("--npz", type=str, default=None,
                    help="Use this file instead of the newest one for --scenario.")
    args = ap.parse_args()

    npz = resolve_input(
        os.path.join(args.project_dir, "data", "path_data",
                     f"*_{args.scenario}_paths.npz"),
        args.npz, label="raw paths")
    d = np.load(npz, allow_pickle=True)

    meta = d["metadata"].item() if "metadata" in d.files else {}
    f_c = float(meta.get("frequency_hz", 0.0))
    if f_c > 0:
        print(f"f_carrier: {f_c/1e9:.4f} GHz")

    interactions = d["interactions"] if "interactions" in d.files else None
    if interactions is None:
        print("[!] No 'interactions' field in this .npz — it predates the path-type "
              "capture. Falling back to delay-only LoS detection (re-run path_gen.py "
              "to get true types).")
    else:
        codes = meta.get("interaction_type_codes")
        if codes:
            print(f"Interaction codes: {codes}")

    num_valid = d["num_valid_paths"]
    has = num_valid > 0
    valid_idx = np.argwhere(has)
    if valid_idx.size == 0:
        print("No links with surviving paths.")
        return

    n = min(args.num_links, len(valid_idx))
    spacing = max(1, len(valid_idx) // n)
    for r, t in valid_idx[::spacing][:n]:
        link_table(d, int(r), int(t), interactions, args.max_paths, f_c)

    scenario_summary(d, interactions, f_c)


if __name__ == "__main__":
    main()
