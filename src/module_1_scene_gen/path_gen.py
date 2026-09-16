# =============================================================================
# path_gen.py — Batched Ray-Tracing Path Generation
# =============================================================================
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import csv
import gc
import glob
import json
import time
import shutil
import hashlib
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

load_dotenv()

import sionna
from sionna.rt import (
    load_scene, Transmitter, Receiver, PlanarArray,
    RadioMaterial, PathSolver,
)
from configs.materials import apply_ground_materials


# ---------------------------------------------------------------------------
# CSV / materials helpers
# ---------------------------------------------------------------------------

def load_devices_from_csv(csv_path):
    """Reads TX/RX positions (and optional look_at) from a CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Device layout CSV not found at: {csv_path}")

    devices = []
    print(f"Loading devices from: {csv_path}")
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dev = {
                "name": row["name"],
                "type": row["type"].strip().lower(),
                "pos":  [float(row["x"]), float(row["y"]), float(row["z"])],
            }
            if row.get("look_at_x") and row.get("look_at_y") and row.get("look_at_z"):
                dev["look_at"] = [
                    float(row["look_at_x"]),
                    float(row["look_at_y"]),
                    float(row["look_at_z"]),
                ]
            else:
                dev["look_at"] = None
            devices.append(dev)
    return devices


def apply_materials(scene, cfg):
    """
    Apply material overrides to a loaded scene.

    Handles two cases:
      1. P.527 ground materials — assigned via itu_materials module,
         keeping the frequency callback alive.
      2. Static overrides — for scattering and thickness properties
         that Sionna's ITURadioMaterial doesn't set by default.
         These do NOT touch ε_r or σ, so the ITU frequency callback
         stays alive and materials continue to auto-update.

    Config structure (in scenario YAML):
    ------------------------------------
    material_config:
      ground:
        type: "medium_dry"              # wet | medium_dry | very_dry
        objects: ["floor", "terrain"]   # optional, auto-detects if omitted
      overrides:                        # optional static overrides
        concrete:
          scattering_coefficient: 0.3   # S ∈ [0,1], default 0.0
          xpd_coefficient: 0.0          # K_x ∈ [0,1], default 0.0
          thickness: 0.15               # slab thickness (m), default 0.1
          scattering_pattern: lambertian  # lambertian | directive | backscattering
        glass:
          scattering_coefficient: 0.05
          thickness: 0.01
    """
    if "material_config" not in cfg:
        return

    mat_cfg = cfg.material_config

    # ── 1. P.527 ground materials (frequency-aware callbacks) ──
    if "ground" in mat_cfg:
        ground = mat_cfg.ground
        ground_type = str(ground.get("type", "medium_dry"))
        ground_objects = (
            list(ground.objects) if "objects" in ground else None
        )
        apply_ground_materials(
            scene,
            ground_type=ground_type,
            ground_object_names=ground_objects,
        )

    # ── 2. Static overrides (for any scene materials) ──
    # Only the properties you list get patched; omitted ones stay at
    # whatever value Sionna's ITURadioMaterial already set.
    # These properties do not conflict with the ITU frequency callback,
    # so ε_r and σ continue to auto-update with scene.frequency.
    _PATCHABLE = {
        "scattering_coefficient",
        "xpd_coefficient",
        "thickness",
    }

    if "overrides" in mat_cfg:
        count = 0
        for key, props in mat_cfg.overrides.items():
            for mat_name, mat in scene.radio_materials.items():
                if key in mat_name.lower():
                    patched_fields = []

                    for field in _PATCHABLE:
                        if field in props:
                            setattr(mat, field, float(props[field]))
                            patched_fields.append(f"{field}={props[field]}")

                    # Scattering pattern
                    if "scattering_pattern" in props:
                        pattern_name = str(props["scattering_pattern"])
                        from sionna.rt import LambertianPattern, DirectivePattern, BackscatteringPattern
                        pattern_map = {
                            "lambertian":     LambertianPattern(),
                            "directive":      DirectivePattern(),
                            "backscattering": BackscatteringPattern(),
                        }
                        if pattern_name in pattern_map:
                            mat.scattering_pattern = pattern_map[pattern_name]
                            patched_fields.append(f"scattering_pattern={pattern_name}")
                        else:
                            print(f"  [Warning] Unknown scattering_pattern "
                                  f"'{pattern_name}' for '{mat_name}', skipping.")

                    count += 1
                    print(f"  Patched '{mat_name}': {', '.join(patched_fields)}")

        print(f"  Patched {count} materials via static overrides.")


# ---------------------------------------------------------------------------
# Checkpoint / shard helpers
# ---------------------------------------------------------------------------
# path_gen traces in batches and writes each group of batches to a shard .npz
# under data/path_data/_shards/<scenario>/ as it goes. If the job is killed
# (walltime / OOM) before finishing, the completed shards survive on disk: they
# can be merged into a (partial) dataset, and a resubmit RESUMES from the last
# shard instead of retracing. On a clean finish the shards are merged into the
# single timestamped *_paths.npz (identical layout to before) and removed.
# Tune via a `checkpoint:` block in the scenario YAML:
#   checkpoint:
#     every_batches: 25     # shard cadence (<=0 -> one shard at the very end)
#     resume: true          # reuse matching shards from a prior run
#     keep_shards: false    # keep shards even after a complete merge
# Recover/merge a dead run without retracing:  +finalize_only=true

_PATH_KEYS = ["a", "tau", "theta_r", "phi_r", "theta_t", "phi_t",
              "interactions", "valid"]
_PAD_FILL = {"a": 0j, "tau": np.nan, "interactions": 0, "valid": False}


def _merge_path_dicts(dicts, max_paths_global=None):
    """Pad a list of per-batch/per-shard path dicts on the last (path) axis to a
    common width, then concatenate along the TX axis (second-to-last).
    Returns (merged_dict, max_paths_global)."""
    if max_paths_global is None:
        max_paths_global = max(d["tau"].shape[-1] for d in dicts)
    merged = {}
    for k in dicts[0].keys():
        pieces = []
        for d in dicts:
            v = d[k]
            pad = max_paths_global - v.shape[-1]
            if pad > 0:
                pad_tuple = [(0, 0)] * (v.ndim - 1) + [(0, pad)]
                v = np.pad(v, pad_tuple, constant_values=_PAD_FILL.get(k, 0.0))
            pieces.append(v)
        tx_axis = pieces[0].ndim - 2
        merged[k] = (np.concatenate(pieces, axis=tx_axis)
                     if len(pieces) > 1 else pieces[0])
    return merged, max_paths_global


def _run_fingerprint(solver_kwargs, num_tx, num_rx, batch_size,
                     scenario_name, frequency, csv_path):
    """Short hash identifying this trace configuration. Shards only resume/merge
    when their fingerprint matches, so a config change can't silently mix data."""
    payload = {
        "solver_kwargs": dict(solver_kwargs),
        "num_tx": int(num_tx), "num_rx": int(num_rx),
        "batch_size": int(batch_size),
        "scenario_name": str(scenario_name),
        "frequency_hz": float(frequency),
        "csv": os.path.basename(csv_path),
    }
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _scan_done_batches(shard_dir, fingerprint):
    """Return the set of batch indices already covered by matching shards."""
    done = set()
    if not os.path.isdir(shard_dir):
        return done
    for f in sorted(glob.glob(os.path.join(shard_dir, "shard_b*.npz"))):
        try:
            d = np.load(f, allow_pickle=False)
            if str(d["fingerprint"]) != fingerprint:
                continue
            for bi in range(int(d["bi_start"]), int(d["bi_end"])):
                done.add(bi)
        except Exception as ex:
            print(f"[resume] ignoring unreadable shard "
                  f"{os.path.basename(f)}: {ex}")
    return done


def _write_shard(shard_dir, bi_start, bi_end, tx_start, tx_end, buffer, fingerprint):
    """Atomically write one shard (a group of consecutive batches) to disk."""
    merged, local_max = _merge_path_dicts(buffer)
    os.makedirs(shard_dir, exist_ok=True)
    name = f"shard_b{bi_start:05d}_b{bi_end:05d}.npz"
    final = os.path.join(shard_dir, name)
    tmp = os.path.join(shard_dir, "." + name + ".tmp.npz")
    np.savez(
        tmp,
        fingerprint=np.array(fingerprint),
        bi_start=bi_start, bi_end=bi_end,
        tx_start=tx_start, tx_end=tx_end,
        local_max_paths=local_max,
        **merged,
    )
    os.replace(tmp, final)   # atomic: a kill mid-write can't corrupt a shard
    return final


def _finalize_from_shards(shard_dir, fingerprint, output_dir, cfg, loc_id,
                          scene_path, frequency, solver_kwargs,
                          tx_list, rx_list, num_rx, num_tx):
    """Merge the contiguous, fingerprint-matching shards into one *_paths.npz.
    Returns (filename, complete); complete is False when the shards only cover a
    TX prefix (an interrupted run) — that file is suffixed _partialNNNNN."""
    files = sorted(glob.glob(os.path.join(shard_dir, "shard_b*.npz")))
    metas = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        if str(d["fingerprint"]) != fingerprint:
            continue
        metas.append((int(d["bi_start"]), int(d["bi_end"]), int(d["tx_start"]),
                      int(d["tx_end"]), int(d["local_max_paths"]), f))
    if not metas:
        return None, False
    metas.sort(key=lambda m: m[0])

    # Keep only the contiguous prefix starting at batch 0.
    contig, expect = [], 0
    for m in metas:
        if m[0] != expect:
            print(f"[finalize] batch gap before {m[0]} (expected {expect}); "
                  f"merging the contiguous prefix only.")
            break
        contig.append(m)
        expect = m[1]
    if not contig:
        print("[finalize] no contiguous shards starting at batch 0.")
        return None, False

    global_max = max(m[4] for m in contig)
    num_tx_covered = contig[-1][3]
    complete = num_tx_covered >= num_tx
    print(f"[finalize] merging {len(contig)} shard(s): TX 0..{num_tx_covered} "
          f"of {num_tx}, max_paths={global_max} "
          f"({'COMPLETE' if complete else 'PARTIAL'})")

    dicts = []
    for (_, _, _, _, _, f) in contig:
        d = np.load(f, allow_pickle=False)
        dicts.append({k: d[k] for k in _PATH_KEYS})
    merged, _ = _merge_path_dicts(dicts, max_paths_global=global_max)

    assert merged["tau"].shape == (num_rx, num_tx_covered, global_max), (
        f"Merged shape {merged['tau'].shape} != "
        f"({num_rx}, {num_tx_covered}, {global_max})")

    # Validity from Sionna's own mask (invalid slots have tau = -1, not NaN).
    valid = merged["valid"].astype(bool)
    tau_ns = merged["tau"] * 1e9
    num_valid_paths = valid.sum(axis=-1).astype(np.int32)
    tau_ns = np.where(valid, tau_ns, 0.0)
    for k in ("theta_r", "phi_r", "theta_t", "phi_t"):
        merged[k] = np.where(valid, merged[k], 0.0)
    merged["a"] = np.where(valid, merged["a"], 0 + 0j)
    merged["interactions"] = np.where(valid[np.newaxis], merged["interactions"], 0)

    def _pack(devs):
        names = np.array([d["name"] for d in devs])
        pos = np.array([d["pos"] for d in devs], dtype=np.float64)
        look_at = np.array(
            [d["look_at"] if d["look_at"] is not None else [np.nan]*3 for d in devs],
            dtype=np.float64,
        )
        return names, pos, look_at

    tx_used = tx_list[:num_tx_covered]
    tx_names, tx_pos, tx_look_at = _pack(tx_used)
    rx_names, rx_pos, rx_look_at = _pack(rx_list)

    metadata = {
        "scenario_name":      cfg.scenario_name,
        "selected_id":        loc_id,
        "scene_path":         scene_path,
        "frequency_hz":       frequency,
        "tx_array":           OmegaConf.to_container(cfg.tx_array, resolve=True),
        "rx_array":           OmegaConf.to_container(cfg.rx_array, resolve=True),
        "num_rx":             int(num_rx),
        "num_tx":             int(num_tx_covered),
        "num_tx_target":      int(num_tx),
        "partial":            (not complete),
        "max_paths":          int(global_max),
        "solver_kwargs":      solver_kwargs,
        "sionna_version":     getattr(sionna, "__version__", "unknown"),
        "config_snapshot":    OmegaConf.to_container(cfg, resolve=True),
        "interaction_type_codes": {
            "NONE": 0, "SPECULAR": 1, "DIFFUSE": 2,
            "REFRACTION": 4, "DIFFRACTION": 8,
        },
    }

    suffix = "" if complete else f"_partial{num_tx_covered:05d}of{num_tx:05d}"
    filename = os.path.join(
        output_dir,
        f"{time.strftime('%Y%m%d_%H%M%S')}_{cfg.scenario_name}{suffix}_paths.npz",
    )
    print(f"\nSaving to {filename}")
    np.savez(
        filename,
        a               = merged["a"],
        tau_ns          = tau_ns,
        theta_r         = merged["theta_r"],
        phi_r           = merged["phi_r"],
        theta_t         = merged["theta_t"],
        phi_t           = merged["phi_t"],
        num_valid_paths = num_valid_paths,
        interactions    = merged["interactions"],
        tx_names        = tx_names,
        tx_pos          = tx_pos,
        tx_look_at      = tx_look_at,
        rx_names        = rx_names,
        rx_pos          = rx_pos,
        rx_look_at      = rx_look_at,
        metadata        = metadata,
    )
    return filename, complete


@hydra.main(version_base=None, config_path="../../configs/scenarios")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    loc_id      = cfg.selected_id
    project_dir = cfg.project_dir
    frequency   = float(cfg.frequency)

    # ---- GPS DB + CSV routing ----
    gps_path = cfg.gps_location_path
    if os.path.isdir(gps_path):
        gps_path = os.path.join(gps_path, "gps_locations.yaml")
    gps_db = OmegaConf.load(gps_path)

    if loc_id in gps_db and "coordinate_file" in gps_db[loc_id]:
        csv_path = gps_db[loc_id].coordinate_file.replace("${project_dir}", project_dir)
    else:
        print(f"[Info] '{loc_id}' not in GPS database. Using default CSV routing.")
        yaml_dir = os.path.dirname(gps_path)
        csv_path = os.path.join(yaml_dir, f"{loc_id}_tx_rx.csv")

    # ---- Scene path ----
    if "scene_xml_path" in cfg:
        raw_path = cfg.scene_xml_path.replace("${project_dir}", project_dir)
        scene_path = os.path.abspath(raw_path)
    else:
        scene_path = os.path.join(project_dir, "data", "scenes", loc_id, "scene.xml")
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Geometry not found: {scene_path}")

    # ---- Load scene ----
    print(f"Loading scene: {scene_path}")
    
    # merge_shapes=False preserves individual mesh names so that apply_materials()
    # can do keyword-based material matching against object names.
    scene = load_scene(scene_path, merge_shapes=False)

    # Apply materials before setting frequency so that P.527 callbacks
    # are registered and will fire when scene.frequency is assigned.
    apply_materials(scene, cfg)

    scene.frequency = frequency

    # ---- Antenna arrays from config ----
    # These define the element pattern and polarization used during tracing.
    tx_arr = cfg.tx_array
    rx_arr = cfg.rx_array

    scene.tx_array = PlanarArray(
        num_rows            = int(tx_arr.num_rows),
        num_cols            = int(tx_arr.num_cols),
        vertical_spacing    = float(tx_arr.get("vertical_spacing", 0.5)),
        horizontal_spacing  = float(tx_arr.get("horizontal_spacing", 0.5)),
        pattern             = str(tx_arr.get("pattern", "iso")),
        polarization        = str(tx_arr.get("polarization", "V")),
    )
    scene.rx_array = PlanarArray(
        num_rows            = int(rx_arr.num_rows),
        num_cols            = int(rx_arr.num_cols),
        vertical_spacing    = float(rx_arr.get("vertical_spacing", 0.5)),
        horizontal_spacing  = float(rx_arr.get("horizontal_spacing", 0.5)),
        pattern             = str(rx_arr.get("pattern", "iso")),
        polarization        = str(rx_arr.get("polarization", "V")),
    )

    print(f"\n--- Antenna Arrays ---")
    print(f"TX: {tx_arr.num_rows}x{tx_arr.num_cols} "
          f"pattern={tx_arr.get('pattern','iso')} pol={tx_arr.get('polarization','V')}")
    print(f"RX: {rx_arr.num_rows}x{rx_arr.num_cols} "
          f"pattern={rx_arr.get('pattern','iso')} pol={rx_arr.get('polarization','V')}")

    # ---- Devices ----
    devices   = load_devices_from_csv(csv_path)
    tx_list   = [d for d in devices if d["type"] == "tx"]
    rx_list   = [d for d in devices if d["type"] == "rx"]
    num_tx    = len(tx_list)
    num_rx    = len(rx_list)

    print("\n--- Device Layout ---")
    print(f"TX: {num_tx}  |  RX: {num_rx}")

    # RXs are persistent — add them once
    for rx in rx_list:
        kwargs = {"name": rx["name"], "position": rx["pos"]}
        if rx["look_at"] is not None:
            kwargs["look_at"] = rx["look_at"]
        scene.add(Receiver(**kwargs))

    # ---- PathSolver kwargs from config ----
    rt_cfg = cfg.ray_tracing_params
    solver_kwargs = dict(
        max_depth           = int(rt_cfg.max_depth),
        synthetic_array     = bool(rt_cfg.synthetic_array),
        los                 = bool(rt_cfg.get("los", True)),
        specular_reflection = bool(rt_cfg.get("specular_reflection", True)),
        diffuse_reflection  = bool(rt_cfg.get("diffuse_reflection", False)),
        refraction          = bool(rt_cfg.get("refraction", False)),
        diffraction         = bool(rt_cfg.get("diffraction", False)),
        edge_diffraction    = bool(rt_cfg.get("edge_diffraction", False)),
        diffraction_lit_region = bool(rt_cfg.get("diffraction_lit_region", True)),
    )
    if "samples_per_src" in rt_cfg:
        solver_kwargs["samples_per_src"] = int(rt_cfg.samples_per_src)
    if "max_num_paths_per_src" in rt_cfg:
        solver_kwargs["max_num_paths_per_src"] = int(rt_cfg.max_num_paths_per_src)
    if "seed" in rt_cfg:
        solver_kwargs["seed"] = int(rt_cfg.seed)

    # ---- Batched tracing ----
    batch_size  = min(int(rt_cfg.get("batch_size", 50)), num_tx)
    num_batches = int(np.ceil(num_tx / batch_size))
    print(f"\nTracing {num_tx} TXs in {num_batches} batches "
          f"(batch_size={batch_size}, max_depth={solver_kwargs['max_depth']})")

    # ---- Output + checkpoint setup ----
    output_dir = os.path.join(project_dir, "data", "path_data")
    os.makedirs(output_dir, exist_ok=True)
    shard_dir = os.path.join(output_dir, "_shards", str(cfg.scenario_name))

    ckpt_cfg      = cfg.get("checkpoint", {}) or {}
    ckpt_every    = int(ckpt_cfg.get("every_batches", 25))
    resume        = bool(ckpt_cfg.get("resume", True))
    keep_shards   = bool(ckpt_cfg.get("keep_shards", False))
    finalize_only = bool(cfg.get("finalize_only", False))
    if ckpt_every <= 0:
        ckpt_every = num_batches  # single shard flushed at the very end

    fingerprint = _run_fingerprint(solver_kwargs, num_tx, num_rx, batch_size,
                                   cfg.scenario_name, frequency, csv_path)
    print(f"Checkpoint shards -> {shard_dir}  "
          f"(every {ckpt_every} batch(es), fingerprint {fingerprint})")

    if not finalize_only:
        # ---- Resume: skip batches a prior run already checkpointed ----
        done_batches = _scan_done_batches(shard_dir, fingerprint) if resume else set()
        if done_batches:
            print(f"[resume] {len(done_batches)} batch(es) already on disk — skipping them.")

        # ---- Batched tracing with shard checkpointing ----
        # Each shard is written atomically as it completes, so a walltime/OOM
        # kill leaves the finished batches recoverable on disk (and a resubmit
        # resumes from here instead of retracing).
        buffer, buf_bi_start, buf_bi_end = [], None, None
        for bi in range(num_batches):
            s = bi * batch_size
            e = min(s + batch_size, num_tx)
            if bi in done_batches:
                print(f"  -> Batch {bi+1}/{num_batches}: TX {s+1}..{e}  [checkpointed, skip]")
                continue
            chunk = tx_list[s:e]
            print(f"  -> Batch {bi+1}/{num_batches}: TX {s+1}..{e}")

            # Add this batch's TXs
            for tx in chunk:
                kwargs = {"name": tx["name"], "position": tx["pos"]}
                if tx["look_at"] is not None:
                    kwargs["look_at"] = tx["look_at"]
                scene.add(Transmitter(**kwargs))

            solver = PathSolver()
            paths  = solver(scene=scene, **solver_kwargs)

            # Complex amplitudes: Sionna returns (real, imag) tuple
            # Shape: (num_rx, num_rx_ant, num_tx_batch, num_tx_ant, num_paths)
            a_re = paths.a[0].numpy()
            a_im = paths.a[1].numpy()
            assert a_re.shape[1] == 1 and a_re.shape[3] == 1, (
                f"Expected single-antenna iso trace, got shape {a_re.shape}. "
                "This script assumes num_rx_ant == num_tx_ant == 1."
            )
            a_complex = (a_re + 1j * a_im)[:, 0, :, 0, :]  # (num_rx, num_tx_batch, num_paths)

            # Per-bounce interaction type, for path classification (LoS / specular /
            # diffuse / refraction / diffraction). InteractionType codes:
            #   NONE=0, SPECULAR=1, DIFFUSE=2, REFRACTION=4, DIFFRACTION=8.
            # Shape under synthetic_array=True: (max_depth, num_rx, num_tx_batch, num_paths).
            # Older builds may keep antenna singletons -> collapse them if present.
            inter = np.asarray(paths.interactions.numpy())
            if inter.ndim == 6:                       # (D, R, A_r, T, A_t, P)
                inter = inter[:, :, 0, :, 0, :]
            inter = inter.astype(np.int16)            # small codes; -> (D, R, T_b, P)

            # Validity mask. Sionna RT marks invalid path slots with tau = -1 (not
            # NaN), so validity MUST come from paths.valid, never from isnan(tau).
            valid_mask = np.asarray(paths.valid.numpy())
            if valid_mask.ndim == 5:                  # (R, A_r, T, A_t, P)
                valid_mask = valid_mask[:, 0, :, 0, :]
            valid_mask = valid_mask.astype(bool)      # -> (R, T_b, P)

            # tau, angles with synthetic_array=True: (num_rx, num_tx_batch, num_paths)
            if buf_bi_start is None:
                buf_bi_start = bi
            buffer.append({
                "a":            a_complex,
                "tau":          paths.tau.numpy(),
                "theta_r":      paths.theta_r.numpy(),
                "phi_r":        paths.phi_r.numpy(),
                "theta_t":      paths.theta_t.numpy(),
                "phi_t":        paths.phi_t.numpy(),
                "interactions": inter,
                "valid":        valid_mask,
            })
            buf_bi_end = bi + 1

            # Remove this batch's TXs before adding the next
            for tx in chunk:
                scene.remove(tx["name"])

            del paths, solver
            gc.collect()

            # Flush a shard once enough new batches are buffered
            if len(buffer) >= ckpt_every:
                _write_shard(shard_dir, buf_bi_start, buf_bi_end,
                             buf_bi_start * batch_size,
                             min(buf_bi_end * batch_size, num_tx),
                             buffer, fingerprint)
                print(f"     [checkpoint] shard batches {buf_bi_start+1}..{buf_bi_end} written")
                buffer, buf_bi_start, buf_bi_end = [], None, None
                gc.collect()

        # Flush the trailing partial shard
        if buffer:
            _write_shard(shard_dir, buf_bi_start, buf_bi_end,
                         buf_bi_start * batch_size,
                         min(buf_bi_end * batch_size, num_tx),
                         buffer, fingerprint)
            print(f"     [checkpoint] shard batches {buf_bi_start+1}..{buf_bi_end} written")
            buffer = []
            gc.collect()
    else:
        print("[finalize-only] skipping tracing; merging existing shards.")

    # ---- Merge checkpoint shards into the final dataset ----
    print("\nMerging checkpoint shards...")
    filename, complete = _finalize_from_shards(
        shard_dir, fingerprint, output_dir, cfg, loc_id, scene_path,
        frequency, solver_kwargs, tx_list, rx_list, num_rx, num_tx,
    )
    if filename is None:
        print("[finalize] no matching shards found — nothing to save.")
        return

    if complete and not keep_shards:
        shutil.rmtree(shard_dir, ignore_errors=True)
        print(f"[finalize] complete — removed checkpoint shards in {shard_dir}")
    elif not complete:
        print(f"[finalize] PARTIAL dataset saved; shards kept in {shard_dir} "
              f"so a resubmit can resume tracing.")

    print("SUCCESS")


if __name__ == "__main__":
    main()