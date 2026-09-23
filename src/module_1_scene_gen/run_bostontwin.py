# =============================================================================
# run_bostontwin.py — Hydra wrapper around BostonTwin's scene generator
# =============================================================================
#
# Uses the BostonTwin class from third_party/boston_twin directly:
#   bt = BostonTwin(dataset_dir)
#   bt.generate_scene_from_radius(scene_name, center_lon, center_lat, side_m)
#
# Defers all scene assembly to BostonTwin's own pipeline and keeps its native
# output layout (the older "unwrapped" variant — which read the GeoJSON and
# PLYs directly to build scene.xml from scratch — has been removed):
#
#   data/scenes/<scenario_id>/
#     scene.xml                               Mitsuba XML with relative mesh paths
#                                             (renamed from BostonTwin's
#                                             <scenario_id>.xml so downstream
#                                             stages find it without each
#                                             scenario YAML setting scene_xml_path)
#     <scenario_id>.geojson                   Selected buildings (kept under the
#                                             scene-specific name so
#                                             extract_building_footprints.py
#                                             can locate it by selected_id)
#     <scenario_id>_tileinfo.geojson          Scene center / bounds metadata
#     <scenario_id>_Frame.ply                 Per-scene ground frame PLY, MOVED
#                                             out of <BOSTONTWIN_ROOT>/boston3d/
#                                             meshes/ into the scene folder so
#                                             that shared dir doesn't grow by
#                                             one PLY per scene generated. The
#                                             scene.xml's ground shape is
#                                             rewritten to reference this
#                                             local copy by basename.
#
# BostonTwin's native XML references each mesh as "meshes/<id>.ply" — relative
# to boston3d/, where BostonTwin originally writes the XML next to the shared
# meshes/ folder (see obj_utils.py:49/102/131, "ply_path.relative_to
# (ply_path_relative)" with ply_path_relative = boston3d/). When we copy the
# XML into data/scenes/<scenario_id>/, those relative paths no longer resolve.
# To keep the XML portable (machine-independent — no absolute paths baked in,
# no per-scenario duplication of the ~GB meshes library), we rewrite each
# "value=\"meshes/..\"" in the XML to a path relative to the XML's directory
# that points at the shared boston3d/meshes/ folder, e.g.
# "value=\"../../bostontwin/boston3d/meshes/<id>.ply\"". As long as the laptop
# mirrors the same scenes/ ↔ bostontwin/ relative layout, the XML loads as-is.
#
# Per-building translation correction:
# BostonTwin's generate_scene_from_model_gdf translates each PLY by the
# polygon centroid computed via geopandas (`model_gdf.centroid`). The PLY
# meshes themselves, however, were originally authored against the
# `Centr_X_Ft` / `Centr_Y_Ft` metadata fields on each building (see
# obj_utils.obj2ply_mi + BostonModelDownloader.generate_dataset). For ~10%
# of buildings (irregular footprints) those two anchors disagree by 4–66 m,
# producing visibly misplaced structures. We post-process the generated XML:
# for every building shape (`id` starting with "BOS-"), we replace the
# translation columns of its `<matrix>` with
#     tx = Centr_X_Ft * 0.3048 - scene_center_x_m
#     ty = Centr_Y_Ft * 0.3048 - scene_center_y_m
# which matches the translations baked into BostonTwin's pre-built tile XMLs
# (e.g. BOS_H_4.xml) to sub-millimetre precision. See
# tools/check_bos_h_4_residuals.py for the validating diagnostic.
#
# CONFIG:
#   gps_locations.yaml entry (selects WHERE in the world the scene is):
#     <selected_id>:
#       description: "..."           # optional human-readable label
#       center_lat: 42.36042393
#       center_lon: -71.06096354
#       side_m: 250.0                # half-side: scene query bbox is 2*side_m
#       coordinate_file: ...         # consumed by downstream pipeline stages
#
#   scenario YAML (e.g. boston_cambridge_street_test.yaml) — per-experiment knobs only:
#     force: false               # top-level; overwrite an existing data/scenes/<id>/
#                                # (shared with run_g2sm.py)
#
#   Dataset root: BOSTONTWIN_ROOT env var (from .env). The wrapper also
#   accepts a `bostontwin_data_root` field on the gps entry as a fallback,
#   but the env var is the recommended path.
#
# Side effect: BostonTwin writes the new scene files into <data_root>/boston3d/.
# We then copy the three sidecar files to data/scenes/<scenario_id>/. The
# originals are left in boston3d/ (BostonTwin re-uses them on the next load).
# =============================================================================

import os
# Quiet TF/CUDA chatter from Sionna's import (BostonTwin.py imports sionna.rt).
# Set before any TF import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import hydra
import numpy.typing  # noqa: F401 — exposes np.typing for older numpy versions
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

# BostonTwin's local CRS is in feet; PLYs were authored against
# (Centr_X_Ft, Centr_Y_Ft) scaled to meters with this factor (see
# obj_utils.obj2ply_mi + BostonModelDownloader.generate_dataset).
_FT2M = 0.3048

load_dotenv()

# BostonTwin's package uses relative imports rooted at its own `src/` —
# add the package root to sys.path so `from src.classes.BostonTwin import ...`
# resolves to third_party/boston_twin/src/.
_BOSTONTWIN_PKG = (
    Path(__file__).resolve().parents[2]
    / "third_party" / "boston_twin"
)
if str(_BOSTONTWIN_PKG) not in sys.path:
    sys.path.insert(0, str(_BOSTONTWIN_PKG))

from src.classes.BostonTwin import BostonTwin  # noqa: E402


def _load_centroids_ft(geojson_path: Path) -> dict:
    """Return {Model_ID: (Centr_X_Ft, Centr_Y_Ft)} from a BostonTwin geojson.
    These are the building-anchor metadata fields the PLY meshes were
    originally centered against. Entries without both fields are skipped."""
    with open(geojson_path) as f:
        gj = json.load(f)
    out = {}
    for feat in gj.get("features", []):
        p = feat.get("properties", {})
        mid = p.get("Model_ID")
        cx = p.get("Centr_X_Ft")
        cy = p.get("Centr_Y_Ft")
        if mid is None or cx is None or cy is None:
            continue
        out[mid] = (float(cx), float(cy))
    return out


def _load_scene_center_m(tileinfo_path: Path) -> tuple:
    """Return (Centr_X_m, Centr_Y_m) from a BostonTwin _tileinfo.geojson."""
    with open(tileinfo_path) as f:
        gj = json.load(f)
    props = gj["features"][0]["properties"]
    return float(props["Centr_X_m"]), float(props["Centr_Y_m"])


def _rewrite_xml(src_xml: Path, rel_meshes: str,
                 centroids_ft: dict, scene_center_m: tuple,
                 ground_frame_name: str) -> tuple:
    """Parse the XML BostonTwin wrote and apply three corrections in one pass:

      1. Rewrite every `<string name="filename" value="meshes/X.ply"/>` to
         use `rel_meshes/X.ply` (so the XML doesn't depend on a sibling
         `meshes/` folder).
      2. Special-case the ground shape (id="ground") — rewrite its filename
         to the bare basename `ground_frame_name`, since we move the per-
         scene `_Frame.ply` out of boston3d/meshes/ and into the scene
         folder so it doesn't accumulate there across many scenes.
      3. Replace each building shape's `<matrix>` translation columns (tx, ty)
         with the values implied by the PLY's authoring anchor:
            tx = Centr_X_Ft * 0.3048 - scene_center_x_m
            ty = Centr_Y_Ft * 0.3048 - scene_center_y_m
         This corrects ~10% of buildings whose PLY anchor doesn't match the
         polygon centroid that `generate_scene_from_model_gdf` translates by.
         See tools/check_bos_h_4_residuals.py for the diagnostic that
         confirmed this against BOS_H_4.xml's pre-built translations.

    Returns (xml_bytes, n_paths_rewritten, n_translations_corrected,
             max_delta_m, n_skipped_no_metadata).
    """
    tree = ET.parse(src_xml)
    root = tree.getroot()
    n_paths = 0
    n_corrected = 0
    n_skipped_no_metadata = 0
    max_delta = 0.0
    mesh_prefix = f"{rel_meshes}/"

    for shape in root.iter("shape"):
        if shape.get("type") != "ply":
            continue

        sid = shape.get("id", "")
        is_ground = (sid == "ground")

        # 1+2) Rewrite the meshes/X.ply filename reference. The ground shape
        # gets a bare basename (local file in the scene dir); everything else
        # gets the relative-path prefix into boston3d/meshes/.
        for s in shape.iter("string"):
            if s.get("name") != "filename":
                continue
            val = s.get("value", "")
            if val.startswith("meshes/"):
                if is_ground:
                    s.set("value", ground_frame_name)
                else:
                    s.set("value", mesh_prefix + val[len("meshes/"):])
                n_paths += 1

        # 3) Correct the translation for building shapes only (skip ground/frame).
        if not sid.startswith("BOS-"):
            continue
        if sid not in centroids_ft:
            n_skipped_no_metadata += 1
            continue
        cx_ft, cy_ft = centroids_ft[sid]
        new_tx = cx_ft * _FT2M - scene_center_m[0]
        new_ty = cy_ft * _FT2M - scene_center_m[1]

        mat = shape.find(".//matrix")
        if mat is None:
            continue
        vals = mat.get("value", "").split()
        if len(vals) < 12:
            continue
        old_tx = float(vals[3])
        old_ty = float(vals[7])
        delta = max(abs(new_tx - old_tx), abs(new_ty - old_ty))
        if delta > max_delta:
            max_delta = delta
        vals[3] = f"{new_tx:.6f}"
        vals[7] = f"{new_ty:.6f}"
        mat.set("value", " ".join(vals))
        n_corrected += 1

    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return xml_bytes, n_paths, n_corrected, max_delta, n_skipped_no_metadata


def _copy_native_artifacts(boston3d_dir: Path, scene_name: str,
                           target_dir: Path) -> tuple:
    """Copy BostonTwin's outputs into target_dir, with three XML fixups:

    - mesh refs `meshes/X.ply` → `<rel>/meshes/X.ply` (relative to target_dir);
    - per-building translations recomputed from `Centr_X_Ft`/`Centr_Y_Ft`
      so PLYs land on the polygon footprints they were authored against
      (see _rewrite_xml);
    - the per-scene ground frame PLY (`<scene>_Frame.ply`) is MOVED out of
      the shared `boston3d/meshes/` folder into the scene folder, and the
      ground shape's filename reference is rewritten to that local copy.
      This keeps `boston3d/meshes/` from growing by one PLY per scene
      generated.

    Returns (copied_names, rel_meshes, stats) where stats is a dict with
    'paths_rewritten', 'translations_corrected', 'max_delta_m', 'no_metadata'.
    """
    # BostonTwin writes files named after the scene; we copy the XML under
    # the generic name `scene.xml` so downstream stages (path_gen.py default,
    # any other consumer) can find it without each scenario YAML having to
    # set `scene_xml_path`. Sidecar geojsons keep BostonTwin's source names
    # because extract_building_footprints.py looks them up by selected_id.
    src_xml_name = f"{scene_name}.xml"
    geojson_name = f"{scene_name}.geojson"
    tileinfo_name = f"{scene_name}_tileinfo.geojson"
    frame_name = f"{scene_name}_Frame.ply"
    dst_xml_name = "scene.xml"

    src_xml = boston3d_dir / src_xml_name
    src_geojson = boston3d_dir / geojson_name
    src_tileinfo = boston3d_dir / tileinfo_name
    src_frame = boston3d_dir / "meshes" / frame_name
    for p in (src_xml, src_geojson, src_tileinfo, src_frame):
        if not p.is_file():
            raise FileNotFoundError(
                f"BostonTwin did not produce expected file: {p}"
            )

    # Relative path from the XML's directory to boston3d/meshes/. Both sides
    # are resolved first so that any symlinks in the chain (e.g. project-side
    # `data/` → `/ocean/.../data/`) don't break the common-ancestor lookup
    # and produce a hideous `../../../../...` walk. After resolution both live
    # under the shared data root, so the result is the clean project-side
    # relative form, e.g. "../../bostontwin/boston3d/meshes".
    rel_meshes = os.path.relpath(
        (boston3d_dir / "meshes").resolve(),
        start=target_dir.resolve(),
    )

    centroids_ft = _load_centroids_ft(src_geojson)
    scene_center_m = _load_scene_center_m(src_tileinfo)

    xml_bytes, n_paths, n_corrected, max_delta, n_no_meta = _rewrite_xml(
        src_xml, rel_meshes, centroids_ft, scene_center_m, frame_name,
    )
    if n_paths == 0:
        raise RuntimeError(
            f"Unexpected XML format in {src_xml}: no 'meshes/...' references "
            f"found to rewrite."
        )

    # Move the per-scene Frame PLY out of boston3d/meshes/ and into the
    # scene folder. The XML's ground shape was already rewritten above to
    # reference this local copy as a bare basename.
    shutil.move(str(src_frame), str(target_dir / frame_name))

    (target_dir / dst_xml_name).write_bytes(xml_bytes)
    shutil.copy(src_geojson, target_dir / geojson_name)
    shutil.copy(src_tileinfo, target_dir / tileinfo_name)

    stats = {
        "paths_rewritten": n_paths,
        "translations_corrected": n_corrected,
        "max_delta_m": max_delta,
        "no_metadata": n_no_meta,
    }
    return ([dst_xml_name, geojson_name, tileinfo_name, frame_name],
            rel_meshes, stats)


@hydra.main(version_base=None, config_path="../../configs/scenarios")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    location_path = cfg.gps_location_path
    selected_id = cfg.selected_id
    output_root = cfg.output_root

    print(f"Project Root: {cfg.project_dir}")
    print(f"Loading location from: {location_path}")

    gps_location = OmegaConf.load(location_path)
    if selected_id not in gps_location:
        raise KeyError(
            f"Location '{selected_id}' not found in {location_path}"
        )
    loc_data = gps_location[selected_id]

    # Location (where in the world) — from gps_locations.yaml entry.
    for key in ("center_lat", "center_lon", "side_m"):
        if key not in loc_data:
            raise KeyError(
                f"gps entry '{selected_id}' is missing '{key}'. "
                f"BostonTwin-backed entries need center_lat / center_lon / "
                f"side_m in {location_path}."
            )
    center_lat = float(loc_data.center_lat)
    center_lon = float(loc_data.center_lon)
    side_m = float(loc_data.side_m)

    # Per-experiment knobs — top-level `force:` is shared with other stage-1
    # generators (run_g2sm.py). Defaults to false (skip if scene exists).
    force = bool(cfg.get("force", False))

    data_root = str(os.environ.get("BOSTONTWIN_ROOT", "")
                    or loc_data.get("bostontwin_data_root", ""))
    if not data_root:
        raise ValueError(
            "Set BOSTONTWIN_ROOT in your .env (or 'bostontwin_data_root' in "
            "the gps entry) so the wrapper can locate the BostonTwin dataset."
        )
    data_root = Path(data_root)
    boston3d_dir = data_root / "boston3d"
    if not boston3d_dir.is_dir():
        raise FileNotFoundError(
            f"BostonTwin 'boston3d' directory not found under: {data_root}"
        )

    target_dir = Path(output_root) / selected_id
    if target_dir.exists() and any(target_dir.iterdir()):
        if not force:
            print(f"[Skip] Geometry for '{selected_id}' already exists in: "
                  f"{target_dir}  (set force=true to overwrite)")
            return
        print(f"[Force] Removing existing {target_dir}")
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Generating Geometry for: {selected_id} "
          f"(BostonTwin wrapper, side_m={side_m:.1f}) ---")

    # BostonTwin only writes <scene>.geojson if the file doesn't already exist
    # (BostonModel.py:140: `if not gdf_out_path.is_file()`). It DOES always
    # overwrite the .xml and _tileinfo.geojson. Re-running with a different
    # center/side leaves a stale geojson that mismatches the new XML, which
    # silently breaks our Centr_X_Ft/Centr_Y_Ft translation correction. Same
    # concern for the Frame PLY (sized to the previous scene_size). Wipe these
    # so BostonTwin regenerates a consistent set.
    for stale in (
        boston3d_dir / f"{selected_id}.xml",
        boston3d_dir / f"{selected_id}.geojson",
        boston3d_dir / f"{selected_id}_tileinfo.geojson",
        boston3d_dir / "meshes" / f"{selected_id}_Frame.ply",
    ):
        if stale.is_file():
            stale.unlink()

    bt = BostonTwin(dataset_dir=data_root)
    try:
        bt.generate_scene_from_radius(
            scene_name=selected_id,
            center_lon=center_lon,
            center_lat=center_lat,
            side_m=side_m,
            load=False,
        )
    except Exception as e:
        print(f"[Error] BostonTwin.generate_scene_from_radius failed: {e}")
        if target_dir.exists() and not any(target_dir.iterdir()):
            target_dir.rmdir()
        raise

    copied, rel_meshes, stats = _copy_native_artifacts(
        boston3d_dir, selected_id, target_dir,
    )
    print(f"[Success] Copied native artifacts to {target_dir}:")
    for name in copied:
        print(f"   - {name}")
    print(f"   XML mesh refs rewritten to: {rel_meshes}/  "
          f"({stats['paths_rewritten']} refs)")
    print(f"   PLY translations recomputed from Centr_X_Ft/Centr_Y_Ft: "
          f"{stats['translations_corrected']} buildings "
          f"(max delta from polygon-centroid: {stats['max_delta_m']:.2f} m"
          f"; {stats['no_metadata']} skipped for missing metadata)")


if __name__ == "__main__":
    main()
