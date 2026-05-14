"""
ITU Material Properties Data

This module provides predefined material properties based on
ITU-R Recommendation P.2040-2: "[Effects of building materials and structures
on radiowave propagation above about 100 MHz](https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.2040-3-202308-I!!PDF-E.pdf)".

P.527-3 ground material callbacks use conductivity/permittivity curves
digitized from the ITU-R P.527-3 PDF, stored as CSVs under the
configs.materials.data package (originally from scene_generation.data).
These cover 10 kHz – 300 GHz and are
interpolated via smoothing splines at runtime.

TODO: Verify the exact material names using by Mitsuba and Sionna. Especially how they named the low/high frequency glass and ceiling board materials.
TODO: Check what is the exact colors using by Mitsuba/Sionna/Blender/Blosm for the materials.
"""

ITU_MATERIALS = {
    "mat-vacuum": {
        "name": "Vacuum (\u2248Air)",
        "lower_freq_limit": 0.001e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color": (0.1216, 0.4667, 0.7059),
    },
    "mat-itu_concrete": {
        "name": "Concrete",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color": (0.539479, 0.539479, 0.539480),
    },
    "mat-itu_brick": {
        "name": "Brick",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 40e9,
        "mitsuba_color":(1.0000, 0.4980, 0.0549),
    },
    "mat-itu_plasterboard": {
        "name": "Plasterboard",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color":(1.0000, 0.7333, 0.4706),
    },
    "mat-itu_wood": {
        "name": "Wood",
        "lower_freq_limit": 0.001e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color": (0.043, 0.58, 0.184)
    },
    "mat-itu_glass": {
        "name": "Glass",
        "lower_freq_limit": [0.1e9,220e9],
        "upper_freq_limit": [100e9,450e9],
        "mitsuba_color":(0.5961, 0.8745, 0.5412),
    },
    "mat-itu_ceiling_board": {
        "name": "Ceiling Board",
        "lower_freq_limit": [1e9, 220e9],
        "upper_freq_limit": [100e9, 450e9],
        "mitsuba_color":(1.0000, 0.5961, 0.5882),
    },
    "mat-itu_chipboard": {
        "name": "Chipboard",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color":(0.7725, 0.6902, 0.8353),
    },
    "mat-itu_plywood": {
        "name": "Plywood",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 40e9,
        "mitsuba_color":(0.5490, 0.3373, 0.2941),
    },
    "mat-itu_marble": {
        "name": "Marble",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 60e9,
        "mitsuba_color": (0.701101, 0.644479, 0.485150),
    },
    "mat-itu_floorboard": {
        "name": "Floorboard",
        "lower_freq_limit": 50e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color":(0.8902, 0.4667, 0.7608),
    },
    "mat-itu_metal": {
        "name": "Metal",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color": (0.219526, 0.219526, 0.254152)
    },
    "mat-itu_very_dry_ground": {
        "name": "Very Dry Ground",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 10e9,
        "mitsuba_color": (0.4980, 0.4980, 0.4980),
    },
    "mat-itu_medium_dry_ground": {
        "name": "Medium Dry Ground",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 10e9,
        "mitsuba_color":(0.7804, 0.7804, 0.7804),
    },
    "mat-itu_wet_ground": {
        "name": "Wet Ground",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 10e9,
        "mitsuba_color": (0.91, 0.569, 0.055)
    },
    "mat-very_dry_ground_P.527": {
        "name": "Very Dry Ground (Extended Range, ITU P.527-3)",
        "lower_freq_limit": 1e4,
        "upper_freq_limit": 3e11,
        "mitsuba_color": (0.4980, 0.4980, 0.4980),
    },
    "mat-medium_dry_ground_P.527": {
        "name": "Medium Dry Ground (Extended Range, ITU P.527-3)",
        "lower_freq_limit": 1e4,
        "upper_freq_limit": 3e11,
        "mitsuba_color":(0.7804, 0.7804, 0.7804),
    },
    "mat-wet_ground_P.527": {
        "name": "Wet Ground (Extended Range, ITU P.527-3)",
        "lower_freq_limit": 1e4,
        "upper_freq_limit": 3e11,
        "mitsuba_color": (0.91, 0.569, 0.055)
    },
}


# ─────────────────────────────────────────────────────────────
#  Imports
#
#  NOTE: The P.527 CSV data files were originally stored under
#  the collaborators' package "scene_generation.data". They have
#  been relocated to "configs.materials.data" to keep all
#  material-related resources together. If syncing with the
#  original scene_generation repo, copy the CSV files from
#  scene_generation/data/ into configs/materials/data/.
# ─────────────────────────────────────────────────────────────

import pandas as pd
import configs.materials.data

from functools import lru_cache
from importlib_resources import files
from scipy.interpolate import make_smoothing_spline


# ─────────────────────────────────────────────────────────────
#  P.527 spline loading — cached version of the original
#  CSV-reading + spline-fitting logic.
#
#  CSV files live in the configs.materials.data package
#  (originally from collaborators' scene_generation.data).
#  File naming convention:
#    {prefix}_con.csv  → conductivity vs frequency (MHz)
#    {prefix}_per.csv  → relative permittivity vs frequency (MHz)
#  Each CSV has two columns (no header): frequency_mhz, value
# ─────────────────────────────────────────────────────────────

_P527_CSV_FILES = {
    "wet": {
        "con": "B_wet_ground_con.csv",
        "per": "B_wet_ground_per.csv",
    },
    "medium_dry": {
        "con": "D_medium_dry_ground_con.csv",
        "per": "D_medium_dry_ground_per.csv",
    },
    "very_dry": {
        "con": "E_very_dry_ground_con.csv",
        "per": "E_very_dry_ground_per.csv",
    },
}


@lru_cache(maxsize=None)
def _load_p527_splines(soil_type):
    """
    Load conductivity/permittivity CSVs from configs.materials.data and
    fit smoothing splines. Called once per soil type, then cached.

    This performs the exact same operations as the original per-callback
    code, just factored out so the CSV read + spline fit happens once.

    Parameters
    ----------
    soil_type : str
        One of "wet", "medium_dry", "very_dry".

    Returns
    -------
    (spl_per, spl_con) : tuple of smoothing spline callables
        spl_per(f_mhz) -> relative permittivity
        spl_con(f_mhz) -> conductivity  [S/m]
    """
    csv_files = _P527_CSV_FILES[soil_type]

    # Read conductivity CSV — same as original
    with files('configs.materials.data').joinpath(csv_files["con"]).open("r") as f:
        df_con = pd.read_csv(f, header=None, names=["x", "y"])
    df_con = df_con.sort_values(by="x")

    # Read permittivity CSV — same as original
    with files('configs.materials.data').joinpath(csv_files["per"]).open("r") as f:
        df_per = pd.read_csv(f, header=None, names=["x", "y"])
    df_per = df_per.sort_values(by="x")

    # Fit smoothing splines — same as original
    spl_con = make_smoothing_spline(df_con["x"], df_con["y"])
    spl_per = make_smoothing_spline(df_per["x"], df_per["y"])

    return spl_per, spl_con


# ─────────────────────────────────────────────────────────────
#  P.527 frequency_update_callbacks
#
#  These have the exact same signature and return values as the
#  original callbacks. The only difference is that the CSV loading
#  and spline fitting is cached via _load_p527_splines() instead
#  of being repeated on every call.
# ─────────────────────────────────────────────────────────────

def wet_ground_ITU_P527_callback(f_hz):
    f_mhz = f_hz / 1e6
    if f_mhz < 1e-2 or f_mhz > 3e5:
        return (-1.0, -1.0)

    spl_per, spl_con = _load_p527_splines("wet")

    relative_permittivity = spl_per(f_mhz)
    conductivity = spl_con(f_mhz)

    return (relative_permittivity.item(), conductivity.item())


def medium_dry_ground_ITU_P527_callback(f_hz):
    f_mhz = f_hz / 1e6
    if f_mhz < 1e-2 or f_mhz > 3e5:
        return (-1.0, -1.0)

    spl_per, spl_con = _load_p527_splines("medium_dry")

    relative_permittivity = spl_per(f_mhz)
    conductivity = spl_con(f_mhz)

    return (relative_permittivity.item(), conductivity.item())


def very_dry_ground_ITU_P527_callback(f_hz):
    f_mhz = f_hz / 1e6
    if f_mhz < 1e-2 or f_mhz > 3e5:
        return (-1.0, -1.0)

    spl_per, spl_con = _load_p527_splines("very_dry")

    relative_permittivity = spl_per(f_mhz)
    conductivity = spl_con(f_mhz)

    return (relative_permittivity.item(), conductivity.item())


# ─────────────────────────────────────────────────────────────
#  get_material() — same interface as the original
#
#  Returns a Sionna RadioMaterial with the appropriate P.527
#  frequency_update_callback. Uses the same callback function
#  names so any code that referenced them directly still works.
# ─────────────────────────────────────────────────────────────

def get_material(material_name):
    from sionna.rt import RadioMaterial
    material_dict = {
        "mat-wet_ground_P.527":RadioMaterial("wet_ground_P.527_impl",
                                                 frequency_update_callback=wet_ground_ITU_P527_callback),
        "mat-medium_dry_ground_P.527":RadioMaterial("medium_dry_ground_P.527_impl",
                                                        frequency_update_callback=medium_dry_ground_ITU_P527_callback),
        "mat-very_dry_ground_P.527":RadioMaterial("very_dry_ground_P.527_impl",
                                                        frequency_update_callback=very_dry_ground_ITU_P527_callback),
    }
    if material_name not in material_dict.keys():
        return None
    else:
        return material_dict[material_name]


# ─────────────────────────────────────────────────────────────
#  apply_ground_materials() — NEW addition for pipeline integration
#
#  This function is NOT in the original collaborators' script.
#  It provides a clean API for path_gen.py to assign P.527
#  ground materials to scene objects, driven by the scenario
#  YAML config.
# ─────────────────────────────────────────────────────────────

_GROUND_TYPE_ALIASES = {
    "wet":        "mat-wet_ground_P.527",
    "medium_dry": "mat-medium_dry_ground_P.527",
    "very_dry":   "mat-very_dry_ground_P.527",
}


def apply_ground_materials(scene, ground_type="medium_dry",
                           ground_object_names=None):
    """
    Assign a P.527 ground material to floor/ground objects in a scene.

    Parameters
    ----------
    scene : sionna.rt.Scene
        Loaded Sionna scene.
    ground_type : str
        One of "wet", "medium_dry", "very_dry".
    ground_object_names : list[str] or None
        Names of scene objects to assign the ground material to.
        If None, searches for objects whose name contains "floor"
        or "ground" (case-insensitive).

    Returns
    -------
    list[str]
        Names of objects that were assigned the new material.
    """
    mat_key = _GROUND_TYPE_ALIASES.get(ground_type)
    if mat_key is None:
        raise ValueError(
            f"Unknown ground_type '{ground_type}'. "
            f"Use one of: {list(_GROUND_TYPE_ALIASES.keys())}"
        )

    mat = get_material(mat_key)
    if mat is None:
        raise RuntimeError(f"Failed to create material for {mat_key}")

    # ── Neutralize old P.2040 ground materials in the registry ──
    # The scene XML may have assigned a P.2040 ground material
    # (e.g. itu_wet_ground) which is only valid up to 10 GHz.
    # Even after we reassign the ground object to a P.527 material,
    # the old material still lives in scene.radio_materials and its
    # frequency_update() callback fires when scene.frequency is set.
    # If the operating frequency exceeds 10 GHz, this throws a
    # ValueError. We replace the old callback with the P.527 one
    # so it returns valid values at any frequency.
    _OLD_GROUND_KEYWORDS = ["wet_ground", "dry_ground", "medium_dry"]
    for mat_name, old_mat in scene.radio_materials.items():
        if any(kw in mat_name.lower() for kw in _OLD_GROUND_KEYWORDS):
            old_mat._frequency_update_callback = mat._frequency_update_callback
            print(f"  Replaced callback on orphaned '{mat_name}' with P.527 {ground_type}")
    # Auto-detect ground objects if none specified
    if ground_object_names is None:
        ground_object_names = [
            name for name in scene.objects
            if "floor" in name.lower() or "ground" in name.lower()
        ]

    assigned = []
    for name in ground_object_names:
        obj = scene.objects.get(name)
        if obj is None:
            print(f"  [Warning] Object '{name}' not found in scene, skipping.")
            continue
        obj.radio_material = mat
        assigned.append(name)
        print(f"  Assigned P.527 {ground_type} ground to: {name}")

    if not assigned:
        print(f"  [Warning] No ground objects found/assigned.")

    return assigned