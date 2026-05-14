"""
configs.materials.data — P.527-3 ground material CSV data files.

Contains conductivity and permittivity vs. frequency curves digitized
from ITU-R P.527-3 for three soil types:

    B_wet_ground_con.csv         — wet ground conductivity
    B_wet_ground_per.csv         — wet ground permittivity
    D_medium_dry_ground_con.csv  — medium dry ground conductivity
    D_medium_dry_ground_per.csv  — medium dry ground permittivity
    E_very_dry_ground_con.csv    — very dry ground conductivity
    E_very_dry_ground_per.csv    — very dry ground permittivity

Each CSV has two columns (no header):
    column 0: frequency in MHz
    column 1: material property value (ε_r or σ in S/m)

These files were originally maintained under the scene_generation.data
package by Yiming's code. They have been moved here to keep all
material-related resources under configs/materials/.

If updating these files, ensure the digitized values are verified
against the ITU-R P.527-3 source document.
"""

from .itu_materials import apply_ground_materials

__all__ = ["apply_ground_materials"]
