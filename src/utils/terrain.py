"""Scene-local ground heights from explicitly selected Mitsuba PLY shapes.

Only placement with height_reference=terrain needs PyVista/VTK. No geographic
reprojection, raster interpolation, or dependency on another scene is involved.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _transform(shape):
    matrix = np.eye(4)
    transform = shape.find("transform[@name='to_world']")
    if transform is None:
        if any(item.get("name") == "to_world" for item in shape):
            raise ValueError("Terrain to_world must be an inline transform")
        return matrix
    for item in transform:
        operation = np.eye(4)
        if item.tag == "matrix":
            operation = np.fromstring(item.attrib["value"].replace(",", " "), sep=" ").reshape(4, 4)
        elif item.tag in ("translate", "scale"):
            default = 1.0 if item.tag == "scale" else 0.0
            if "value" in item.attrib:
                values = np.fromstring(item.attrib["value"].replace(",", " "), sep=" ")
                if len(values) == 1:
                    values = np.repeat(values, 3)
                if len(values) != 3:
                    raise ValueError("Expected one or three transform values")
            else:
                values = [float(item.get(k, default)) for k in ("x", "y", "z")]
            if item.tag == "scale":
                operation[:3, :3] = np.diag(values)
            else:
                operation[:3, 3] = values
        elif item.tag == "rotate":
            axis = np.array([float(item.get(k, 0)) for k in ("x", "y", "z")])
            norm = np.linalg.norm(axis)
            if norm == 0:
                raise ValueError("Rotation axis must be nonzero")
            x, y, z = axis / norm
            angle = np.deg2rad(float(item.attrib["angle"]))
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            operation[:3, :3] = (np.eye(3) * np.cos(angle)
                + (1 - np.cos(angle)) * np.outer(axis / norm, axis / norm)
                + np.sin(angle) * skew)
        else:
            raise ValueError(f"Unsupported terrain transform: {item.tag}")
        matrix = operation @ matrix
    if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError("Terrain requires a finite affine transform")
    return matrix


class TerrainSurface:
    """Vertical intersection with ground triangles; NaN outside mesh coverage.

    Shape IDs must identify only ground, never buildings or a fallback plane
    beneath a DEM. For overlapping selected surfaces, the highest hit wins.
    XML transforms are applied before building each spatial index.
    """

    def __init__(self, scene_path, shape_ids):
        if not shape_ids or isinstance(shape_ids, str):
            raise ValueError("deployment.terrain_shape_ids must be a nonempty list of XML ground shape IDs")
        import pyvista as pv
        from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator

        path = Path(scene_path)
        root = ET.parse(path).getroot()
        self._surfaces = []
        for shape_id in shape_ids:
            matches = [s for s in root.findall("shape") if s.get("id") == shape_id]
            if len(matches) != 1 or matches[0].get("type") != "ply":
                raise ValueError(f"Terrain shape {shape_id!r} must identify exactly one top-level PLY shape")
            shape = matches[0]
            filename = shape.find("string[@name='filename']")
            if filename is None:
                raise ValueError(f"Missing terrain filename: {shape_id}")
            mesh = pv.read(path.parent / filename.attrib["value"])
            if not mesh.is_all_triangles:
                raise ValueError(f"Terrain must contain only triangles: {shape_id}")
            if mesh.n_cells == 0 or not np.isfinite(mesh.points).all():
                raise ValueError(f"Empty or nonfinite terrain: {shape_id}")
            mesh.points = np.asarray(mesh.points, dtype=float)
            mesh.transform(_transform(shape), inplace=True)
            if not np.isfinite(mesh.points).all():
                raise ValueError(f"Nonfinite transformed terrain: {shape_id}")
            locator = vtkStaticCellLocator()
            locator.SetDataSet(mesh)
            locator.BuildLocator()
            self._surfaces.append((mesh, locator))
        self.bounds = np.array([m.bounds for m, _ in self._surfaces])

    def heights(self, xy):
        from vtkmodules.vtkCommonCore import reference
        from vtkmodules.vtkCommonDataModel import vtkGenericCell

        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        heights = np.full(len(xy), np.nan)
        for mesh, locator in self._surfaces:
            xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
            inside = (np.isfinite(xy).all(axis=1) & (xy[:, 0] >= xmin)
                      & (xy[:, 0] <= xmax) & (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax))
            t, sub_id, cell_id = reference(0.0), reference(0), reference(0)
            hit, pcoords, cell = [0.0] * 3, [0.0] * 3, vtkGenericCell()
            for i in np.flatnonzero(inside):
                x, y = xy[i]
                found = locator.IntersectWithLine(
                    [x, y, zmax + 1.0], [x, y, zmin - 1.0], 1e-9,
                    t, hit, pcoords, sub_id, cell_id, cell)
                if found:
                    heights[i] = np.fmax(heights[i], hit[2])
        return heights
