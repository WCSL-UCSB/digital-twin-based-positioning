"""Resolve geometry independently of the deployment/CSV identifier."""

import math
import os


def resolve_scene_path(cfg):
    """Return the absolute scene XML path for a resolved scenario config.

    An explicit ``scene_xml_path`` takes precedence. Otherwise use
    ``<output_root>/<selected_id>/scene.xml``, with ``output_root`` defaulting
    to ``<project_dir>/data/scenes``. Relative paths are project-relative.

    Keep ``selected_id`` unchanged: it may identify a deployment whose CSV
    is distinct from the physical scene it reuses.
    """
    project_dir = os.path.abspath(os.path.expanduser(str(cfg["project_dir"])))
    path = cfg.get("scene_xml_path")
    if not path:
        output_root = cfg.get("output_root") or os.path.join(
            project_dir, "data", "scenes")
        path = os.path.join(str(output_root), str(cfg["selected_id"]), "scene.xml")
    path = os.path.expanduser(str(path))
    if not os.path.isabs(path):
        path = os.path.join(project_dir, path)
    return os.path.abspath(path)


def _checked_buffer_m(value):
    buffer_m = float(value)
    if not math.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError("building_exclusion.buffer_m must be finite and nonnegative")
    return 0.0 if buffer_m == 0 else buffer_m


def footprint_buffer_m(cfg):
    """Configured footprint clearance, sharing extraction's 1 m default."""
    deployment = cfg.get("deployment") or {}
    exclusion = deployment.get("building_exclusion") or {}
    return _checked_buffer_m(exclusion.get("buffer_m", 1.0))


def buffered_footprint_path(scene_dir, buffer_m):
    """Keep different clearances separate without replacing legacy footprints.

    Python's round-trip float representation avoids filename collisions from
    rounding nearby buffer settings. Equivalent values (1 and 1.0) share a path.
    """
    buffer_m = _checked_buffer_m(buffer_m)
    return os.path.join(str(scene_dir), "footprints", f"buffer_{buffer_m}m",
                        "building_footprints.npz")
