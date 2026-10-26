"""Explicit height-source selection for the pinned WCSL Geo2SigMap package.

The package's CLI always initializes its Overture lookup. This wrapper makes
the selected mode and Overture release explicit without changing the package.
"""
import json
import os
from pathlib import Path
import random
import re


def generate_scene(box, target_directory, settings, lidar_terrain, dem_terrain):
    import numpy as np
    from scene_generation import core, utils

    mode = settings["height_source"]
    if mode not in {"osm", "overture", "lidar"}:
        raise ValueError("height_source must be osm, overture, or lidar")
    release = settings.get("overture_release")
    if mode == "overture" and not re.fullmatch(r"\d{4}-\d{2}-\d{2}\.\d+", str(release)):
        raise ValueError("Overture requires an explicit YYYY-MM-DD.N release")
    if mode == "lidar" and not (lidar_terrain and dem_terrain):
        raise ValueError("LiDAR height generation requires both terrain switches")
    if mode != "lidar" and (lidar_terrain or dem_terrain):
        raise ValueError("The osm/overture examples use flat ground; use lidar for aligned DEM terrain")

    out = Path(target_directory).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_dir = out / "source_inputs"
    source_dir.mkdir(exist_ok=True)
    seed = int(settings.get("seed", 71))
    random.seed(seed)
    np.random.seed(seed)
    bounds = tuple(float(box[k]) for k in ("min_lon", "min_lat", "max_lon", "max_lat"))
    if not all(np.isfinite(bounds)) or not (-180 <= bounds[0] < bounds[2] <= 180 and
                                           -90 <= bounds[1] < bounds[3] <= 90):
        raise ValueError("Invalid WGS84 bounding box")

    old_cwd = os.getcwd()
    old_init = core.init_overture_lookup
    old_lookup = utils.OVERTURE_LOOKUP
    old_features = core.ox.features.features_from_bbox
    old_cache = core.ox.settings.use_cache, core.ox.settings.cache_folder
    details = dict(source="Geo2SigMap", height_source=mode, seed=seed,
                   overture_release=release if mode == "overture" else None,
                   bbox_wgs84=list(bounds))

    def save_features(*args, **kwargs):
        result = old_features(*args, **kwargs)
        (source_dir / "osm_buildings.geojson").write_text(result.to_json())
        return result

    try:
        core.init_overture_lookup = lambda *args, **kwargs: None
        utils.OVERTURE_LOOKUP = None
        core.ox.settings.use_cache = True
        core.ox.settings.cache_folder = str(source_dir / "osm_cache")
        if mode == "overture":
            from scene_generation.overture_height_lookup import OvertureHeightLookup
            lookup = OvertureHeightLookup(bbox=bounds, release_url=release, verbose=True)
            if lookup.buildings.empty:
                raise RuntimeError("No Overture buildings were retrieved; refusing an OSM-only output labeled Overture")
            utils.OVERTURE_LOOKUP = lookup
            details["overture_buildings_retrieved"] = len(lookup.buildings)
            details["overture_parts_retrieved"] = len(lookup.parts)
            lookup.buildings.to_file(source_dir / "overture_buildings.geojson", driver="GeoJSON")
            if not lookup.parts.empty:
                lookup.parts.to_file(source_dir / "overture_parts.geojson", driver="GeoJSON")
        if mode == "lidar":
            if __package__:
                from .geo2sigmap_adapter import generate_scene as generate_lidar
            else:
                from geo2sigmap_adapter import generate_scene as generate_lidar
            result = generate_lidar(box, str(out), settings, lidar_terrain, dem_terrain)
            result.update(details)
            return result
        core.ox.features.features_from_bbox = save_features
        lo, la, hi, ha = bounds
        os.chdir(out)
        core.Scene()(points=[(lo, la), (lo, ha), (hi, ha), (hi, la)],
                     data_dir=str(out), hag_tiff_path=str(out / "unused_hag.tif"),
                     lidar_calibration=False,
                     generate_building_map=True, lidar_terrain=False, dem_terrain=False,
                     ground_scale=float(settings.get("ground_scale", 1.5)),
                     ground_material_type=settings.get("ground_material", "mat-itu_medium_dry_ground"),
                     wall_material_type=settings.get("wall_material", "mat-itu_concrete"),
                     rooftop_material_type=settings.get("rooftop_material", "mat-itu_concrete"))
        if not (out / "scene.xml").is_file():
            raise RuntimeError("Geo2SigMap did not produce scene.xml")
        return details
    finally:
        core.init_overture_lookup = old_init
        utils.OVERTURE_LOOKUP = old_lookup
        core.ox.features.features_from_bbox = old_features
        core.ox.settings.use_cache, core.ox.settings.cache_folder = old_cache
        os.chdir(old_cwd)
        (out / "height_source.json").write_text(json.dumps(details, indent=2) + "\n")
