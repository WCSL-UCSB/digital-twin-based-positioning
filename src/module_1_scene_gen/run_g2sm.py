# =============================================================================
# run_g2sm.py — Geometry-to-Signal-Map (G2SM) Scene Generation
# =============================================================================
#
# AIM:
#   Generates 3D building/scene geometry for a given GPS bounding box using
#   the `scenegen` CLI tool. This is the first step in the digital twin
#   pipeline, converting real-world GPS coordinates into Sionna-compatible
#   scene geometry for downstream ray tracing and channel simulation.
#
# INPUTS:
#   - Hydra config (from configs/scenarios/*.yaml):
#       * gps_location_path : path to GPS locations YAML
#                             (default: data/gps_coordinates/gps_locations.yaml)
#       * selected_id       : location key to look up (e.g. "nyc_high_rise")
#       * output_root       : root directory for generated scenes
#                             (default: data/scenes/)
#   - GPS locations YAML file containing bounding boxes (min/max lat/lon)
#     for each location ID.
#
# OUTPUTS:
#   - Scene geometry files written to:
#       <output_root>/<selected_id>/
#     Skips existing scenes; reports incomplete nonempty directories as errors.
#
# CONNECTED DIRECTORIES:
#   - configs/scenarios/       — Hydra config files (test0207.yaml, etc.)
#   - data/gps_coordinates/    — GPS bounding box definitions
#   - data/scenes/             — Output scene geometry (consumed by path_gen.py
#                                in module_1 and downstream modules)
#
# USAGE:
#   This script calls the `scenegen` CLI provided by geo2sigmap. Two
#   compatible installs are supported — pick the env that matches the
#   variant you want, then run the script as usual:
#
#     conda activate g2sm_hydra            # upstream geo2sigmap
#         OR
#     conda activate geo2sigmap-overture   # our fork with new features
#
#     python run_g2sm.py --config-name=<scenario>
#
#   Both envs expose the same `scenegen bbox ...` CLI, so the script body
#   is identical; only the active env decides which build runs.
#
# =============================================================================

import os
import shutil
import subprocess
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

load_dotenv()

def run_scenegen_box(location_id, box, output_root,
                    enable_lidar_terrain=True, enable_dem_terrain=True,
                    force=False, generation_settings=None):
    target_directory = os.path.join(output_root, location_id)

    if os.path.exists(target_directory) and os.listdir(target_directory):
        resume = bool(generation_settings and generation_settings.get("resume_incomplete", False))
        if resume and not force and not os.path.exists(os.path.join(target_directory, "scene.xml")):
            print(f"[Resume] Reusing source inputs for incomplete scene: {target_directory}")
        elif not force:
            if not os.path.isfile(os.path.join(target_directory, "scene.xml")):
                raise RuntimeError(
                    f"Incomplete scene directory: {target_directory}; scene.xml is missing. "
                    "Use a new output directory or force=true to regenerate."
                )
            print(f"[Skip] Geometry for '{location_id}' already exists in: "
                  f"{target_directory}  (set force=true to overwrite)")
            return
        else:
            print(f"[Force] Removing existing {target_directory}")
            shutil.rmtree(target_directory)

    os.makedirs(target_directory, exist_ok=True)
    print(f"--- Generating Geometry for: {location_id} ---")

    if generation_settings and generation_settings.get("height_source"):
        if __package__:
            from .configured_geo2sigmap import generate_scene
        else:
            from configured_geo2sigmap import generate_scene
        return generate_scene(box, target_directory, generation_settings,
                              enable_lidar_terrain, enable_dem_terrain)

    if generation_settings and generation_settings.get("lidar_building_heights", False):
        if not (enable_lidar_terrain and enable_dem_terrain):
            raise ValueError("The LiDAR/DEM adapter requires both terrain switches enabled")
        # Supports both direct script execution and package imports in tests.
        if __package__:
            from .geo2sigmap_adapter import generate_scene
        else:
            from geo2sigmap_adapter import generate_scene
        return generate_scene(box, target_directory, generation_settings,
                              enable_lidar_terrain, enable_dem_terrain)

    cmd = [
        "scenegen", "bbox",
        str(box['min_lon']), str(box['min_lat']),
        str(box['max_lon']), str(box['max_lat']),
        "--data-dir", "."
    ]
    if enable_lidar_terrain:
        cmd.append("--enable-lidar-terrain")
    if enable_dem_terrain:
        cmd.append("--enable-dem-terrain")

    print(f"[CMD] {' '.join(cmd)}")

    try:
        subprocess.run(cmd, cwd=target_directory, check=True, text=True)
        if not os.path.isfile(os.path.join(target_directory, "scene.xml")):
            raise RuntimeError("Geo2SigMap exited without producing scene.xml")
        print(f"[Success] Geometry saved to {target_directory}")
    except Exception as e:
        print(f"[Error] {e}")
        if os.path.exists(target_directory) and not os.listdir(target_directory):
            os.rmdir(target_directory)
        sys.exit(1)

@hydra.main(version_base=None, config_path="../../configs/scenarios")
def main(cfg: DictConfig):

    OmegaConf.resolve(cfg)

    location_path = cfg.gps_location_path
    selected_id = cfg.selected_id
    output_root = cfg.output_root

    print(f"Project Root: {cfg.project_dir}")
    print(f"Loading location from: {location_path}")

    #  Load the location file
    gps_location = OmegaConf.load(location_path)

    if selected_id not in gps_location:
        raise KeyError(f"Location '{selected_id}' not found in {location_path}")

    #  Extract the box
    loc_data = gps_location[selected_id]
    box = loc_data['box']

    # Run generation if the geometry doesn't already exist (top-level
    # `force: true` in the scenario YAML overrides the skip).
    details = run_scenegen_box(
        selected_id, box, output_root,
        enable_lidar_terrain=bool(cfg.get("enable_lidar_terrain", True)),
        enable_dem_terrain=bool(cfg.get("enable_dem_terrain", True)),
        generation_settings=OmegaConf.to_container(cfg.scene_generation, resolve=True)
                            if cfg.get("scene_generation") is not None else None,
        force=bool(cfg.get("force", False)),
    )

    if details is not None:
        if __package__:
            from .scene_run_record import record_generation
        else:
            from scene_run_record import record_generation
        record_generation(cfg, os.path.join(output_root, selected_id), details)

if __name__ == "__main__":
    main()
