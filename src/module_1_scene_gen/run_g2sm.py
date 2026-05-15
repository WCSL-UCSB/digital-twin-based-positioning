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
#     Skips generation if the output directory already exists and is non-empty.
#
# CONNECTED DIRECTORIES:
#   - configs/scenarios/       — Hydra config files (test0207.yaml, etc.)
#   - data/gps_coordinates/    — GPS bounding box definitions
#   - data/scenes/             — Output scene geometry (consumed by path_gen.py
#                                in module_1 and downstream modules)
#
# USAGE:
#   python run_g2sm.py --config-name=<scenario>
#   e.g.: python run_g2sm.py --config-name=test0217
#
# =============================================================================

import os
import subprocess
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

load_dotenv()

def run_scenegen_box(location_id, box, output_root,
                    enable_lidar_terrain=True, enable_dem_terrain=True):
    target_directory = os.path.join(output_root, location_id)

    if os.path.exists(target_directory) and os.listdir(target_directory):
        print(f"[Skip] Geometry for '{location_id}' already exists in: {target_directory}")
        return

    os.makedirs(target_directory, exist_ok=True)
    print(f"--- Generating Geometry for: {location_id} ---")

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
    
    # Run generation if the geometry doesn't already exist
    run_scenegen_box(selected_id, box, output_root)

if __name__ == "__main__":
    main()