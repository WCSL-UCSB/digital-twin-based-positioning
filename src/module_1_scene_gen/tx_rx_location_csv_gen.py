# =============================================================================
# tx_rx_location_csv_gen.py — TX/RX Position Layout Generator
# =============================================================================
#
# AIM:
#   Generates a CSV file defining transmitter (TX) and receiver (RX) positions
#   for a given deployment scenario. TX positions are randomly sampled within
#   a configured bounding box, while the RX (base station) is placed at a
#   fixed location. This layout is used by downstream modules (path_gen,
#   ray tracing) to simulate the wireless channel between each TX-RX pair.
#
# INPUTS:
#   - Hydra config (from configs/scenarios/*.yaml):
#       * selected_id              : location key (e.g. "nyc_high_rise")
#       * deployment.num_samples   : number of TX positions to generate
#       * deployment.tx_box        : bounding box for random TX placement
#                                    (x_min/max, y_min/max, z_min/max)
#       * deployment.rx_position   : fixed RX coordinates [x, y, z]
#       * deployment.rx_look_at    : RX look-at direction [x, y, z]
#       * deployment.rx_name       : name label for the RX entry
#
# OUTPUTS:
#   - CSV file written to:
#       data/gps_coordinates/<selected_id>_tx_rx.csv
#     Columns: name, type, x, y, z, look_at_x, look_at_y, look_at_z
#     Contains N TX rows (tx1, tx2, ...) + 1 RX row.
#
# CONNECTED DIRECTORIES:
#   - configs/scenarios/       — Hydra config files defining deployment params
#   - data/gps_coordinates/    — Output CSV consumed by path_gen.py and
#                                visualize_deployment.py in module_1
#
# USAGE:
#   python tx_rx_location_csv_gen.py --config-name=<scenario>
#   e.g.: python tx_rx_location_csv_gen.py --config-name=test0217
#
# =============================================================================

import os
import csv
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

load_dotenv()

@hydra.main(version_base=None, config_path="../../configs/scenarios")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    project_dir = cfg.project_dir
    loc_id = cfg.selected_id
    
    # Generate TX Positions in the box
    box = cfg.deployment.tx_box
    N = cfg.deployment.num_samples
    
    rand_x = np.random.uniform(box.x_min, box.x_max, N)
    rand_y = np.random.uniform(box.y_min, box.y_max, N)
    rand_z = np.random.uniform(box.z_min, box.z_max, N)
    tx_positions = np.stack([rand_x, rand_y, rand_z], axis=1)

    # Setup Output Path
    output_dir = os.path.join(project_dir, "data", "gps_coordinates")
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f"{loc_id}_tx_rx.csv")

    # Write CSV
    fieldnames = ['name', 'type', 'x', 'y', 'z', 'look_at_x', 'look_at_y', 'look_at_z']
    
    print(f"Generating {N} TX samples for location: {loc_id}")
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Add the Randomized Transmitters (tx1, tx2, ...)
        for i, pos in enumerate(tx_positions):
            writer.writerow({
                'name': f'tx{i+1}',
                'type': 'tx',
                'x': f"{pos[0]:.3f}", 
                'y': f"{pos[1]:.3f}", 
                'z': f"{pos[2]:.3f}",
                'look_at_x': '', 'look_at_y': '', 'look_at_z': ''
            })

        # Add the Fixed Receiver
        rx_pos = cfg.deployment.rx_position
        rx_look = cfg.deployment.rx_look_at
        writer.writerow({
            'name': cfg.deployment.rx_name,
            'type': 'rx',
            'x': f"{rx_pos[0]:.3f}", 
            'y': f"{rx_pos[1]:.3f}", 
            'z': f"{rx_pos[2]:.3f}",
            'look_at_x': f"{rx_look[0]:.3f}", 
            'look_at_y': f"{rx_look[1]:.3f}", 
            'look_at_z': f"{rx_look[2]:.3f}"
        })

    print(f"Success :)! Generated layout with RX saved to: {csv_filename}")

if __name__ == "__main__":
    main()