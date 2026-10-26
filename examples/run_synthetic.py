"""Run the small scene-to-positioning example with the current Python environment."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "src/module_1_scene_gen/tx_rx_location_csv_gen.py",
    "src/module_1_scene_gen/path_gen.py",
    "src/module_2_path_refinement/filter_and_adjust_paths.py",
    "src/module_2_path_refinement/apply_link_budget.py",
    "src/module_3_channel_gen/channel_taps_gen.py",
    "src/module_4_phy/beam_delay_gen.py",
    "src/module_6_localization/ml_preprocessing.py",
    "src/module_6_localization/knn_baseline.py",
)

def main():
    env = dict(os.environ, PROJECT_DIR=str(ROOT), MPLBACKEND="Agg")
    for stage in STAGES:
        print(f"\nRunning {stage}", flush=True)
        subprocess.run([sys.executable, str(ROOT / stage), "--config-name=synthetic"],
                       cwd=ROOT, env=env, check=True)
    print("\nCompleted. Fingerprints: data/preprocessed_datasets/synthetic")
    print("Positioning results: models/synthetic/baseline_knn_*/metrics.json")

if __name__ == "__main__":
    main()
