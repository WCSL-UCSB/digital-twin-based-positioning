"""Render the TX/RX deployment PNG for an EXISTING tx_rx CSV, without
re-sampling TX points (which tx_rx_location_csv_gen.py would do). Reuses the
exact deployment-plot style (gray buildings, green TX, colored BS stars).

Used for the best urban scenario (bs3_lo10) whose CSV was built by reusing the
20k urban TX draw — so we must plot FROM that CSV, not regenerate it.

Run (tf_loc or any numpy+matplotlib env):
  python src/module_1_scene_gen/plot_deployment_from_csv.py \
      configs/gps_coordinates/boston_cambridge_street_bs3_lo10_tx_rx.csv \
      boston_cambridge_street
Args: <csv_path> <scene_id>   (scene_id -> data/scenes/<scene_id>/building_footprints.npz)
Output: configs/gps_coordinates/<csv_stem_without _tx_rx>_tx_deployment.png
"""
import os, sys, csv
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tx_rx_location_csv_gen import load_footprints, save_deployment_image

load_dotenv()
PD = os.environ["PROJECT_DIR"]


def main():
    csv_path = sys.argv[1]
    scene_id = sys.argv[2]
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(PD, csv_path)

    tx, bs = [], []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
            if row["type"] == "tx":
                tx.append([x, y, z])
            elif row["type"] == "rx":
                bs.append({"name": row["name"], "position": [x, y, z]})
    tx = np.asarray(tx, float)
    print(f"CSV: {os.path.basename(csv_path)}  TX={len(tx)}  RX={len(bs)}")
    for b in bs:
        print(f"   {b['name']}: z={b['position'][2]:.0f} m")

    polys = load_footprints(os.path.join(PD, "data", "scenes", scene_id))
    print(f"buildings: {len(polys)}")

    stem = os.path.basename(csv_path).replace("_tx_rx.csv", "")
    out = os.path.join(PD, "configs", "gps_coordinates", f"{stem}_tx_deployment.png")
    save_deployment_image(tx, polys, out, base_stations=bs)


if __name__ == "__main__":
    main()
