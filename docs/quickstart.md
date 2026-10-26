# Quickstart

## Small complete example

After [setup](setup.md), run from the repository root:

```bash
python examples/run_synthetic.py
```

This traces 64 user locations in a synthetic ground-and-wall scene, constructs
8-element receiver fingerprints, and evaluates the predefined weighted k-NN
baselines. It runs deployment, tracing, refinement, link budget, channel taps,
beam-delay generation, preprocessing, and positioning in order. No city data is
needed. This is a pipeline demonstration, not a positioning-performance benchmark.

Outputs are written to `data/` and `models/synthetic/`; inspect each
`metrics.json` for positioning error in metres. All k/distance settings are
reported on the same cached test split. Do not select a winning hyperparameter
from these test results; use a separate validation procedure for a study.

## Geographic scenes

In the Geo2SigMap environment, choose one source:

```bash
python src/module_1_scene_gen/run_g2sm.py --config-name=nyc_osm
python src/module_1_scene_gen/run_g2sm.py --config-name=nyc_overture
python src/module_1_scene_gen/run_g2sm.py --config-name=nyc_lidar_dem
```

Each writes to `data/scenes/<selected_id>/`. OSM and Overture examples use flat
ground; the LiDAR example derives building heights and aligned DEM terrain.
The Overture example pins release `2026-10-21.0`. Archived release availability
is controlled upstream; choose an available explicit release if it is retired.

In the separate BostonTwin environment, with its dataset configured:

```bash
python src/module_1_scene_gen/run_bostontwin.py --config-name=boston
```

City configs initially define geometry only. Inspect the scene, then provide
`deployment.base_stations` in scene-local metres and choose a placement mode
before tracing. In this uplink pipeline, sampled user positions are TXs and
base stations are RXs. Geographical latitude/longitude is not a device coordinate.

For outdoor sampling, run `extract_building_footprints.py` with the same config
and exclusion buffer before `tx_rx_location_csv_gen.py`. Supported placement
modes are `box`, `polygon`, `box_polygon`, and `rooftop`; `box` does not exclude
buildings. For terrain-relative user heights, use
`deployment.height_reference: terrain` with a polygon mode and select the ground
shapes through `deployment.terrain_shape_ids`. Base-station coordinates remain
explicit scene-local coordinates.

Then run the same stages listed in `examples/run_synthetic.py`, replacing
`--config-name=synthetic` with your scenario. To use an existing Mitsuba scene,
set `scene_xml_path` to its XML and configure its deployment. Generated scenes
are skipped when present; use a new scene ID to preserve an earlier output.
