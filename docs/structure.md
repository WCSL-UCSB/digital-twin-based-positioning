# Repository guide

```text
configs/
  scenarios/              Shared physics and small example configurations
  gps_coordinates/        Geographic areas of interest
  materials/              ITU material callbacks and ground-property curves
src/
  module_1_scene_gen/      Scene adapters, footprints, placement, ray tracing
  module_2_path_refinement/ Field-of-view filtering, delays, link budget
  module_3_channel_gen/    Per-element channel impulse responses
  module_4_phy/            Pilot transmission and beam-delay fingerprints
  module_6_localization/   Preprocessing, splits, weighted k-NN positioning
  utils/                  Input selection, provenance, geometry, noise
examples/                 Synthetic scene and executable pipeline example
requirements/             Dependencies for separate environments
tests/                    Physics, input validation, routing, and geometry tests
docs/                     Setup, usage, visual examples, and research results
  assets/                 Scene, fingerprint, and positioning figures
  results/                Compact per-run metrics for published figures
third_party/boston_twin/   Optional pinned upstream submodule
```

Module numbers follow the pipeline stage names. Run scripts from the repository
root; installation as a Python package is not required. A single Hydra scenario
configures all stages. Duplicate an example and give it a unique `selected_id`
and `scenario_name` for a new scene/deployment.

Generated files stay local:

- `data/scenes/`: scene XML, meshes, downloaded inputs, generation records.
- `configs/gps_coordinates/*_tx_rx.csv`: local deployment coordinates.
- `data/path_data/`: raw, refined, and calibrated paths.
- `data/channel_data/`: channel taps.
- `data/ml_datasets/` and `data/preprocessed_datasets/`: fingerprints and model inputs.
- `models/`: k-NN predictions and metrics; `outputs/`: Hydra logs.

Downstream stages select the newest matching filename timestamp by default.
For a reproducible run, pin each stage's input with `+inputs.paths_npz=...`,
`+inputs.filtered_npz=...`, `+inputs.calibrated_npz=...`,
`+inputs.channels_npz=...`, `+inputs.bdp_npz=...`, or
`+inputs.preprocessed_npz=...`. Check the stage's printed input and provenance
sidecar. Use only trusted NPZ files: some contain pickled metadata.
