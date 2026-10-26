# Positioning accuracy map

![Positioning error for two CNN models](assets/positioning-results.png)

Each dot is a held-out user location in one BostonTwin scene. Color shows raw
2D positioning error averaged across training seeds 71–73. Both models use the
same 3,841 locations and nominal training/testing conditions. The shared color
scale saturates at 30 m. No spatial interpolation or prediction ensembling is used.

Early fusion (C03) averages **11.09 m** error. Late fusion (I-joint) averages
**5.55 m**. The systems also differ in input normalization and supervision.
C03 runs use A100 GPUs. I-joint uses A40 for seeds 71/73 and A100 for seed 72,
whose inference disables TF32.

The shared split contains 13,440 training, 1,920 validation, and 3,841 test
positions. An additional 799 uncovered positions were excluded upstream.
Checkpoints were selected on validation data. This is a simulation example
within one scene, with a test roster used in prior studies.

CNN code, weights, and the full dataset are not yet released.
[Map values and provenance](results/positioning-map.json) accompany the figure.
Replot with `python docs/plot_accuracy_map.py`.

Geometry: [BostonTwin / WiNES Lab](https://github.com/wineslab/boston_twin),
derived from the Boston planning agency's 3D city model.
