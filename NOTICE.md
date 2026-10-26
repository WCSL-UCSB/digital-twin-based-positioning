# Attribution and third-party terms

Original project contributions by Canan Cebeci, Josef Ullrich, and contributors
are licensed under [CC BY-NC 4.0](LICENSE), except where another license or
attribution is identified. Credit the contributors, link the repository and
license, and indicate modifications when sharing adapted work. Commercial use
of these contributions requires separate permission from the relevant rights
holders. This license does not replace the terms of third-party software or data.

## Geo2SigMap

This project integrates [Geo2SigMap](https://github.com/functions-lab/geo2sigmap)
and the [WCSL fork](https://github.com/WCSL-UCSB/geo2sigmap). The scene package at
revision `ab0667265e5cf696bd57d3273600a968fc125107` is licensed under
[CC BY-NC 4.0](LICENSES/Geo2SigMap-CC-BY-NC-4.0.txt).

`configs/materials/itu_materials.py` and the six CSV files under
`configs/materials/data/` derive from Geo2SigMap's `scene_generation/itu_materials.py`
and `scene_generation/data/`, credited to the Geo2SigMap authors and contributors.
The curves represent digitized ITU-R P.527-3 ground properties. Adaptations
relocate the resources, cache spline loading, and expose callbacks for this
pipeline. The upstream package's license is preserved in `LICENSES/`.

## BostonTwin and Sionna

- [BostonTwin](https://github.com/wineslab/boston_twin), copyright 2024 WiNES Lab:
  MIT. The optional submodule retains its upstream license. Dataset terms are
  separate; consult the [BostonTwin data documentation](https://github.com/wineslab/boston_twin#data).
- [Sionna](https://github.com/NVlabs/sionna): Apache-2.0. It and its renderer
  dependencies are installed separately and retain their upstream licenses.

## Geographic data

- OpenStreetMap data: © OpenStreetMap contributors, under the
  [Open Database License](https://www.openstreetmap.org/copyright).
- Overture Maps: follow the selected release's source-specific
  [attribution and licensing](https://docs.overturemaps.org/attribution/).
- USGS LiDAR/DEM: retain the selected products' metadata and source attribution;
  consult [USGS data policies](https://www.usgs.gov/information-policies-and-instructions).

No third-party city dataset, trained weights, or generated radio map is included.
