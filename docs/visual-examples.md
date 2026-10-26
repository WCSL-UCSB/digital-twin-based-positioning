# Visual examples

## New York scene generation

![Times Square scene comparison](assets/nyc-scene-comparison.png)

Both views show actual generated XML/PLY geometry for the Times Square area,
using the same camera, scale, 58 OSM building footprints, and USGS DEM terrain.
The area of interest spans longitude −73.988155 to −73.984597 and latitude
40.756979 to 40.759673. Source data were retrieved on October 23, 2026.

- **Left — OSM + Overture:** OSM height/level tags supply 45 building heights;
  matching Overture features supply 11; two use seeded fallback heights.
  Overture release: `2026-10-21.0`.
- **Right — Geo2SigMap LiDAR:** each building height is the mean of up to
  30 valid LiDAR height-above-ground samples inside its footprint.
  All 58 buildings had valid samples in this example.

These methods estimate different height quantities; the comparison illustrates
their outputs, rather than establishing which geometry is more accurate.
OSM lifecycle tags are not filtered: the tall “The Torch” feature uses a
325.1 m OSM height tag despite being marked as under construction. The picture
therefore should not be read as a survey of completed buildings.
Both panels use Mitsuba 3.7.1 optical rendering with neutral display materials.
Lighting and shadows help show the geometry; they are not radio-coverage results.

The pictured Overture scene reuses the LiDAR example's DEM for a shared terrain
reference. The public `nyc_osm` and `nyc_overture` quickstart configurations use
flat ground; `nyc_lidar_dem` includes terrain. See [scene sources](scene-sources.md)
and [quickstart](quickstart.md) for the available routes.

**Credits:** generated with [Geo2SigMap](https://github.com/functions-lab/geo2sigmap)
and the [WCSL fork](https://github.com/WCSL-UCSB/geo2sigmap).
© [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
[Overture Maps Foundation](https://docs.overturemaps.org/attribution/).
The retrieved Overture features identify OpenStreetMap, USGS LiDAR, and Microsoft
building data as sources; their underlying database terms remain applicable.
OSM and Microsoft building data in that release are identified as ODbL-1.0.
Terrain uses the [USGS 3DEP](https://www.usgs.gov/3d-elevation-program)
1 m DEM product `USGS_one_meter_x58y452_NY_CMPG_2013`.
LiDAR inputs include [USGS NY_NewYorkCity](https://s3-us-west-2.amazonaws.com/usgs-lidar-public/NY_NewYorkCity/ept.json)
and [NOAA coastal LiDAR dataset 6209](https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/entwine/geoid18/6209/ept.json).
See [third-party notices](../NOTICE.md).

## Beam-delay fingerprints

![Synthetic beam-delay fingerprints](assets/synthetic-fingerprints.png)

This saved output comes from `python examples/run_synthetic.py`, run on
October 26, 2026. The example uses a ground-and-wall scene, 64 sampled user
positions, one base station with an eight-element receive array, 7 GHz carrier
frequency, and 100 MHz bandwidth. The panels show three different user positions.
Horizontal position is the delay bin; vertical position is the beam index.
The 50 dB display range is for visualization.

These fingerprints illustrate an intermediate pipeline output. They are separate
from the BostonTwin CNN study in [the research results](results.md).
