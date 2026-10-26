# Scene sources

All routes supply a Mitsuba XML scene and geometry for Sionna RT. They share
the downstream placement, propagation, fingerprint, and positioning stages.

## Geo2SigMap / OpenStreetMap

[Geo2SigMap](https://github.com/functions-lab/geo2sigmap) retrieves OpenStreetMap
building footprints and tags through OSMnx/Overpass, then extrudes footprints
into meshes. In the pinned WCSL fork, heights use OSM height/level tags where
available and seeded random fallback otherwise. The `nyc_osm` example disables
the fork's Overture lookup explicitly.

## Geo2SigMap / Overture fallback

Overture is an additional height source inside the Geo2SigMap route. OSM
footprints and usable height tags retain priority. Missing heights can use
matching Overture buildings/building parts downloaded as GeoParquet through
DuckDB from the public Overture S3 release. Unmatched buildings retain the
seeded fallback. This produces one extruded height per OSM footprint, rather
than reconstructing each Overture building part as independent geometry.

The [Overture buildings guide](https://docs.overturemaps.org/guides/buildings/)
describes its combined source datasets. The `sources` field identifies the
provenance of individual features; the upstream height lookup used here does
not fetch that field. Saved lookup geometries and a release ID support replay
but are not a complete per-feature attribution record. Consult the original
release's source records before redistributing generated city datasets.

## Geo2SigMap / LiDAR and DEM

The optional route uses USGS 3DEP LiDAR height-above-ground samples to estimate
building heights and USGS 1 m DEM tiles for terrain. The adapter averages up to
30 valid LiDAR samples per footprint, falls back to the configured package's
height routine where needed, and aligns terrain XY to the building origin.
It records input selection and the DEM's vertical offset. Coverage depends on
the source datasets; this is not a worldwide high-resolution terrain service.
See [USGS 3DEP](https://www.usgs.gov/3d-elevation-program).

## BostonTwin

[BostonTwin](https://github.com/wineslab/boston_twin) assembles existing Boston
3D meshes and geographic metadata. Its building data originates from the
Boston planning agency's 3D city model; antenna locations are available in
the upstream dataset. Our adapter assembles geometry, applies the mesh-origin
correction, and writes relative asset paths. Device placement is configured
separately. BostonTwin covers Boston, so it cannot generate the New York example.

## Existing scenes

An existing Mitsuba XML scene can enter through `scene_xml_path`, including a
scene exported from Blender. Geometry units, local origin, materials, antenna
placement, and terrain heights must be consistent before comparing sources.
Visual similarity alone does not establish radio-propagation equivalence.

Downloaded data, generated city meshes, and antenna datasets are not bundled.
Their source licenses and attribution requirements are separate from the
project code license; see [third-party notices](../NOTICE.md).
