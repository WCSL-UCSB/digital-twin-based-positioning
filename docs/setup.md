# Setup

## Pipeline environment

Use Python 3.11 and the pinned Sionna RT 1.2.1 API. The scene-to-positioning
example can run on CPU; larger ray-tracing jobs benefit from a supported GPU.

```bash
git clone https://github.com/WCSL-UCSB/digital-twin-based-positioning.git
cd digital-twin-based-positioning
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/pipeline.txt
export PROJECT_DIR="$PWD"
```

Alternatively, copy `.env.example` to `.env` and set `PROJECT_DIR` to this
checkout's absolute path. Sionna RT requires its supported Mitsuba/Dr.Jit
runtime; consult [Sionna installation](https://nvlabs.github.io/sionna/installation.html)
for driver/platform requirements. The 1.2.1 pin is intentional; this pipeline
has not been migrated to the newer 2.x API.

## Geo2SigMap environment (optional)

Use a separate environment for the geospatial stack. The pinned public WCSL
fork supplies the Overture height lookup; the repository wrapper selects OSM,
Overture fallback, or LiDAR heights explicitly.

```bash
conda create -n dt-scenes -c conda-forge python=3.11 pip pdal python-pdal
conda activate dt-scenes
python -m pip install -r requirements/geo2sigmap.txt
export PROJECT_DIR="$PWD"
```

PDAL must provide both the Python module and the `pdal` executable inside the
environment. LiDAR/DEM generation downloads source data and may need substantial
disk space. See the [upstream setup instructions](https://github.com/WCSL-UCSB/geo2sigmap/blob/ab0667265e5cf696bd57d3273600a968fc125107/package/README.md)
for geospatial dependency troubleshooting. Keep the two environments separate;
switch back to the pipeline environment after producing `scene.xml`.

## BostonTwin environment (optional)

```bash
git submodule update --init third_party/boston_twin
```

Follow [BostonTwin's installation and dataset instructions](https://github.com/wineslab/boston_twin#quickstart)
in its own environment/container, then install `hydra-core==1.3.2` and
`python-dotenv`. BostonTwin uses an older Sionna environment for scene assembly;
use the pipeline environment above for downstream tracing. Mount this checkout
when using a container. Set `BOSTONTWIN_ROOT` to the parent of the dataset's
`boston3d` directory. Use a writable local dataset copy: the upstream generator
writes scene artifacts into it.

## Tests

In the pipeline environment:

```bash
python -m unittest discover -s tests -v
```

Geometry-specific tests skip when optional geospatial dependencies are absent.
To run them, install `pyvista`, `rasterio`, and `pyproj` in a test environment.
The tests use temporary fixtures; they do not download city data or submit jobs.
