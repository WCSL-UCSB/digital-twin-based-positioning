"""Exercise aligned DEM export with a local raster and stubbed data providers."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from src.module_1_scene_gen.geo2sigmap_adapter import generate_scene


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ("pyvista", "rasterio", "pyproj", "shapely")),
                     "DEM export requires the Geo2SigMap geospatial environment")
class AlignedDemTests(unittest.TestCase):
    def test_raster_alignment_nodata_and_hook_restoration(self):
        import pyvista as pv
        import rasterio
        from rasterio.transform import from_origin
        from pyproj import CRS, Transformer
        box = dict(min_lon=-71.1, min_lat=42.3, max_lon=-71.099, max_lat=42.301)
        crs = CRS.from_epsg(32619)
        xy = Transformer.from_crs(4326, crs, always_xy=True)
        # Deliberately offset raster footprint: recentering it independently would fail.
        x, y = xy.transform(-71.1, 42.3)
        transform = from_origin(x - 20, y + 30, 2, 2)
        values = np.array([[10., 12., -9999], [14., 20., 22.], [24., 26., 28.]], dtype="float32")
        core = ModuleType("scene_generation.core")
        core.__file__ = __file__
        lidar = ModuleType("scene_generation.USGS_LiDAR_HAG")
        core.ox = SimpleNamespace(features=SimpleNamespace(features_from_bbox=Mock()),
                                 settings=SimpleNamespace(use_cache=False, cache_folder="original"))
        core.get_utm_epsg_code_from_gps = lambda *args: crs
        core.random_building_height = Mock(return_value=5.)
        core.generate_terrain_mesh_dem = Mock()
        core.generate_random_points = lambda *args: [SimpleNamespace(x=x, y=y)]
        handler = SimpleNamespace(query=lambda *args: [7.], src=SimpleNamespace(close=Mock()))
        core.GeoTIFFHandler = lambda *args: handler
        lidar.generate_hag = Mock()
        lidar.pdal = SimpleNamespace(Pipeline=Mock())
        original_dem, original_height = core.generate_terrain_mesh_dem, core.random_building_height
        original_hag, original_pipeline = lidar.generate_hag, lidar.pdal.Pipeline
        package = ModuleType("scene_generation")
        package.core, package.USGS_LiDAR_HAG = core, lidar
        dem = ModuleType("scene_generation.dem_utils")
        gpkg = ModuleType("scene_generation.dem_gpkg_data")
        locator = ModuleType("scene_generation.USGS1mLocator_parallel")
        gpkg.get_fesm_paths = lambda: ["a", "b"]
        locator.USGS1mLocator = lambda *args, **kw: SimpleNamespace(get_links=lambda aoi: ["local-fixture"])
        modules = {m.__name__: m for m in (package, core, lidar, dem, gpkg, locator)}
        old_cwd = os.getcwd()
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as d:
                out = Path(d)
                (out / "mesh").mkdir()
                def clip(*args, **kwargs):
                    with rasterio.open(out / "mesh/dem.tif", "w", driver="GTiff", height=3, width=3,
                                       count=1, dtype="float32", crs=crs, transform=transform, nodata=-9999) as ds:
                        ds.write(values, 1)
                    return {"source": "local-fixture"}
                dem.clip_reproject_dem_to_wgs84_utm = clip
                class FakeScene:
                    def __call__(self, **kwargs):
                        core.generate_terrain_mesh_dem(None, out / "mesh/ground.ply")
                        core.random_building_height(None, SimpleNamespace(bounds=[0, 0, 1, 1]))
                        if fail:
                            raise RuntimeError("provider failed")
                        (out / "scene.xml").write_text("<scene/>")
                core.Scene = FakeScene
                with patch.dict(sys.modules, modules):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "provider failed"):
                            generate_scene(box, out, {}, True, True)
                    else:
                        generate_scene(box, out, {}, True, True)
                self.assertEqual(os.getcwd(), old_cwd)
                self.assertIs(core.generate_terrain_mesh_dem, original_dem)
                self.assertIs(core.random_building_height, original_height)
                self.assertIs(lidar.generate_hag, original_hag)
                self.assertIs(lidar.pdal.Pipeline, original_pipeline)
                self.assertEqual(core.ox.settings.cache_folder, "original")
                audit = json.loads((out / "generation_sources.json").read_text())
                mesh = pv.read(out / "mesh/ground.ply")
                origin = np.asarray(audit["horizontal_origin_m"])
                mean = values[values != -9999].mean()
                rows, cols = np.indices(values.shape)
                xs, ys = rasterio.transform.xy(transform, rows.ravel(), cols.ravel(), offset="center")
                expected = np.column_stack([np.asarray(xs) - origin[0], np.asarray(ys) - origin[1], values.ravel() - mean])
                expected = expected[values.ravel() != -9999]
                self.assertEqual(mesh.n_cells, 7)  # only the triangle touching nodata is omitted
                for point in mesh.points:
                    self.assertLess(np.linalg.norm(expected - point, axis=1).min(), 1e-4)
                self.assertEqual(audit["terrain"]["vertical_origin_m"], float(mean))
                self.assertNotIn(b"obj_info ", (out / "mesh/ground.ply").read_bytes().split(b"end_header")[0])


if __name__ == "__main__":
    unittest.main()
