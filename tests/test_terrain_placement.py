"""Terrain placement regressions using tiny meshes, without Sionna or downloads."""
import contextlib
import csv
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
from src.module_1_scene_gen import tx_rx_location_csv_gen as deployment
from src.module_1_scene_gen import run_g2sm
from src.utils.terrain import TerrainSurface
import xml.etree.ElementTree as ET


@unittest.skipUnless(importlib.util.find_spec("pyvista"), "Terrain placement requires PyVista/VTK")
class TerrainPlacementTests(unittest.TestCase):
    def setUp(self):
        import pyvista as pv
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        # Split on the 00--11 diagonal: g(x,y) = min(x,y), not x*y.
        mesh = pv.PolyData(np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]]),
                           [3, 0, 1, 3, 3, 0, 3, 2])
        mesh.save(self.root / "ground.ply")
        mesh.translate([0, 0, 100]).save(self.root / "roof.ply")
        self.xml = self.root / "scene.xml"
        self.xml.write_text('''<scene>
          <shape type="ply" id="ground"><string name="filename" value="ground.ply"/></shape>
          <shape type="ply" id="roof"><string name="filename" value="roof.ply"/></shape>
        </scene>''')
        self.cfg = OmegaConf.create({
            "project_dir": str(self.root), "scene_xml_path": str(self.xml),
            "selected_id": "independent", "ray_tracing_params": {"seed": 71},
            "deployment": {"mode": "box_polygon", "height_reference": "terrain",
                "terrain_shape_ids": ["ground"], "num_samples": 100, "tx_height": 1.5,
                "tx_box": {"x_min": -0.5, "x_max": 1.5, "y_min": -0.5,
                           "y_max": 1.5, "z_min": 1.5, "z_max": 1.5},
                "building_exclusion": {"buffer_m": 0., "oversample_factor": 4},
                "base_stations": [{"name": "bs", "position": [0.5, 0.5, 12.],
                                   "look_at": [0.25, 0.25, 1.75]}]}})
        self.footprint = np.array([[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7], [0.3, 0.3]])
        np.savez(self.root / "building_footprints.npz", num_buildings=[1],
                 poly_0000=self.footprint, buffer_m=[0.])
        stdout = contextlib.redirect_stdout(io.StringIO())
        stdout.__enter__()
        self.addCleanup(stdout.__exit__, None, None, None)

    def test_triangle_heights_and_missing_ground_without_roof_hits(self):
        surface = TerrainSurface(self.xml, ["ground"])
        z = surface.heights([[0.75, 0.25], [0.25, 0.75], [0.5, 0.5], [2, 0.5], [np.nan, 0]])
        np.testing.assert_allclose(z[:3], [0.25, 0.25, 0.5], atol=1e-9)
        self.assertTrue(np.isnan(z[3:]).all())

    def test_xml_transform_is_applied_in_order(self):
        tree = ET.parse(self.xml)
        transform = ET.SubElement(tree.getroot().find("shape"), "transform", name="to_world")
        ET.SubElement(transform, "scale", x="2", y="3", z="4")
        ET.SubElement(transform, "translate", x="10", y="20", z="-5")
        tree.write(self.xml)
        surface = TerrainSurface(self.xml, ["ground"])
        np.testing.assert_allclose(surface.heights([[11.5, 20.75]]), [-4.], atol=1e-9)
        self.assertTrue(np.isnan(surface.heights([[0.5, 0.5]])[0]))

    def test_holes_are_not_filled(self):
        import pyvista as pv
        pv.PolyData(np.array([[0., 0, 0], [1, 0, 0], [1, 1, 1]]), [3, 0, 1, 2]).save(self.root / "ground.ply")
        surface = TerrainSurface(self.xml, ["ground"])
        z = surface.heights([[0.75, 0.25], [0.25, 0.75]])
        self.assertAlmostEqual(z[0], 0.25)
        self.assertTrue(np.isnan(z[1]))

    def test_unknown_shape_and_unsupported_transform_fail(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            TerrainSurface(self.xml, ["missing"])
        tree = ET.parse(self.xml)
        transform = ET.SubElement(tree.getroot().find("shape"), "transform", name="to_world")
        ET.SubElement(transform, "lookat")
        tree.write(self.xml)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            TerrainSurface(self.xml, ["ground"])

    def test_polygon_and_box_polygon_follow_ground_and_exclude_buildings(self):
        for mode, sampler in [("polygon", deployment.sample_scene_polygon),
                              ("box_polygon", deployment.sample_box_polygon)]:
            with self.subTest(mode=mode):
                self.cfg.deployment.mode = mode
                a, polygons = sampler(self.cfg, np.random.default_rng(71))
                b, _ = sampler(self.cfg, np.random.default_rng(71))
                np.testing.assert_array_equal(a, b)
                self.assertEqual(a.shape, (100, 3))
                self.assertTrue(((a[:, :2] >= 0) & (a[:, :2] <= 1)).all())
                self.assertTrue(deployment.batch_polygon_check(a[:, 0], a[:, 1], polygons).all())
                np.testing.assert_allclose(a[:, 2], np.minimum(a[:, 0], a[:, 1]) + 1.5, atol=1e-9)

    def test_csv_keeps_base_station_coordinates_and_terrain_clearance(self):
        with patch.object(deployment, "save_deployment_image"):
            deployment.main.__wrapped__(self.cfg)
        with (self.root / "configs/gps_coordinates/independent_tx_rx.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 101)
        xyz = np.array([[float(row[k]) for k in ("x", "y", "z")] for row in rows[:-1]])
        np.testing.assert_allclose(xyz[:, 2] - np.minimum(xyz[:, 0], xyz[:, 1]), 1.5, atol=0.001)
        self.assertEqual(rows[-1]["z"], "12.000")
        self.assertEqual(rows[-1]["look_at_z"], "1.750")

    def test_missing_coverage_stops_at_draw_budget(self):
        self.cfg.deployment.tx_box.x_min = 2
        self.cfg.deployment.tx_box.x_max = 3
        self.cfg.deployment.max_tx_sampling_draws = 37
        with self.assertRaisesRegex(RuntimeError, "after 37 draws"):
            deployment.sample_box_polygon(self.cfg, np.random.default_rng(71))

    def test_scene_default_does_not_load_terrain(self):
        del self.cfg.deployment.height_reference
        with patch.object(TerrainSurface, "__init__", side_effect=AssertionError("unexpected mesh load")):
            points, _ = deployment.sample_box_polygon(self.cfg, np.random.default_rng(71))
        np.testing.assert_array_equal(points[:, 2], np.full(100, 1.5))

    def test_unsupported_modes_and_nonpositive_height_are_rejected(self):
        for mode in ["box", "rooftop"]:
            self.cfg.deployment.mode = mode
            with self.assertRaisesRegex(ValueError, "requires polygon"):
                deployment.load_placement_terrain(self.cfg)
        self.cfg.deployment.mode = "polygon"
        self.cfg.deployment.tx_height = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            deployment.sample_scene_polygon(self.cfg, np.random.default_rng(71))


class TerrainSwitchTests(unittest.TestCase):
    def test_adapter_is_optional_and_receives_resolved_settings(self):
        from src.module_1_scene_gen import geo2sigmap_adapter
        with tempfile.TemporaryDirectory() as d:
            with patch.object(geo2sigmap_adapter, "generate_scene", return_value={"terrain": "aligned"}) as generate, \
                    patch.object(run_g2sm.subprocess, "run") as cli:
                result = run_g2sm.run_scenegen_box("test", {}, d,
                    generation_settings={"lidar_building_heights": True, "seed": 19})
            self.assertEqual(result, {"terrain": "aligned"})
            self.assertEqual(generate.call_args.args[2]["seed"], 19)
            cli.assert_not_called()

    def test_incomplete_directory_is_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "test"
            folder.mkdir()
            (folder / "partial.ply").touch()
            with patch.object(run_g2sm.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "Incomplete scene"):
                    run_g2sm.run_scenegen_box("test", {}, d)
            run.assert_not_called()
            self.assertTrue((folder / "partial.ply").exists())

    def test_cli_success_without_scene_is_reported_as_failure(self):
        with tempfile.TemporaryDirectory() as d, patch.object(run_g2sm.subprocess, "run"):
            with self.assertRaises(SystemExit) as error:
                run_g2sm.run_scenegen_box("test", dict(min_lon=1, min_lat=2, max_lon=3, max_lat=4), d)
            self.assertEqual(error.exception.code, 1)

    def test_config_switches_reach_cli_with_legacy_defaults(self):
        for switches in ({}, {"enable_lidar_terrain": False, "enable_dem_terrain": True},
                         {"enable_lidar_terrain": True, "enable_dem_terrain": False}):
            with self.subTest(switches=switches), tempfile.TemporaryDirectory() as d:
                gps = Path(d) / "gps.yaml"
                gps.write_text('test:\n  box: {min_lon: 1, min_lat: 2, max_lon: 3, max_lat: 4}\n')
                cfg = OmegaConf.create(dict(project_dir=d, gps_location_path=str(gps),
                    selected_id="test", output_root=d, **switches))
                with patch.object(run_g2sm.subprocess, "run",
                        side_effect=lambda *args, **kwargs: (Path(kwargs["cwd"]) / "scene.xml").write_text("<scene/>")) as run, contextlib.redirect_stdout(io.StringIO()):
                    run_g2sm.main.__wrapped__(cfg)
                command = run.call_args.args[0]
                self.assertEqual("--enable-lidar-terrain" in command, switches.get("enable_lidar_terrain", True))
                self.assertEqual("--enable-dem-terrain" in command, switches.get("enable_dem_terrain", True))


if __name__ == "__main__":
    unittest.main()
