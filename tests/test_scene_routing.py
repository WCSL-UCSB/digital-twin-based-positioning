"""Routing regressions; run with python -m unittest discover -s tests.

Only NumPy, Hydra/OmegaConf, dotenv, and matplotlib are needed. Mock the
geometry engines and plots so tests never trace rays or touch shared data.
"""

import contextlib
import csv
import io
from pathlib import Path
import struct
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from omegaconf import OmegaConf

from src.module_1_scene_gen import extract_building_footprints as footprints
from src.module_1_scene_gen import tx_rx_location_csv_gen as deployment
from src.utils.scene_io import resolve_scene_path, buffered_footprint_path


class SceneRoutingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.scene_dir = self.root / "geometry" / "boston_cambridge_street"
        self.scene_dir.mkdir(parents=True)
        self.xml = self.scene_dir / "scene.xml"
        self.xml.touch()
        self.cfg = OmegaConf.create({
            "project_dir": str(self.root),
            "output_root": str(self.root / "unused_scenes"),
            "selected_id": "boston_cambridge_street_bs2_diff",
            "scene_xml_path": str(self.xml),
            "ray_tracing_params": {"seed": 71},
            "deployment": {
                "mode": "box_polygon", "num_samples": 20, "tx_height": 1.5,
                "tx_box": {"x_min": -10, "x_max": 10, "y_min": -10,
                           "y_max": 10, "z_min": 1.5, "z_max": 1.5},
                "building_exclusion": {"buffer_m": 0.5, "oversample_factor": 4},
                "base_stations": [{"name": "bs1", "position": [0, 0, 15],
                                   "look_at": [10, 0, 1.5]}],
            },
        })
        self.square = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2], [-2, -2]])
        stdout = contextlib.redirect_stdout(io.StringIO())
        stdout.__enter__()
        self.addCleanup(stdout.__exit__, None, None, None)

    def metadata_pair(self, name):
        (self.scene_dir / (name + ".geojson")).write_text("{}")
        (self.scene_dir / (name + "_tileinfo.geojson")).write_text("{}")

    def save_footprints(self):
        np.savez(self.scene_dir / "building_footprints.npz",
                 num_buildings=np.array([1]), poly_0000=self.square, buffer_m=np.array([0.5]))

    def scene_loader(self):
        bbox = SimpleNamespace(min=SimpleNamespace(x=-10, y=-10),
                               max=SimpleNamespace(x=10, y=10))
        loader = Mock(return_value=SimpleNamespace(
            mi_scene=SimpleNamespace(bbox=lambda: bbox)))
        package = ModuleType("sionna")
        rt = ModuleType("sionna.rt")
        rt.load_scene = loader
        package.rt = rt
        return loader, patch.dict("sys.modules", {"sionna": package, "sionna.rt": rt})

    def test_extraction_then_deployment_preserves_csv_identity(self):
        self.metadata_pair("boston_cambridge_street")
        polygon = SimpleNamespace(exterior=SimpleNamespace(coords=self.square), area=16)
        loader, engine = self.scene_loader()
        with engine, patch.object(footprints, "extract_footprints_bostontwin",
                                  return_value=([polygon], [polygon], ["building"])) as extract, \
                patch.object(footprints, "save_verification_image"):
            footprints.main.__wrapped__(self.cfg)
        extract.assert_called_once_with(str(self.scene_dir), "boston_cambridge_street",
                                        buffer_m=0.5)
        loader.assert_called_once_with(str(self.xml))
        self.assertTrue(Path(buffered_footprint_path(self.scene_dir, 0.5)).is_file())

        with patch.object(deployment, "save_deployment_image"):
            deployment.main.__wrapped__(self.cfg)
        csv_dir = self.root / "configs" / "gps_coordinates"
        csv_path = csv_dir / "boston_cambridge_street_bs2_diff_tx_rx.csv"
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 21)
        self.assertEqual(rows[-1]["name"], "bs1")
        self.assertEqual(rows[-1]["type"], "rx")
        points = np.array([[float(r["x"]), float(r["y"])] for r in rows[:-1]])
        polygons = deployment.load_footprints(str(self.scene_dir))
        self.assertTrue(deployment.batch_polygon_check(points[:, 0], points[:, 1], polygons).all())
        self.assertFalse((csv_dir / "boston_cambridge_street_tx_rx.csv").exists())
        self.assertEqual(self.cfg.selected_id, "boston_cambridge_street_bs2_diff")
        self.assertFalse(Path(self.cfg.output_root).exists())

    def test_scene_wide_sampling_loads_explicit_named_xml(self):
        self.save_footprints()
        named_xml = self.scene_dir / "custom_geometry.xml"
        named_xml.touch()
        self.cfg.scene_xml_path = str(named_xml)
        loader, engine = self.scene_loader()
        with engine:
            positions, polygons = deployment.sample_scene_polygon(
                self.cfg, np.random.default_rng(71))
        loader.assert_called_once_with(str(named_xml))
        self.assertEqual(positions.shape, (20, 3))
        self.assertTrue(deployment.batch_polygon_check(
            positions[:, 0], positions[:, 1], polygons).all())

    def test_rooftop_sampling_uses_explicit_scene_directory(self):
        mesh_dir = self.scene_dir / "mesh"
        mesh_dir.mkdir()
        header = ("ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "element face 1\nproperty list uchar int vertex_indices\nend_header\n")
        with (mesh_dir / "building_rooftop.ply").open("wb") as f:
            f.write(header.encode())
            f.write(struct.pack("<9f", 0, 0, 10, 1, 0, 10, 0, 1, 10))
            f.write(struct.pack("<B3i", 3, 0, 1, 2))
        positions = deployment.sample_scene_rooftop(self.cfg, np.random.default_rng(71))
        self.assertEqual(positions.shape, (20, 3))
        triangles = deployment.load_rooftop_triangles(str(self.scene_dir), 1.5)
        self.assertEqual(len(triangles), 1)
        self.assertFalse(deployment.points_on_rooftop_mask(positions[:, :2], triangles).any())

    def test_relative_xml_resolves_from_project_not_working_directory(self):
        self.cfg.scene_xml_path = "geometry/boston_cambridge_street/scene.xml"
        self.assertEqual(resolve_scene_path(self.cfg), str(self.xml))

    def test_no_override_uses_output_root_then_default(self):
        self.cfg.scene_xml_path = None
        self.cfg.output_root = "custom_scenes"
        expected = self.root / "custom_scenes" / self.cfg.selected_id / "scene.xml"
        self.assertEqual(resolve_scene_path(self.cfg), str(expected))
        del self.cfg.scene_xml_path
        del self.cfg.output_root
        expected = self.root / "data" / "scenes" / self.cfg.selected_id / "scene.xml"
        self.assertEqual(resolve_scene_path(self.cfg), str(expected))

    def test_relocated_bostontwin_scene_keeps_original_metadata_names(self):
        self.metadata_pair("original_scene")
        self.assertEqual(footprints._detect_scene_source(str(self.xml)),
                         ("bostontwin", "original_scene"))

    def test_ambiguous_metadata_is_rejected(self):
        self.metadata_pair("first")
        self.metadata_pair("second")
        with self.assertRaisesRegex(ValueError, "Ambiguous BostonTwin metadata"):
            footprints._detect_scene_source(str(self.xml))
        self.assertEqual(footprints._detect_scene_source(str(self.scene_dir / "second.xml")),
                         ("bostontwin", "second"))

    def test_mesh_scene_is_detected_without_bostontwin_metadata(self):
        mesh_dir = self.scene_dir / "mesh"
        mesh_dir.mkdir()
        (mesh_dir / "building_rooftop.ply").touch()
        self.assertEqual(footprints._detect_scene_source(str(self.xml)), ("mesh", None))


if __name__ == "__main__":
    unittest.main()
