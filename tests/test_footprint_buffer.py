"""Buffer selection, validation, and isolation regressions (no shared data)."""

import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from src.module_1_scene_gen import extract_building_footprints as extraction
from src.module_1_scene_gen import tx_rx_location_csv_gen as deployment
from src.utils.scene_io import buffered_footprint_path, footprint_buffer_m


class FootprintBufferTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.scene = self.root / "scene"
        self.scene.mkdir()
        self.legacy = self.scene / "building_footprints.npz"
        self.cfg = OmegaConf.create({
            "project_dir": str(self.root), "selected_id": "deployment",
            "scene_xml_path": str(self.scene / "scene.xml"),
            "deployment": {
                "mode": "box_polygon", "num_samples": 4, "tx_height": 1.5,
                "tx_box": {"x_min": -10, "x_max": 10, "y_min": -10,
                           "y_max": 10, "z_min": 1.5, "z_max": 1.5},
                "building_exclusion": {"buffer_m": 0.5},
                "base_stations": [{"name": "bs", "position": [0, 0, 10],
                                   "look_at": [1, 0, 10]}],
            },
        })
        stdout = contextlib.redirect_stdout(io.StringIO())
        stdout.__enter__()
        self.addCleanup(stdout.__exit__, None, None, None)

    def square(self, buffer):
        lo, hi = -buffer, 1 + buffer
        return np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi], [lo, lo]])

    def write(self, path, buffer, recorded=True):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = {"num_buildings": np.array([1]), "poly_0000": self.square(buffer),
                  "poly_0000_raw": self.square(0)}
        if recorded:
            fields["buffer_m"] = np.array([buffer])
        np.savez(path, **fields)

    def test_stale_legacy_stops_sampling_before_csv_is_written(self):
        self.write(self.legacy, 5)
        before = self.legacy.read_bytes()
        with self.assertRaisesRegex(ValueError, "Footprint buffer mismatch.*requested 0.5"):
            deployment.main.__wrapped__(self.cfg)
        self.assertFalse((self.root / "configs").exists())
        self.assertEqual(self.legacy.read_bytes(), before)

    def test_requested_cache_wins_and_changes_exclusion(self):
        self.write(self.legacy, 5)
        before = self.legacy.read_bytes()
        self.write(buffered_footprint_path(self.scene, 0.5), 0.5)
        small = deployment.load_footprints(str(self.scene), buffer_m=0.5)
        large = deployment.load_footprints(str(self.scene), buffer_m=5)
        x, y = np.array([1.25, 2.0]), np.array([0.5, 0.5])
        np.testing.assert_array_equal(deployment.batch_polygon_check(x, y, small), [False, True])
        np.testing.assert_array_equal(deployment.batch_polygon_check(x, y, large), [False, False])
        self.assertEqual(self.legacy.read_bytes(), before)

    def test_missing_buffer_metadata_cannot_be_used_for_sampling(self):
        self.write(self.legacy, 5, recorded=False)
        with self.assertRaisesRegex(ValueError, "records None"):
            deployment.load_footprints(str(self.scene), buffer_m=0.5)
        self.assertEqual(len(deployment.load_footprints(str(self.scene))), 1)

    def test_matching_legacy_metadata_is_accepted(self):
        self.write(self.legacy, 0.5)
        polygons = deployment.load_footprints(str(self.scene), buffer_m=0.5)
        np.testing.assert_array_equal(polygons[0]["vertices"], self.square(0.5))
        self.assertFalse((self.scene / "footprints").exists())

    def test_cache_filename_does_not_override_wrong_metadata(self):
        self.write(self.legacy, 0.5)
        self.write(buffered_footprint_path(self.scene, 0.5), 5)
        with self.assertRaisesRegex(ValueError, "Footprint buffer mismatch"):
            deployment.load_footprints(str(self.scene), buffer_m=0.5)

    def test_extraction_keeps_legacy_and_other_buffer_outputs(self):
        self.write(self.legacy, 5)
        before = self.legacy.read_bytes()
        (self.scene / "scene.geojson").write_text("{}")
        (self.scene / "scene_tileinfo.geojson").write_text("{}")

        def polygons(scene_dir, scene_name, buffer_m):
            def poly(buffer):
                return SimpleNamespace(exterior=SimpleNamespace(coords=self.square(buffer)),
                                       area=(1 + 2 * buffer) ** 2)
            return [poly(buffer_m)], [poly(0)], ["building"]

        with patch.object(extraction, "extract_footprints_bostontwin", side_effect=polygons), \
                patch.object(extraction, "save_verification_image") as plot:
            extraction.main.__wrapped__(self.cfg)
            first = Path(buffered_footprint_path(self.scene, 0.5))
            first_bytes = first.read_bytes()
            self.cfg.deployment.building_exclusion.buffer_m = 2
            extraction.main.__wrapped__(self.cfg)
        second = Path(buffered_footprint_path(self.scene, 2))
        self.assertEqual(first.read_bytes(), first_bytes)
        self.assertEqual(self.legacy.read_bytes(), before)
        self.assertTrue(second.is_file())
        self.assertEqual(Path(plot.call_args_list[0][0][2]), first.with_suffix(".png"))
        self.assertEqual(Path(plot.call_args_list[1][0][2]), second.with_suffix(".png"))
        with np.load(first) as data:
            self.assertEqual(float(data["buffer_m"][0]), 0.5)
        with np.load(second) as data:
            self.assertEqual(float(data["buffer_m"][0]), 2)
        np.testing.assert_array_equal(deployment.load_footprints(str(self.scene), 0.5)[0]["vertices"],
                                      self.square(0.5))

    def test_omitted_setting_uses_extraction_default_of_one_meter(self):
        del self.cfg.deployment.building_exclusion.buffer_m
        self.assertEqual(footprint_buffer_m(self.cfg), 1)
        self.write(self.legacy, 0.5)
        with self.assertRaisesRegex(ValueError, "requested 1 m"):
            deployment.main.__wrapped__(self.cfg)

    def test_zero_buffer_and_equivalent_values(self):
        self.write(buffered_footprint_path(self.scene, 0), 0)
        polygons = deployment.load_footprints(str(self.scene), buffer_m=0)
        self.assertTrue(deployment.batch_polygon_check(np.array([1.25]), np.array([0.5]), polygons)[0])
        self.assertEqual(buffered_footprint_path(self.scene, 0), buffered_footprint_path(self.scene, -0.0))
        self.assertEqual(buffered_footprint_path(self.scene, 1), buffered_footprint_path(self.scene, 1.0))
        self.assertNotEqual(buffered_footprint_path(self.scene, 0.5),
                            buffered_footprint_path(self.scene, 0.50000001))

    def test_invalid_requested_buffers_are_rejected(self):
        for value in (-1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(buffer=value):
                self.cfg.deployment.building_exclusion.buffer_m = value
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    footprint_buffer_m(self.cfg)
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    deployment.load_footprints(str(self.scene), buffer_m=value)

    def test_missing_cache_has_regeneration_instructions(self):
        with self.assertRaisesRegex(FileNotFoundError, "extract_building_footprints.py"):
            deployment.load_footprints(str(self.scene), buffer_m=0.5)

    def test_plotting_can_use_one_cache_but_requires_choice_for_multiple(self):
        self.write(buffered_footprint_path(self.scene, 0.5), 0.5)
        self.assertEqual(len(deployment.load_footprints(str(self.scene))), 1)
        self.write(buffered_footprint_path(self.scene, 2), 2)
        with self.assertRaisesRegex(ValueError, "specify buffer_m"):
            deployment.load_footprints(str(self.scene))


if __name__ == "__main__":
    unittest.main()
