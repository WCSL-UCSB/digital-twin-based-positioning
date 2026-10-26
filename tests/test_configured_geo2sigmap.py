"""Explicit scene source routing without network requests or city downloads."""
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.module_1_scene_gen.configured_geo2sigmap import generate_scene


class ConfiguredSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.box = dict(min_lon=-74, min_lat=40, max_lon=-73.99, max_lat=40.01)
        self.original_lookup = object()
        self.initializer = Mock()
        self.features = Mock(return_value=SimpleNamespace(to_json=lambda: '{"features":[]}'))
        self.settings = SimpleNamespace(use_cache=False, cache_folder="original")
        self.scene = Mock(side_effect=self.build_scene)
        self.core = SimpleNamespace(init_overture_lookup=self.initializer,
            Scene=Mock(return_value=self.scene), ox=SimpleNamespace(settings=self.settings,
                features=SimpleNamespace(features_from_bbox=self.features)))
        self.utils = SimpleNamespace(OVERTURE_LOOKUP=self.original_lookup)
        package = ModuleType("scene_generation")
        package.core, package.utils = self.core, self.utils
        lookup_module = ModuleType("scene_generation.overture_height_lookup")
        self.buildings = Mock(empty=False)
        self.buildings.__len__ = Mock(return_value=2)
        self.parts = Mock(empty=True)
        self.parts.__len__ = Mock(return_value=0)
        self.lookup = SimpleNamespace(buildings=self.buildings, parts=self.parts)
        self.factory = Mock(return_value=self.lookup)
        lookup_module.OvertureHeightLookup = self.factory
        self.enterContext(patch.dict("sys.modules", {"scene_generation": package,
            "scene_generation.overture_height_lookup": lookup_module}))
        self.cwd = os.getcwd()

    def build_scene(self, **kwargs):
        self.core.init_overture_lookup()
        self.core.ox.features.features_from_bbox()
        (self.out / "scene.xml").write_text("<scene/>")

    def assert_restored(self):
        self.assertIs(self.core.init_overture_lookup, self.initializer)
        self.assertIs(self.utils.OVERTURE_LOOKUP, self.original_lookup)
        self.assertIs(self.core.ox.features.features_from_bbox, self.features)
        self.assertEqual((self.settings.use_cache, self.settings.cache_folder), (False, "original"))
        self.assertEqual(os.getcwd(), self.cwd)

    def test_osm_disables_overture_and_preserves_source_snapshot(self):
        result = generate_scene(self.box, self.out, {"height_source": "osm"}, False, False)
        self.factory.assert_not_called()
        self.initializer.assert_not_called()
        self.assertEqual(result["height_source"], "osm")
        self.assertIsNone(result["overture_release"])
        self.assertTrue((self.out / "source_inputs/osm_buildings.geojson").is_file())
        self.assertFalse(self.scene.call_args.kwargs["lidar_calibration"])
        self.assert_restored()

    def test_overture_requires_and_forwards_explicit_release(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            generate_scene(self.box, self.out, {"height_source": "overture"}, False, False)
        result = generate_scene(self.box, self.out,
            {"height_source": "overture", "overture_release": "2026-10-21.0"}, False, False)
        self.factory.assert_called_once_with(bbox=(-74., 40., -73.99, 40.01),
                                            release_url="2026-10-21.0", verbose=True)
        self.assertEqual(result["overture_buildings_retrieved"], 2)
        self.buildings.to_file.assert_called_once()
        self.assert_restored()

    def test_failed_lookup_does_not_fall_back_to_mislabeled_osm(self):
        self.buildings.empty = True
        with self.assertRaisesRegex(RuntimeError, "No Overture"):
            generate_scene(self.box, self.out,
                {"height_source": "overture", "overture_release": "2026-10-21.0"}, False, False)
        self.scene.assert_not_called()
        self.assert_restored()

    def test_generator_failure_restores_package_state(self):
        self.scene.side_effect = RuntimeError("fixture failure")
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            generate_scene(self.box, self.out, {"height_source": "osm"}, False, False)
        self.assert_restored()

    def test_lidar_routes_to_aligned_adapter_with_overture_disabled(self):
        def adapter(*args):
            self.assertIsNone(self.utils.OVERTURE_LOOKUP)
            return {"lidar_building_count": 1}
        with patch("src.module_1_scene_gen.geo2sigmap_adapter.generate_scene", side_effect=adapter) as run:
            result = generate_scene(self.box, self.out, {"height_source": "lidar"}, True, True)
        self.assertEqual(run.call_args.args[-2:], (True, True))
        self.assertEqual(result["lidar_building_count"], 1)
        self.assert_restored()

if __name__ == "__main__":
    unittest.main()
