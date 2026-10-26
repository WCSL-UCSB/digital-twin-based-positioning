"""Scene config/routing regressions without downloads or rendering."""
import contextlib
import io
import json
import zipfile
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from src.module_1_scene_gen import run_g2sm

ROOT = Path(__file__).resolve().parents[1]


class SceneGenerationTests(unittest.TestCase):
    def test_scene_configs_share_physics_without_a_deployment(self):
        with patch.dict(os.environ, {'PROJECT_DIR': str(ROOT)}), initialize_config_dir(
                version_base=None, config_dir=str(ROOT/'configs/scenarios')):
            configs = [compose(config_name=name) for name in
                ('boston', 'nyc_lidar_dem')]
            self.assertEqual(configs[0].ray_tracing_params, configs[1].ray_tracing_params)
            for cfg in configs:
                OmegaConf.resolve(cfg)
                self.assertEqual(cfg.material_config.defaults.scattering_coefficient, 0.5)
                self.assertTrue(all(v.scattering_coefficient == 0.5 for v in cfg.material_config.overrides.values()))
                self.assertFalse(cfg.ray_tracing_params.diffuse_reflection)
                self.assertTrue(cfg.ray_tracing_params.diffraction_lit_region)
                self.assertFalse(cfg.ray_tracing_params.edge_diffraction)
                self.assertEqual(list(cfg.deployment.base_stations), [])
                self.assertEqual(cfg.deployment.mode, 'pending')
                self.assertFalse(cfg.force)
            self.assertNotEqual(configs[0].scene_xml_path, configs[1].scene_xml_path)

    def test_yaml_terrain_switches_reach_generator(self):
        with tempfile.TemporaryDirectory() as d:
            gps = Path(d)/'gps.yaml'
            gps.write_text('test:\n  box: {min_lon: 1, min_lat: 2, max_lon: 3, max_lat: 4}\n')
            cfg = OmegaConf.create(dict(project_dir=d, gps_location_path=str(gps),
                selected_id='test',output_root=d,enable_lidar_terrain=False,enable_dem_terrain=False))
            with patch.object(run_g2sm,'run_scenegen_box',return_value=None) as run, contextlib.redirect_stdout(io.StringIO()):
                run_g2sm.main.__wrapped__(cfg)
            self.assertFalse(run.call_args.kwargs['enable_lidar_terrain'])
            self.assertFalse(run.call_args.kwargs['enable_dem_terrain'])

    def test_adapter_success_records_resolved_config_and_source_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            gps = root / 'gps.yaml'
            gps.write_text('test:\n  box: {min_lon: 1, min_lat: 2, max_lon: 3, max_lat: 4}\n')
            folder = root / 'test'
            folder.mkdir()
            (folder / 'scene.xml').write_text('<scene/>')
            source = root / 'src/module_1_scene_gen'
            source.mkdir(parents=True)
            (source / 'fixture.py').write_text('# source snapshot\n')
            cfg = OmegaConf.create(dict(project_dir=d, gps_location_path=str(gps),
                selected_id='test', output_root=d,
                scene_generation={'lidar_building_heights': True}))
            details = {'source': 'Geo2SigMap', 'terrain': {'vertical_origin_m': 12.0}}
            with patch.object(run_g2sm, 'run_scenegen_box', return_value=details), \
                    patch('src.module_1_scene_gen.scene_run_record.subprocess.check_output', return_value='fixture'), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_g2sm.main.__wrapped__(cfg)
            record = json.loads((folder / 'generation.json').read_text())
            self.assertEqual(record['details'], details)
            self.assertEqual(record['config']['selected_id'], 'test')
            self.assertTrue((folder / 'config_resolved.yaml').is_file())
            with zipfile.ZipFile(folder / 'generation_code.zip') as archive:
                self.assertEqual(archive.read('src/module_1_scene_gen/fixture.py'), b'# source snapshot\n')
                self.assertIn('config_resolved.yaml', archive.namelist())

    def test_explicit_resume_preserves_partial_source_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / 'test'
            folder.mkdir()
            partial = folder / 'source.tif'
            partial.write_bytes(b'cached input')
            def complete(*args, **kwargs):
                (Path(kwargs['cwd']) / 'scene.xml').write_text('<scene/>')
            with patch.object(run_g2sm.subprocess, 'run', side_effect=complete) as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                run_g2sm.run_scenegen_box('test', dict(min_lon=1, min_lat=2, max_lon=3, max_lat=4), d,
                    generation_settings={'resume_incomplete': True})
            run.assert_called_once()
            self.assertEqual(partial.read_bytes(), b'cached input')
            self.assertTrue((folder / 'scene.xml').is_file())

    def test_existing_scene_is_not_replaced_by_resume(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'scene';folder.mkdir();(folder/'scene.xml').write_text('existing')
            with patch.object(run_g2sm.subprocess,'run') as run, contextlib.redirect_stdout(io.StringIO()):
                run_g2sm.run_scenegen_box('scene',{},d,generation_settings={'resume_incomplete':True})
            run.assert_not_called()
            self.assertEqual((folder/'scene.xml').read_text(),'existing')

if __name__=='__main__':
    unittest.main()
