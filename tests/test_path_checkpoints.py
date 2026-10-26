"""Checkpoint regressions using temporary inputs and a deterministic fake solver.

Run in the Sionna environment with python -m unittest discover -s tests -v.
No scene engine or shared dataset is used by these tests.
"""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from src.module_1_scene_gen import path_gen as tracing


class Array:
    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self.value


class Scene:
    def __init__(self):
        self.devices = {}

    def add(self, device):
        self.devices[device.name] = device

    def remove(self, name):
        del self.devices[name]


class PathCheckpointTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.scene = self.root / 'scene.xml'
        self.scene.write_text('<scene version="3.0.0"/>')
        self.gps = self.root / 'gps'
        self.gps.mkdir()
        (self.gps / 'gps_locations.yaml').write_text('{}')
        self.csv = self.gps / 'audit_tx_rx.csv'
        with self.csv.open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'type', 'x', 'y', 'z', 'look_at_x', 'look_at_y', 'look_at_z'])
            for i in range(5):
                writer.writerow([f'tx{i}', 'tx', i + 1, 0, 1.5, 0, 0, 0])
            writer.writerow(['rx', 'rx', 0, 0, 10, 1, 0, 0])
        array = dict(num_rows=1, num_cols=1, pattern='iso', polarization='V')
        self.cfg = OmegaConf.create(dict(
            selected_id='audit', scenario_name='audit', project_dir=str(self.root),
            gps_location_path=str(self.gps / 'gps_locations.yaml'),
            scene_xml_path=str(self.scene), frequency=14e9,
            tx_array=array, rx_array=array,
            ray_tracing_params=dict(max_depth=1, synthetic_array=True, batch_size=2, seed=71),
            checkpoint=dict(every_batches=1, resume=True, keep_shards=True),
        ))
        self.shard_root = self.root / 'data/path_data/_shards/audit'
        self.calls = []
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def fingerprint(self):
        return tracing._run_fingerprint(
            {'seed': 71}, 5, 1, 2, 'audit', 14e9, str(self.csv),
            scene_path=str(self.scene), cfg=self.cfg)

    def solve(self, *, scene, **kwargs):
        txs = [dev for dev in scene.devices.values() if dev.name.startswith('tx')]
        self.calls.append([dev.name for dev in txs])
        values = np.array([dev.position[0] for dev in txs], dtype=np.float32)[None, :, None]
        scalar = Array(values)
        return SimpleNamespace(
            a=(Array(values[:, None, :, None, :]), Array(np.zeros_like(values[:, None, :, None, :]))),
            tau=Array(values * 1e-9), theta_r=scalar, phi_r=scalar,
            theta_t=scalar, phi_t=scalar,
            valid=Array(np.ones(values.shape, dtype=bool)),
            interactions=Array(np.zeros((1,) + values.shape, dtype=np.int16)),
        )

    def run_main(self):
        with patch.object(tracing, 'load_scene', side_effect=lambda *a, **k: Scene()), \
                patch.object(tracing, 'PlanarArray', side_effect=lambda **k: k), \
                patch.object(tracing, 'Transmitter', side_effect=lambda **k: SimpleNamespace(**k)), \
                patch.object(tracing, 'Receiver', side_effect=lambda **k: SimpleNamespace(**k)), \
                patch.object(tracing, 'PathSolver', return_value=self.solve), \
                patch.object(tracing, 'write_sidecar'):
            tracing.main.__wrapped__(self.cfg)

    def runs(self):
        return sorted(self.shard_root.glob('*/run_*'))

    def assert_full_output(self):
        outputs = list((self.root / 'data/path_data').glob('*audit_paths.npz'))
        self.assertTrue(outputs)
        with np.load(max(outputs), allow_pickle=True) as data:
            self.assertEqual(data['tau_ns'].shape, (1, 5, 1))
            np.testing.assert_array_equal(data['a'][0, :, 0].real, np.arange(1, 6))
            np.testing.assert_array_equal(data['tx_pos'][:, 0], np.arange(1, 6))
            self.assertFalse(data['metadata'].item()['partial'])

    def test_padding_and_zero_path_links_use_solver_validity(self):
        counts = np.array([2, 1, 0, 2, 1])

        def solve(*, scene, **kwargs):
            txs = [d for d in scene.devices.values() if d.name.startswith('tx')]
            indices = np.array([int(d.name[2:]) for d in txs])
            width = int(counts[indices].max())
            valid = np.arange(width)[None, None, :] < counts[indices][None, :, None]
            # Invalid slots deliberately contain nonzero junk, as can engine buffers.
            values = np.broadcast_to(np.arange(width)[None, None, :] + 1., valid.shape).copy()
            return SimpleNamespace(
                a=(Array(values[:, None, :, None, :]), Array(values[:, None, :, None, :])),
                tau=Array(np.where(valid, values * 1e-9, -1.0)),
                theta_r=Array(values), phi_r=Array(values),
                theta_t=Array(values), phi_t=Array(values), valid=Array(valid),
                interactions=Array(np.ones((1,) + valid.shape, dtype=np.int16)),
            )

        self.solve = solve
        for no_paths in (False, True):
            with self.subTest(no_paths=no_paths):
                if no_paths:
                    counts[:] = 0
                self.cfg.checkpoint.resume = False
                self.run_main()
                output = max((self.root / 'data/path_data').glob('*_paths.npz'))
                with np.load(output, allow_pickle=True) as data:
                    np.testing.assert_array_equal(data['num_valid_paths'][0], counts)
                    self.assertEqual(data['tau_ns'].shape, (1, 5, int(counts.max())))
                    for t, count in enumerate(counts):
                        np.testing.assert_allclose(data['tau_ns'][0, t, :count], np.arange(count) + 1.)
                        for key in ('a', 'tau_ns', 'theta_r', 'phi_r', 'theta_t', 'phi_t'):
                            np.testing.assert_array_equal(data[key][0, t, count:], 0)
                        np.testing.assert_array_equal(data['interactions'][:, 0, t, count:], 0)

    def test_project_relative_gps_and_coordinate_file(self):
        self.cfg.gps_location_path = 'gps'
        for value in ('gps/audit_tx_rx.csv', '${project_dir}/gps/audit_tx_rx.csv'):
            with self.subTest(coordinate_file=value):
                (self.gps / 'gps_locations.yaml').write_text(
                    'audit:\n  coordinate_file: "' + value + '"\n')
                self.cfg.checkpoint.resume = False
                self.run_main()
                self.assert_full_output()

    def test_invalid_settings_fail_before_loading_scene(self):
        for key, value in [('ray_tracing_params.synthetic_array', False),
                           ('ray_tracing_params.batch_size', 0),
                           ('tx_array.num_cols', 64), ('rx_array.polarization', 'VH'),
                           ('frequency', float('nan'))]:
            original = OmegaConf.create(self.cfg)
            with self.subTest(setting=key), patch.object(tracing, 'load_scene') as loader:
                OmegaConf.update(self.cfg, key, value)
                with self.assertRaises(ValueError):
                    tracing.main.__wrapped__(self.cfg)
                loader.assert_not_called()
            self.cfg = original

    def test_invalid_layouts_fail_before_loading_scene(self):
        original = self.csv.read_text()
        for old, new in [('tx1,tx,', 'tx0,tx,'), ('tx0,tx,', 'tx0,user,'),
                         ('tx0,tx,1,', 'tx0,tx,nan,'),
                         ('tx0,tx,1,0,1.5,0,0,0', 'tx0,tx,1,0,1.5,0,,0'),
                         ('rx,rx,0,0,10,1,0,0', 'rx,rx,0,0,10,0,0,10'),
                         ('rx,rx,', 'rx,tx,')]:
            with self.subTest(change=new), patch.object(tracing, 'load_scene') as loader:
                self.csv.write_text(original.replace(old, new))
                with self.assertRaises(ValueError):
                    tracing.main.__wrapped__(self.cfg)
                loader.assert_not_called()
        self.csv.write_text(original)

    def test_missing_bs_look_at_fails_before_loading_scene(self):
        self.csv.write_text(self.csv.read_text().replace(
            'rx,rx,0,0,10,1,0,0', 'rx,rx,0,0,10,,,'))
        with patch.object(tracing, 'load_scene') as loader:
            with self.assertRaisesRegex(ValueError, "BS/RX 'rx'.*look_at"):
                tracing.main.__wrapped__(self.cfg)
            loader.assert_not_called()

    def test_finalizations_within_one_second_preserve_both_outputs(self):
        with patch.object(tracing.time, 'strftime', return_value='20260101_000000'):
            self.run_main()
            first = next((self.root / 'data/path_data').glob('*_paths.npz'))
            before = first.read_bytes()
            self.run_main()
        self.assertEqual(len(list((self.root / 'data/path_data').glob('*_paths.npz'))), 2)
        self.assertEqual(first.read_bytes(), before)

    def test_fingerprint_changes_with_positions_and_orientations(self):
        original = self.csv.read_text()
        self.csv.write_text(original)  # keep line endings identical across edits
        before = self.fingerprint()
        for old, new in [('tx0,tx,1,', 'tx0,tx,901,'),
                         ('rx,rx,0,0,10,1,0,0', 'rx,rx,0,0,10,0,1,0')]:
            with self.subTest(change=old):
                self.csv.write_text(original.replace(old, new))
                self.assertNotEqual(before, self.fingerprint())
        self.csv.write_text(original)
        self.assertEqual(before, self.fingerprint())

    def test_fingerprint_covers_arrays_materials_and_scene_xml(self):
        before = self.fingerprint()
        for key, val in [('tx_array.pattern', 'dipole'), ('rx_array.polarization', 'H'),
                         ('material_config.overrides.concrete.thickness', 0.3)]:
            original = OmegaConf.create(self.cfg)
            with self.subTest(change=key):
                OmegaConf.update(self.cfg, key, val)
                self.assertNotEqual(before, self.fingerprint())
            self.cfg = original
        self.scene.write_text('<scene version="3.0.0"><shape type="cube"/></scene>')
        self.assertNotEqual(before, self.fingerprint())

    def test_fingerprint_covers_included_xml_and_mesh_contents(self):
        sub = self.root / 'sub'
        sub.mkdir()
        mesh = sub / 'mesh.ply'
        mesh.write_bytes(b'mesh version 1')
        included = sub / 'included.xml'
        included.write_text('<scene><default name="mesh" value="mesh.ply"/>'
                            '<shape type="ply"><string name="filename" value="$mesh"/></shape></scene>')
        self.scene.write_text('<scene><include filename="sub/included.xml"/></scene>')
        before = self.fingerprint()
        mesh.write_bytes(b'mesh version 2')
        self.assertNotEqual(before, self.fingerprint())
        mesh.write_bytes(b'mesh version 1')
        self.assertEqual(before, self.fingerprint())
        included.write_text(included.read_text().replace('type="ply"', 'type="obj"'))
        self.assertNotEqual(before, self.fingerprint())
        mesh.unlink()
        with self.assertRaises(FileNotFoundError):
            self.fingerprint()

    def test_cadence_and_downstream_changes_allow_resume(self):
        before = self.fingerprint()
        self.cfg.checkpoint.every_batches = 4
        self.cfg.checkpoint.resume = False
        self.cfg.channel_params = {'bandwidth': 1e8}
        self.assertEqual(before, self.fingerprint())

    def test_fingerprint_covers_nested_ground_material_curves(self):
        materials = self.root / 'configs/materials/data'
        materials.mkdir(parents=True)
        curve = materials / 'ground.csv'
        curve.write_text('1000,2.0\n')
        fake_module = self.root / 'src/module_1_scene_gen/path_gen.py'
        fake_module.parent.mkdir(parents=True)
        with patch.object(tracing, '__file__', str(fake_module)):
            before = self.fingerprint()
            curve.write_text('1000,3.0\n')
            self.assertNotEqual(before, self.fingerprint())
            curve.write_text('1000,2.0\n')
            self.assertEqual(before, self.fingerprint())

    def test_fresh_run_with_different_cadence_preserves_old_shards(self):
        self.run_main()
        old = self.runs()[0]
        before = {p.name: p.read_bytes() for p in old.glob('*.npz')}
        self.cfg.checkpoint.resume = False
        self.cfg.checkpoint.every_batches = 3
        self.calls.clear()
        self.run_main()
        self.assertEqual(len(self.runs()), 2)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(before, {p.name: p.read_bytes() for p in old.glob('*.npz')})
        newest = self.runs()[-1]
        self.assertEqual(len(list(newest.glob('*.npz'))), 1)
        self.assert_full_output()
        self.cfg.checkpoint.resume = True
        self.calls.clear()
        self.run_main()
        self.assertEqual(self.calls, [])
        self.assertEqual(len(self.runs()), 2)

    def test_resume_fills_holes_with_consecutive_shards(self):
        self.run_main()
        run = self.runs()[0]
        for path in run.glob('*.npz'):
            with np.load(path) as data:
                keep = int(data['bi_start']) == 1
            if not keep:
                path.unlink()
        self.calls.clear()
        self.cfg.checkpoint.every_batches = 3
        self.run_main()
        self.assertEqual(self.calls, [['tx0', 'tx1'], ['tx4']])
        metas = tracing._matching_shards(str(run), run.parent.name)
        self.assertEqual([(m[0], m[1], m[2], m[3]) for m in metas],
                         [(0, 1, 0, 2), (1, 2, 2, 4), (2, 3, 4, 5)])
        self.assert_full_output()

    def test_changed_layout_gets_new_run_and_retraces(self):
        self.run_main()
        before = self.runs()[0]
        self.csv.write_text(self.csv.read_text().replace('tx0,tx,1,', 'tx0,tx,901,'))
        self.calls.clear()
        self.run_main()
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(len(self.runs()), 2)
        self.assertEqual(len({run.parent.name for run in self.runs()}), 2)
        self.assertTrue(before.is_dir())

    def test_finalize_only_uses_newest_matching_run_without_tracing(self):
        self.run_main()
        self.cfg.checkpoint.resume = False
        self.cfg.checkpoint.every_batches = 3
        self.run_main()
        self.cfg.finalize_only = True
        self.calls.clear()
        self.run_main()
        self.assertEqual(self.calls, [])
        self.assertEqual(len(self.runs()), 2)
        self.assert_full_output()

    def test_completed_cleanup_preserves_other_runs(self):
        self.run_main()
        old = self.runs()[0]
        self.cfg.checkpoint.resume = False
        self.cfg.checkpoint.keep_shards = False
        self.run_main()
        self.assertEqual(self.runs(), [old])
        self.assertTrue(list(old.glob('*.npz')))

    def test_overlap_is_rejected_before_merge(self):
        self.run_main()
        run = self.runs()[0]
        source = sorted(run.glob('*.npz'))[0]
        (run / 'shard_b00000_b00003.npz').write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, 'Overlapping checkpoint shards'):
            tracing._scan_done_batches(str(run), run.parent.name)
        self.cfg.finalize_only = True
        with self.assertRaisesRegex(ValueError, 'Overlapping checkpoint shards'):
            self.run_main()

    def test_writer_rejects_nonconsecutive_or_wrong_width_results(self):
        data = {key: np.ones((1, 1, 1)) for key in tracing._PATH_KEYS}
        with self.assertRaisesRegex(ValueError, 'consecutive batch'):
            tracing._write_shard(str(self.root / 'bad'), 0, 3, 0, 3, [data, data], 'fp')
        with self.assertRaisesRegex(ValueError, 'TX range'):
            tracing._write_shard(str(self.root / 'bad'), 0, 1, 0, 2, [data], 'fp')
        self.assertFalse((self.root / 'bad').exists())

    def test_partial_finalization_keeps_prefix_until_missing_batches_resume(self):
        self.run_main()
        run = self.runs()[0]
        for path in run.glob('*.npz'):
            with np.load(path) as data:
                remove = int(data['bi_start']) == 1
            if remove:
                path.unlink()
        self.cfg.finalize_only = True
        self.calls.clear()
        self.run_main()
        self.assertEqual(self.calls, [])
        partial = list((self.root / 'data/path_data').glob('*partial*_paths.npz'))
        self.assertEqual(len(partial), 1)
        with np.load(partial[0], allow_pickle=True) as data:
            self.assertEqual(data['tau_ns'].shape[1], 2)
            self.assertTrue(data['metadata'].item()['partial'])
        self.cfg.finalize_only = False
        self.run_main()
        self.assertEqual(self.calls, [['tx2', 'tx3']])
        self.assert_full_output()

    def test_finalize_only_without_run_does_not_create_checkpoint_directory(self):
        self.cfg.finalize_only = True
        self.run_main()
        self.assertEqual(self.calls, [])
        self.assertFalse(self.shard_root.exists())

    def test_legacy_shards_are_preserved_and_not_selected(self):
        self.shard_root.mkdir(parents=True)
        legacy = self.shard_root / 'shard_b00000_b00001.npz'
        legacy.write_bytes(b'legacy')
        self.run_main()
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(legacy.read_bytes(), b'legacy')


if __name__ == '__main__':
    unittest.main()
