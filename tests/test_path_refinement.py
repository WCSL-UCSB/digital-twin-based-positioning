"""Exercise refinement on explicit raw-path fixtures, without ray tracing."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from src.module_2_path_refinement import filter_and_adjust_paths as stage


class RefinementTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.raw = self.root / 'raw.npz'
        shape = (1, 2, 4)
        self.values = dict(
            a=np.broadcast_to([1+2j, 3+4j, 5+6j, 7+8j], shape).copy(),
            tau_ns=np.broadcast_to([140., 120., 100., 80.], shape).copy(),
            theta_r=np.broadcast_to(np.radians([90., 70., 90., 90.]), shape).copy(),
            phi_r=np.broadcast_to(np.radians([0., 40., 80., 180.]), shape).copy(),
            theta_t=np.full(shape, 1.), phi_t=np.full(shape, 2.),
            interactions=np.zeros((3,) + shape, dtype=np.int16),
            num_valid_paths=np.array([[4, 0]]),
            rx_names=np.array(['rx']), rx_pos=np.array([[0., 0., 0.]]),
            rx_look_at=np.array([[1., 0., 0.]]), tx_names=np.array(['tx0', 'tx1']),
            tx_pos=np.array([[10., 0., 0.], [20., 0., 0.]]),
            tx_look_at=np.full((2, 3), np.nan),
            metadata={'frequency_hz':14e9, 'rx_array':{'polarization':'H'}},
        )
        self.values['interactions'][0, 0, 0, :2] = [1, 4]
        self.cfg = OmegaConf.create(dict(project_dir=str(self.root), scenario_name='audit',
            frequency=14e9, inputs={'paths_npz':str(self.raw)},
            rx_array_geometry=dict(num_rows=2, num_cols=3, horizontal_spacing=.5,
                                   vertical_spacing=.5, fov_az_deg=120., fov_el_deg=60.)))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def run_stage(self):
        np.savez(self.raw, **self.values)
        with patch.object(stage, 'write_sidecar'):
            stage.main.__wrapped__(self.cfg)
        path = max((self.root / 'data/path_data/refined_paths').glob('*.npz'))
        with np.load(path, allow_pickle=True) as f:
            return path, {k:f[k] for k in f.files}

    def test_kept_order_delay_reference_interactions_and_empty_links(self):
        # Ignore padding even when it contains NaNs or no-interaction codes.
        for key in ('a', 'tau_ns', 'theta_r', 'phi_r', 'theta_t', 'phi_t'):
            self.values[key][0, 1] = np.nan
        _, out = self.run_stage()
        np.testing.assert_array_equal(out['num_paths_after_filter'], [[2, 0]])
        np.testing.assert_array_equal(out['tau_ns'], [[[20., 0.], [0., 0.]]])
        np.testing.assert_array_equal(out['min_delay_ns'], [[120., 0.]])
        np.testing.assert_array_equal(out['a'][0, 0], [1+2j, 3+4j])
        np.testing.assert_array_equal(out['interactions'][:, 0, 0], [[1, 4], [0, 0], [0, 0]])
        np.testing.assert_array_equal(out['is_los_kept'], [[False, False]])
        self.assertEqual(out['metadata'].item()['polarization_traced'], 'H')
        self.assertEqual(out['metadata'].item()['link_survival_counts'],
                         dict(total=2, traced=1, front=1, az=1, el=1, final=1))
        np.testing.assert_allclose(out['theta_az'][0, 0], np.radians([0, 40]), atol=1e-14)
        np.testing.assert_allclose(out['theta_el'][0, 0], np.radians([0, 20]), atol=1e-14)

    def test_missing_bs_look_at_fails_without_writing_output(self):
        self.values['rx_look_at'][:] = np.nan
        with self.assertRaisesRegex(ValueError, "BS/RX 'rx'.*look_at"):
            self.run_stage()
        self.assertFalse((self.root / 'data/path_data/refined_paths').exists())

    def test_azimuth_and_elevation_are_separate_full_width_cuts(self):
        self.values['phi_r'][0, 0] = np.radians([59.99, 60.01, 0., 0.])
        self.values['theta_r'][0, 0] = np.pi/2 - np.radians([0., 0., 29.99, 30.01])
        _, out = self.run_stage()
        np.testing.assert_array_equal(out['num_paths_after_filter'], [[2, 0]])
        np.testing.assert_array_equal(out['a'][0, 0], self.values['a'][0, 0, [0, 2]])

    def test_wide_fov_still_rejects_rear_hemisphere(self):
        self.cfg.rx_array_geometry.fov_az_deg = 360.
        self.cfg.rx_array_geometry.fov_el_deg = 180.
        self.values['theta_r'][:] = np.pi/2
        self.values['phi_r'][0, 0] = np.radians([0., 80., 100., 180.])
        _, out = self.run_stage()
        np.testing.assert_array_equal(out['num_paths_after_filter'], [[2, 0]])
        np.testing.assert_array_equal(out['a'][0, 0], self.values['a'][0, 0, :2])

    def test_invalid_look_at_fails(self):
        for look in ([0, 0, 0], [np.nan, 0, 0], [np.inf, 0, 0]):
            with self.subTest(look=look), self.assertRaisesRegex(ValueError, 'look_at'):
                self.values['rx_look_at'][:] = look
                self.run_stage()

    def test_invalid_frequencies_fail(self):
        for bad in (0., -1., np.nan, np.inf, 15e9):
            with self.subTest(config_frequency=bad), self.assertRaises(ValueError):
                self.cfg.frequency = bad
                self.run_stage()
        self.cfg.frequency = 14e9
        for bad in (0., -1., np.nan, np.inf):
            with self.subTest(raw_frequency=bad), self.assertRaises(ValueError):
                self.values['metadata']['frequency_hz'] = bad
                self.run_stage()

    def test_invalid_panel_configuration_fails(self):
        original = OmegaConf.create(self.cfg.rx_array_geometry)
        cases = [('num_rows', 0), ('num_cols', 1.5), ('num_cols', np.nan),
                 ('horizontal_spacing', -0.5), ('vertical_spacing', np.inf),
                 ('fov_az_deg', 0.), ('fov_az_deg', 361.), ('fov_el_deg', 181.),
                 ('fov_el_deg', np.nan)]
        for key, value in cases:
            self.cfg.rx_array_geometry = OmegaConf.create(original)
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                self.cfg.rx_array_geometry[key] = value
                self.run_stage()

    def test_no_survivors_fail_instead_of_returning_success(self):
        self.values['phi_r'][:] = np.pi
        for count in (4, 0):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'No paths survived'):
                self.values['num_valid_paths'][0, 0] = count
                self.run_stage()
        self.assertFalse((self.root / 'data/path_data/refined_paths').exists())

    def test_invalid_valid_paths_fail(self):
        for key in ('a', 'tau_ns', 'theta_r', 'phi_r', 'theta_t', 'phi_t'):
            before = self.values[key].copy()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Nonfinite'):
                self.values[key][0, 0, 0] = np.nan
                self.run_stage()
            self.values[key] = before
        with self.assertRaisesRegex(ValueError, 'Negative raw delay'):
            self.values['tau_ns'][0, 0, 0] = -1
            self.run_stage()

    def test_malformed_path_axes_and_device_tables_fail(self):
        for key, value in [('phi_r', np.zeros((1, 2, 3))),
                           ('interactions', np.zeros((3, 1, 2, 5), dtype=np.int16)),
                           ('tx_names', np.array(['one_name'])),
                           ('rx_pos', np.full((1, 3), np.inf))]:
            before = self.values[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.values[key] = value
                self.run_stage()
            self.values[key] = before

    def test_same_second_outputs_are_preserved(self):
        with patch.object(stage.time, 'strftime', return_value='20260101_000000'):
            first, _ = self.run_stage()
            before = first.read_bytes()
            second, _ = self.run_stage()
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), before)

    def test_vertical_and_general_rotations_are_right_handed(self):
        for broadside in ([0, 0, 1], [0, 0, -1], [1, 2, 3]):
            r = stage.build_rotation_matrix([0, 0, 0], broadside)
            np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(r), 1.)
            np.testing.assert_allclose(r[:, 0], broadside/np.linalg.norm(broadside))

    def test_nearly_vertical_target_keeps_its_yaw(self):
        # A tiny tilt still has a defined yaw. Falling back to world +Y here
        # would rotate the panel 45 degrees relative to the tracing device.
        r = stage.build_rotation_matrix([0, 0, 0], [1e-8, 1e-8, 1])
        np.testing.assert_allclose(r[:, 1], [-2**-.5, 2**-.5, 0.], atol=1e-14)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-14)


if __name__ == '__main__':
    unittest.main()
