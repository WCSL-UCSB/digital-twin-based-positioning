"""Check array delay offsets against geometry and downstream channel phases."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
from scipy.constants import speed_of_light as C

from src.module_2_path_refinement import filter_and_adjust_paths as refinement
from src.module_3_channel_gen.channel_taps_gen import channel_gen_link


class ElementDelayTests(unittest.TestCase):
    frequency = 14e9

    def element_positions(self, rows, cols, horizontal=0.5, vertical=0.5):
        positions = np.zeros((rows, cols, 3))
        positions[..., 1] = (np.arange(cols) - (cols - 1) / 2) * horizontal * C / self.frequency
        positions[..., 2] = (np.arange(rows)[:, None] - (rows - 1) / 2) * vertical * C / self.frequency
        return positions

    def exact_offsets_ns(self, positions, directions):
        sources = 1e5 * np.asarray(directions)
        distances = np.linalg.norm(sources[None, None, :, :] - positions[:, :, None, :], axis=-1)
        return (distances - np.linalg.norm(sources, axis=-1)) / C * 1e9

    def test_offsets_match_geometric_path_lengths_for_rotated_arrays(self):
        rx = np.array([3., -2., 10.])
        rotation = refinement.build_rotation_matrix(rx, rx + [2., -3., 1.])
        az = np.radians([30., -40., 0.])
        el = np.radians([20., -25., 0.])
        directions = np.column_stack((np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)))
        for rows, cols in [(1, 64), (3, 4), (4, 1)]:
            with self.subTest(shape=(rows, cols)):
                positions = self.element_positions(rows, cols) @ rotation.T
                directions_world = directions @ rotation.T
                theta = np.arccos(directions_world[:, 2])
                phi = np.arctan2(directions_world[:, 1], directions_world[:, 0])
                local = refinement.gcs_angles_to_lcs_cartesian(theta, phi, rotation)
                local_az, local_el = refinement.lcs_to_spherical_broadside(local)
                actual = refinement.compute_element_delays_ns(
                    local_az, local_el, rows, cols, 0.5, 0.5, self.frequency)
                expected = self.exact_offsets_ns(positions, directions_world)
                # Exact spherical distances include negligible curvature;
                # the implementation intentionally uses a plane wave.
                np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=0)
                self.assertLess(actual[-1, -1, 0], actual[0, 0, 0])
                self.assertGreater(actual[-1, -1, 1], actual[0, 0, 1])

    def test_broadside_and_single_element_offsets_are_zero(self):
        offsets = refinement.compute_element_delays_ns(
            np.array([0.]), np.array([0.]), 3, 64, 0.5, 0.5, self.frequency)
        np.testing.assert_array_equal(offsets, 0.)
        offsets = refinement.compute_element_delays_ns(
            np.array([0.4, -0.6]), np.array([0.3, -0.2]), 1, 1, 0.5, 0.5, self.frequency)
        np.testing.assert_array_equal(offsets, 0.)

    def test_channel_carrier_and_envelope_share_the_same_delay_direction(self):
        rows, cols = 3, 4
        az, el = np.radians([31.]), np.radians([-17.])
        offsets = refinement.compute_element_delays_ns(az, el, rows, cols, 0.5, 0.5, self.frequency)
        omega_h = np.pi * np.cos(el) * np.sin(az)
        omega_v = np.pi * np.sin(el)
        for baseband_hz in (0., 300e6, -300e6):
            with self.subTest(baseband_hz=baseband_hz):
                # A complex tone exposes the envelope-delay phase directly.
                taps = channel_gen_link(
                    a=np.ones(1, dtype=complex), tau_center_ns=np.zeros(1),
                    omega_h=omega_h, omega_v=omega_v, tau_per_elem_ns=offsets,
                    f_carrier=self.frequency, M=rows, N=cols, dt=1e-9, T_fixed=1,
                    pulse_fn=lambda t, Tsym, beta: np.exp(2j*np.pi*baseband_hz*t),
                    pulse_Tsym=1e-9, pulse_beta=0.,
                    include_true_time_delay=True, include_spatial_phase=True)
                expected = np.exp(-2j*np.pi*(self.frequency + baseband_hz)*offsets[..., 0]*1e-9)
                np.testing.assert_allclose(taps[..., 0], expected, atol=1e-12, rtol=0)

    def test_refinement_saves_correct_offsets_and_preserves_center_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / 'raw.npz'
            az, el = np.radians([30., -25., 120.]), np.radians([20., -10., 0.])
            values = np.ones((1, 1, 3))
            np.savez(raw, a=values.astype(complex), tau_ns=np.array([[[120., 135., 105.]]]),
                     theta_r=(np.pi/2 - el)[None, None, :], phi_r=az[None, None, :],
                     theta_t=values, phi_t=values, num_valid_paths=np.array([[3]]),
                     interactions=np.zeros((1, 1, 1, 3), dtype=np.int16),
                     rx_names=np.array(['rx']), rx_pos=np.array([[0., 0., 0.]]),
                     rx_look_at=np.array([[1., 0., 0.]]), tx_names=np.array(['tx']),
                     tx_pos=np.array([[10., 10., 10.]]), tx_look_at=np.full((1, 3), np.nan),
                     metadata={'frequency_hz': self.frequency})
            before = raw.read_bytes()
            cfg = OmegaConf.create(dict(project_dir=tmp, scenario_name='delay_test',
                frequency=self.frequency, inputs={'paths_npz': str(raw)},
                rx_array_geometry=dict(num_rows=2, num_cols=3, horizontal_spacing=0.5,
                                       vertical_spacing=0.5, fov_az_deg=120., fov_el_deg=180.)))
            with contextlib.redirect_stdout(io.StringIO()), patch.object(refinement, 'write_sidecar'):
                refinement.main.__wrapped__(cfg)
            output, = (root / 'data/path_data/refined_paths').glob('*.npz')
            with np.load(output, allow_pickle=True) as data:
                np.testing.assert_array_equal(data['num_paths_after_filter'], [[2]])
                np.testing.assert_array_equal(data['min_delay_ns'], [[120.]])
                np.testing.assert_array_equal(data['tau_ns'], [[[0., 15.]]])
                np.testing.assert_array_equal(data['a'], np.ones((1, 1, 2)))
                directions = np.column_stack((np.cos(el[:2])*np.cos(az[:2]),
                                              np.cos(el[:2])*np.sin(az[:2]), np.sin(el[:2])))
                expected = self.exact_offsets_ns(self.element_positions(2, 3), directions)
                actual = data['tau_per_element_ns'][0, 0] - data['tau_ns'][0, 0]
                np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=0)
                self.assertLess(data['tau_per_element_ns'][0, 0, -1, -1, 0], 0.)
                self.assertIn('element_delay_convention', data['metadata'].item())
            self.assertEqual(raw.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
