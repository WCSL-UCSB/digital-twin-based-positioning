"""Independent pulse spectra, channel phase, window and validation checks."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
from scipy.integrate import quad
from src.module_3_channel_gen import channel_taps_gen as stage


class PulseTests(unittest.TestCase):
    def test_rc_rrc_against_independent_frequency_integrals(self):
        # Invert the raised-cosine spectrum (or its square root), independently
        # of the time-domain implementation and its special-case limits.
        for beta in (.2, .5, 1.):
            edge, end = (1-beta)/2, (1+beta)/2
            for fn, root in ((stage.rc_pulse, False), (stage.rrc_pulse, True)):
                u = np.array([0., .01, .2, -.31, .83, 1/(4*beta), 1/(2*beta), 3.1])
                expected = []
                for x in u:
                    def integrand(f):
                        value = 1. if f <= edge else .5*(1+np.cos(np.pi*(f-edge)/beta))
                        return 2*(np.sqrt(value) if root else value)*np.cos(2*np.pi*f*x)
                    expected.append(quad(integrand, 0, end, points=[edge], epsabs=1e-12)[0])
                for period in (1., 3.33e-9, 1e-12):
                    with self.subTest(beta=beta, pulse=fn.__name__, period=period):
                        np.testing.assert_allclose(fn(u*period, period, beta), expected, atol=2e-12, rtol=0)

    def test_singularities_are_continuous_even_and_scalar_safe(self):
        for fn, divisor in ((stage.rc_pulse, 2), (stage.rrc_pulse, 4)):
            for beta in (.2, .5, 1.):
                pole = 1/(divisor*beta)
                u = np.array([pole-1e-6, pole, pole+1e-6])
                actual = fn(u*3.33e-9, 3.33e-9, beta)
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, actual[1], atol=3e-6, rtol=0)
                np.testing.assert_allclose(fn(-u*3.33e-9, 3.33e-9, beta), actual, atol=1e-12)
                self.assertAlmostEqual(float(fn(pole*3.33e-9, 3.33e-9, beta)), actual[1])
        self.assertAlmostEqual(float(stage.rrc_pulse(0., 3.33e-9, 1.)), 4/np.pi)
        self.assertAlmostEqual(float(stage.rc_pulse(3.33e-9/2, 3.33e-9, 1.)), .5)

    def test_zero_rolloff_is_sinc(self):
        u = np.linspace(-3, 3, 31)
        for fn in (stage.rc_pulse, stage.rrc_pulse):
            np.testing.assert_allclose(fn(u*1e-9, 1e-9, 0), np.sinc(u))


class SynthesisTests(unittest.TestCase):
    def test_path_sum_and_ttd_spatial_phase_switches(self):
        a = np.array([1e-6+2e-6j, -3e-6j])
        tau = np.array([0., 1.6])
        offsets = np.array([[[-.03, .01], [.03, -.01]]])
        f = 14e9
        omega = -2*np.pi*f*(offsets[0, 1]-offsets[0, 0])*1e-9
        for ttd in (False, True):
            for spatial in (False, True):
                actual = stage.channel_gen_link(a, tau, omega, np.zeros(2), tau+offsets,
                    f, 1, 2, 1e-9, 8, stage.get_pulse_fn('sinc'), 1e-9, 0., ttd, spatial)
                expected = np.zeros((1, 2, 8), complex)
                for n in range(2):
                    for p in range(2):
                        delay = tau[p] + (offsets[0, n, p] if ttd else 0)
                        phase_delay = tau[p] + (offsets[0, n, p] if spatial else 0)
                        expected[0, n] += a[p]*np.exp(-2j*np.pi*f*phase_delay*1e-9)*np.sinc(np.arange(8)-delay)
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-19)

    def test_coherent_cancellation_and_empty_link(self):
        for a in (np.array([1., -1.]), np.array([], complex)):
            k = len(a)
            h = stage.channel_gen_link(a, np.zeros(k), np.zeros(k), np.zeros(k), np.zeros((1, 1, k)),
                14e9, 1, 1, 1e-9, 4, stage.get_pulse_fn('sinc'), 1e-9, 0., True, True)
            np.testing.assert_array_equal(h, np.zeros((1, 1, 4)))


class ChannelEntryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root/'calibrated.npz'
        self.values = dict(a=np.array([[[1e-6, 2e-6, 3e-6, np.nan], [np.nan]*4]], complex),
            tau_ns=np.array([[[0., 3.9, 4., np.nan], [np.nan]*4]]),
            omega_h=np.zeros((1, 2, 4)), omega_v=np.zeros((1, 2, 4)),
            num_paths_after_filter=np.array([[3, 0]]), min_delay_ns=np.array([[100., np.nan]]),
            rx_names=np.array(['bs']), tx_names=np.array(['ue0', 'ue1']), rx_pos=np.zeros((1, 3)),
            tx_pos=np.array([[10., 0., 0.], [20., 0., 0.]]), rx_look_at=np.array([[1., 0., 0.]]),
            is_los_kept=np.array([[False, False]]), snr_raw_db=np.array([[10., -np.inf]]),
            metadata={'source_metadata': {'frequency_hz': 14e9, 'rx_array': dict(num_rows=1,
                num_cols=1, horizontal_spacing_wl=.5, vertical_spacing_wl=.5)}})
        self.values['tau_per_element_ns'] = self.values['tau_ns'][:, :, None, None, :].copy()
        self.cfg = OmegaConf.create(dict(project_dir=str(self.root), scenario_name='test', frequency=14e9,
            inputs={'calibrated_npz':str(self.source)}, rx_array_geometry=dict(num_rows=1, num_cols=1,
                horizontal_spacing=.5, vertical_spacing=.5), channel_params=dict(dt=1e-9,
                pulse=dict(type='sinc', Tsym=1e-9, beta=0.)), training_sequence=dict(max_delay_bins=4)))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def run_stage(self):
        np.savez(self.source, **self.values)
        with patch.object(stage, 'write_sidecar'):
            stage.main.__wrapped__(self.cfg)
        file = max((self.root/'data/channel_data').glob('*.npz'))
        with np.load(file, allow_pickle=True) as data:
            return file, {key:data[key] for key in data.files}

    def test_center_window_boundary_empty_padding_and_diagnostics(self):
        _, data = self.run_stage()
        np.testing.assert_array_equal(data['num_paths'], [[2, 0]])
        t = np.arange(4)
        expected = 1e-6*np.sinc(t) + 2e-6*np.exp(-2j*np.pi*14e9*3.9e-9)*np.sinc(t-3.9)
        np.testing.assert_allclose(data['H'][0, 0, 0, 0], expected, rtol=1e-7, atol=1e-14)
        np.testing.assert_array_equal(data['H'][0, 1], 0.)
        np.testing.assert_array_equal(data['snr_raw_db'], self.values['snr_raw_db'])
        self.assertAlmostEqual(data['metadata'].item()['last_sample_ns'], 3.)
        self.assertAlmostEqual(data['metadata'].item()['window_ns'], 4.)

    def test_positions_change_diagnostics_not_taps(self):
        _, first = self.run_stage()
        self.values['tx_pos'] *= 2
        _, second = self.run_stage()
        np.testing.assert_array_equal(first['H'], second['H'])
        self.assertNotEqual(first['shift_applied'][0, 0], second['shift_applied'][0, 0])

    def test_invalid_config_fails(self):
        for key, value in [('channel_params.dt', 0), ('channel_params.dt', float('nan')),
                           ('channel_params.pulse.Tsym', -1), ('channel_params.pulse.beta', 1.1),
                           ('channel_params.pulse.beta', float('nan')), ('frequency', 15e9),
                           ('rx_array_geometry.num_cols', 1.5), ('rx_array_geometry.horizontal_spacing', .6)]:
            original = OmegaConf.select(self.cfg, key)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                OmegaConf.update(self.cfg, key, value)
                self.run_stage()
            OmegaConf.update(self.cfg, key, original)

    def test_invalid_counts_and_element_delays_fail(self):
        for counts in (np.array([[5, 0]]), np.array([[-1, 0]]), np.array([[1., 0.]])):
            self.values['num_paths_after_filter'] = counts
            with self.assertRaisesRegex(ValueError, 'counts'):
                self.run_stage()
        self.values['num_paths_after_filter'] = np.array([[3, 0]])
        self.values['tau_per_element_ns'][0, 0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'element delays'):
            self.run_stage()

    def test_fallback_window_and_invalid_duration(self):
        del self.cfg.training_sequence.max_delay_bins
        self.cfg.channel_params.max_delay_ns = 3.
        _, data = self.run_stage()
        self.assertEqual(data['H'].shape[-1], 4)
        self.cfg.channel_params.max_delay_ns = -1
        with self.assertRaisesRegex(ValueError, 'max_delay_ns'):
            self.run_stage()

    def test_same_second_runs_are_preserved(self):
        with patch.object(stage.time, 'strftime', return_value='20260101_000000'):
            first, _ = self.run_stage()
            before = first.read_bytes()
            second, _ = self.run_stage()
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
