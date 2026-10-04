"""Calibration boundary cases and independently computed power/noise checks."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
from src.module_2_path_refinement import apply_link_budget as stage


class LinkBudgetTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.input = self.root / 'filtered.npz'
        self.values = dict(a=np.array([[[1e-6+2e-6j, -3e-6j, 0], [0, 0, 0], [0, 0, 0]]]),
            num_paths_after_filter=np.array([[2, 0, 1]]),
            tau_ns=np.array([[[0., 10., 0.], [0., 0., 0.], [0., 0., 0.]]]),
            is_los_kept=np.array([[True, False, False]]),
            metadata={'frequency_hz':14e9, 'rx_array':{'num_rows':2, 'num_cols':4}})
        self.cfg = OmegaConf.create(dict(project_dir=str(self.root), scenario_name='audit',
            inputs={'filtered_npz':str(self.input)}, training_sequence={'length':31},
            link_budget=dict(tx_power_dbm=20.,tx_antenna_gain_db=3.,rx_antenna_gain_db=-1.,
                             bandwidth_hz=300e6, noise_figure_db=5.,temperature_k=290.)))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def run_stage(self):
        np.savez(self.input, **self.values)
        original=self.input.read_bytes()
        with patch.object(stage,'write_sidecar'):
            stage.main.__wrapped__(self.cfg)
        self.assertEqual(self.input.read_bytes(),original)
        path=max((self.root/'data/path_data/calibrated_paths').glob('*.npz'))
        with np.load(path,allow_pickle=True) as f:
            return path, {k:f[k] for k in f.files}

    def test_scale_noise_and_ideal_gains_with_phase_preservation(self):
        _, d=self.run_stage()
        scale=10**((20-30+3-1)/20)
        np.testing.assert_allclose(d['a'],self.values['a']*scale,rtol=1e-14)
        expected_power=(5e-12+9e-12)*scale**2
        expected_dbm=10*np.log10(expected_power)+30
        noise=1.380649e-23*290*300e6*10**.5
        snr=10*np.log10(expected_power/noise)
        self.assertAlmostEqual(d['p_rx_total_dbm'][0,0],expected_dbm)
        self.assertAlmostEqual(d['snr_raw_db'][0,0],snr)
        self.assertAlmostEqual(d['snr_bf_db'][0,0],snr+10*np.log10(8))
        self.assertAlmostEqual(d['snr_detector_db'][0,0],snr+10*np.log10(8*31))
        self.assertAlmostEqual(d['metadata'].item()['link_budget']['noise_power_w'],noise)
        for key in ('tau_ns','num_paths_after_filter','is_los_kept'):
            np.testing.assert_array_equal(d[key],self.values[key])

    def test_padding_never_contributes_to_power(self):
        self.values['a'][0,0,2]=np.nan
        self.values['a'][0,1,:]=1e10
        _, d=self.run_stage()
        self.assertEqual(d['a'][0,0,2],0)
        np.testing.assert_array_equal(d['a'][0,1],0)
        self.assertTrue(np.isfinite(d['p_rx_total_dbm'][0,0]))
        self.assertTrue(np.isneginf(d['p_rx_total_dbm'][0,1]))

    def test_zero_amplitudes_and_empty_links_have_minus_infinite_power(self):
        _, d=self.run_stage()
        for key in ('p_rx_total_dbm','snr_raw_db','snr_bf_db','snr_detector_db'):
            self.assertTrue(np.isneginf(d[key][0,1:]).all())

    def test_weak_complex64_coefficients_are_not_lost_when_squared(self):
        self.values['a']=np.zeros((1,3,3),np.complex64)
        self.values['a'][0,0,0]=1e-25
        _,d=self.run_stage()
        expected=20*np.log10(float(self.values['a'][0,0,0].real))+(20-30+3-1)+30
        self.assertAlmostEqual(d['p_rx_total_dbm'][0,0],expected)

    def test_invalid_counts_and_valid_coefficients_fail(self):
        for counts in (np.array([[4,0,0]]),np.array([[-1,0,0]]),np.array([[1.,0.,0.]]),np.array([[1]])):
            with self.subTest(counts=counts.tolist()), self.assertRaisesRegex(ValueError,'counts'):
                self.values['num_paths_after_filter']=counts
                self.run_stage()
        self.values['num_paths_after_filter']=np.array([[1,0,0]])
        self.values['a'][0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'coefficients must be finite'):
            self.run_stage()

    def test_tx_power_sweep_scales_signal_not_noise(self):
        _,first=self.run_stage()
        self.cfg.link_budget.tx_power_dbm+=10
        _,second=self.run_stage()
        np.testing.assert_allclose(second['a'],first['a']*np.sqrt(10),rtol=1e-14)
        self.assertAlmostEqual(second['snr_raw_db'][0,0]-first['snr_raw_db'][0,0],10.)
        self.assertEqual(second['metadata'].item()['link_budget']['noise_power_w'],
                         first['metadata'].item()['link_budget']['noise_power_w'])

    def test_opposite_phases_still_use_incoherent_path_power(self):
        self.values['a'][0,0]=[1e-6,-1e-6,0]
        _,d=self.run_stage()
        self.assertAlmostEqual(d['p_rx_total_dbm'][0,0],10*np.log10(2e-12)+22.)
        self.assertEqual(d['a'][0,0].sum(),0)

    def test_same_second_outputs_are_preserved(self):
        with patch.object(stage.time,'strftime',return_value='20260101_000000'):
            first,_=self.run_stage(); before=first.read_bytes()
            second,_=self.run_stage()
        self.assertNotEqual(first,second)
        self.assertEqual(first.read_bytes(),before)


if __name__=='__main__':
    unittest.main()
