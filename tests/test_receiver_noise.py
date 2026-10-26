"""Receiver noise statistics, processing gains, and calibration provenance."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from src.utils.radio_power import diagnostic_gains
from src.utils.receiver_noise import complex_receiver_noise, link_noise_rng, recorded_noise_budget
from src.module_2_path_refinement import apply_link_budget as calibration
from src.module_3_channel_gen import channel_taps_gen as channel
from src.module_4_phy import beam_delay_gen as bdp


def pilot(length, root, bins):
    s = bdp.gen_zadoff_chu(length, root)
    cp = bins - 1
    return np.r_[s[-cp:], s] if cp else s.copy(), np.conj(np.fft.fft(s)), cp


class NoiseMathTests(unittest.TestCase):
    def test_fft_pipeline_matches_direct_convolution_dft_and_correlation(self):
        rows, cols, bins, length = 2, 3, 5, 31
        rng = np.random.default_rng(19)
        H = (rng.normal(size=(rows, cols, bins)) + 1j*rng.normal(size=(rows, cols, bins)))*1e-6
        s_cp, spectrum, cp = pilot(length, 3, bins)
        s = s_cp[cp:]
        y = np.array([[np.convolve(s_cp, H[m, n])[cp:cp+length]
                       for n in range(cols)] for m in range(rows)])
        power = 3.8e-12
        noise = complex_receiver_noise(y.shape, power, link_noise_rng(71, 'tx', 'rx'))
        # Explicit matrices and sums, without using the implementation's FFTs.
        wm = np.exp(-2j*np.pi*np.outer(np.arange(rows)-rows//2, np.arange(rows))/rows)/np.sqrt(rows)
        wn = np.exp(-2j*np.pi*np.outer(np.arange(cols)-cols//2, np.arange(cols))/cols)/np.sqrt(cols)
        beams = np.einsum('am,mnl,bn->abl', wm, y+noise, wn)
        expected = np.stack([np.sum(beams*np.conj(np.roll(s, q)), axis=-1)
                             for q in range(bins)], axis=-1)
        actual, powers = bdp.bdp_from_H(H, s_cp, spectrum, bins, cp, length,
            noise_power_w=power, rng=link_noise_rng(71, 'tx', 'rx'), return_diagnostics=True)
        np.testing.assert_allclose(actual, expected, atol=1e-18, rtol=1e-11)
        self.assertAlmostEqual(powers['signal_power_w']/np.mean(abs(y)**2), 1.)
        self.assertAlmostEqual(powers['noise_power_w']/np.mean(abs(noise)**2), 1.)

    def test_bdp_core_rejects_nonfinite_channels(self):
        s_cp, spectrum, cp = pilot(31, 3, 8)
        H = np.zeros((1, 2, 8), complex)
        H[0, 0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            bdp.bdp_from_H(H, s_cp, spectrum, 8, cp, 31)

    def test_complex_noise_power_and_independence(self):
        power = calibration.noise_power_watts(300e6, 5., 290.)
        self.assertAlmostEqual(10*np.log10(power)+30, -84.203974872, places=6)
        noise = complex_receiver_noise((2, 300000), power, link_noise_rng(71, 'tx', 'rx'))
        self.assertLess(abs(np.mean(noise)) / np.sqrt(power), .005)
        self.assertAlmostEqual(np.mean(abs(noise)**2)/power, 1., delta=.01)
        self.assertAlmostEqual(np.var(noise.real)/(power/2), 1., delta=.01)
        self.assertAlmostEqual(np.var(noise.imag)/(power/2), 1., delta=.01)
        self.assertLess(abs(np.mean(noise[0]*noise[1].conj()))/power, .01)
        self.assertLess(abs(np.mean(noise[:, 1:]*noise[:, :-1].conj()))/power, .01)

    def test_seed_and_link_identity_are_reproducible_and_order_independent(self):
        def draw(seed, tx, rx):
            return complex_receiver_noise((4, 31), 1., link_noise_rng(seed, tx, rx))
        first = draw(71, 'tx0', 'rx0')
        draw(71, 'tx1', 'rx0')
        np.testing.assert_array_equal(first, draw(71, 'tx0', 'rx0'))
        for args in ((72,'tx0','rx0'),(71,'tx1','rx0'),(71,'tx0','rx1')):
            self.assertFalse(np.array_equal(first, draw(*args)))
        for seed in (-1, .5, True):
            with self.assertRaises(ValueError):
                link_noise_rng(seed, 'tx', 'rx')

    def test_noiseless_bdp_matches_analytic_beamspace_channel(self):
        s_cp, spectrum, cp = pilot(31, 3, 8)
        rng = np.random.default_rng(7)
        for rows, cols in ((1,4),(3,4)):
            H = rng.normal(size=(rows,cols,8)) + 1j*rng.normal(size=(rows,cols,8))
            expected = 31*np.fft.fftshift(np.fft.fft(H,axis=1),axes=1)/np.sqrt(cols)
            if rows > 1:
                expected = np.fft.fftshift(np.fft.fft(expected,axis=0),axes=0)/np.sqrt(rows)
            actual = bdp.bdp_from_H(H, s_cp, spectrum, 8, cp, 31)
            np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
            explicit_off = bdp.bdp_from_H(H, s_cp, spectrum, 8, cp, 31, noise_power_w=0,
                                          rng=link_noise_rng(1,'tx','rx'))
            np.testing.assert_array_equal(actual, explicit_off)

    def test_bdp_core_rejects_channel_truncation_and_padding(self):
        H = np.zeros((1, 1, 8), dtype=complex)
        H[0, 0, 6] = 1.0  # a late echo must not silently disappear
        original = H.copy()
        for bins in (4, 16):
            with self.subTest(bins=bins):
                s_cp, spectrum, cp = pilot(31, 3, bins)
                with self.assertRaisesRegex(ValueError, 'must equal the channel tap count'):
                    bdp.bdp_from_H(H, s_cp, spectrum, bins, cp, 31)
                np.testing.assert_array_equal(H, original)

    def test_beamforming_preserves_noise_and_correlation_scales_variance(self):
        length = 127
        s_cp, spectrum, cp = pilot(length, 3, length)
        power = 3.8e-12
        result = bdp.bdp_from_H(np.zeros((8,64,length)), s_cp, spectrum, length, cp, length,
                               noise_power_w=power, rng=link_noise_rng(5,'tx','rx'))
        self.assertAlmostEqual(np.mean(abs(result)**2)/(length*power), 1., delta=.02)

    def test_broadside_array_and_pilot_gains_emerge_from_processing(self):
        length = 839
        s_cp, spectrum, cp = pilot(length, 29, 1)
        result = bdp.bdp_from_H(np.ones((1,64,1)), s_cp, spectrum, 1, cp, length)
        self.assertAlmostEqual(abs(result[0,32,0]), np.sqrt(64)*length, places=8)
        # Output signal/noise ratio divided by one antenna's input SNR.
        noise_power = 3.8e-12
        gain = abs(result[0,32,0])**2/(length*noise_power) / (1/noise_power)
        self.assertAlmostEqual(gain, 64*length, places=6)

    def test_increasing_tx_power_changes_signal_but_not_receiver_noise(self):
        s_cp, spectrum, cp = pilot(31, 3, 8)
        args = (s_cp, spectrum, 8, cp, 31)
        H = np.ones((1,4,8), dtype=complex)*1e-5
        clean1 = bdp.bdp_from_H(H, *args)
        clean2 = bdp.bdp_from_H(H*np.sqrt(10), *args)
        noisy1, d1 = bdp.bdp_from_H(H, *args, noise_power_w=1e-10,
                                   rng=link_noise_rng(1,'t','r'), return_diagnostics=True)
        noisy2, d2 = bdp.bdp_from_H(H*np.sqrt(10), *args, noise_power_w=1e-10,
                                   rng=link_noise_rng(1,'t','r'), return_diagnostics=True)
        self.assertAlmostEqual(d2['signal_power_w']/d1['signal_power_w'], 10.)
        self.assertEqual(d1['noise_power_w'], d2['noise_power_w'])
        np.testing.assert_allclose(noisy1-clean1, noisy2-clean2, atol=1e-17)

    def test_pilot_parity_and_invalid_configurations(self):
        for length, root in ((8,1),(31,3)):
            s=bdp.gen_zadoff_chu(length,root)
            correlation=np.fft.ifft(abs(np.fft.fft(s))**2)
            np.testing.assert_allclose(correlation[1:],0,atol=1e-10)
            self.assertAlmostEqual(abs(correlation[0]),length)
        for length,root in ((9,3),(0,1),(7,7),(7,0),(7.5,1)):
            with self.subTest(length=length,root=root), self.assertRaises(ValueError):
                bdp.gen_zadoff_chu(length,root)
        s_cp,spectrum,cp=pilot(7,1,8)
        with self.assertRaises(ValueError):
            bdp.bdp_from_H(np.zeros((1,1,8)),s_cp,spectrum,8,cp,7)

    def test_noise_budget_uses_original_metadata_and_legacy_dbm(self):
        budget=dict(bandwidth_hz=300e6,noise_power_dbm=-84.203974872)
        meta=dict(dt_s=3.33e-9,source_metadata={'link_budget':budget})
        power,_,_=recorded_noise_budget(meta)
        budget['noise_power_w']=power
        self.assertEqual(recorded_noise_budget(meta)[0],power)
        budget['noise_power_w']=power*2
        with self.assertRaisesRegex(ValueError,'disagree'):
            recorded_noise_budget(meta)
        del budget['noise_power_w']
        meta['dt_s']=1e-9
        with self.assertRaisesRegex(ValueError,'bandwidth_hz.*dt_s'):
            recorded_noise_budget(meta)
        with self.assertRaisesRegex(ValueError,'original link budget'):
            recorded_noise_budget({})

    def test_invalid_physical_inputs_and_derived_gains(self):
        for bandwidth, nf, temp in ((0,5,290),(-1,5,290),(300e6,5,0),
                                    (300e6,float('nan'),290),(300e6,-1,290)):
            with self.assertRaises(ValueError):
                calibration.noise_power_watts(bandwidth,nf,temp)
        np.testing.assert_allclose(diagnostic_gains(2,4,31), [10*np.log10(8),10*np.log10(31)])
        with self.assertRaises(ValueError):
            diagnostic_gains(0,4,31)


class NoisePipelineTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        self.filtered=self.root/'filtered.npz'
        counts=np.array([[1,1,0],[0,1,0]])
        shape=(2,3,1)
        tx=np.array([[10.,0.,1.5],[20.,0.,1.5],[30.,0.,1.5]])
        rx=np.array([[0.,0.,5.],[0.,10.,5.]])
        self.devices=dict(tx_names=np.array(['tx0','tx1','tx2']),rx_names=np.array(['bs0','bs1']),
                          tx_pos=tx,rx_pos=rx,rx_look_at=rx+[1,0,0])
        np.savez(self.filtered,a=(counts[...,None]*1e-5).astype(complex),
                 num_paths_after_filter=counts,tau_ns=np.zeros(shape),
                 tau_per_element_ns=np.zeros((2,3,1,4,1)),omega_h=np.zeros(shape),omega_v=np.zeros(shape),
                 min_delay_ns=np.linalg.norm(tx[None,:,:]-rx[:,None,:],axis=-1)/299792458*1e9,
                 is_los_kept=counts>0,interactions=np.zeros((1,)+shape,dtype=np.int16),
                 metadata={'frequency_hz':14e9,'rx_array':{'num_rows':1,'num_cols':4,
                    'horizontal_spacing_wl':.5,'vertical_spacing_wl':.5}}, **self.devices)
        self.cfg=OmegaConf.create(dict(project_dir=str(self.root),scenario_name='noise_test',frequency=14e9,
            inputs={'filtered_npz':str(self.filtered)},
            rx_array_geometry=dict(num_rows=1,num_cols=4,horizontal_spacing=.5,vertical_spacing=.5),
            link_budget=dict(tx_power_dbm=30.,bandwidth_hz=300e6,noise_figure_db=5.,temperature_k=290.),
            channel_params=dict(dt=3.33e-9,pulse=dict(type='sinc',Tsym=3.33e-9,beta=0.),
                                include_true_time_delay=True,include_spatial_phase=True),
            training_sequence=dict(type='zadoff_chu',length=31,root_index=3,max_delay_bins=8,label_dim=2),
            receiver_noise=dict(enabled=True,seed=71)))
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def run_stage(self,module,pattern):
        with patch.object(module,'write_sidecar'):
            if module is bdp:
                with patch.object(bdp,'save_sanity_plot'):
                    module.main.__wrapped__(self.cfg)
            else:
                module.main.__wrapped__(self.cfg)
        path=sorted(self.root.glob(pattern))[-1]
        with np.load(path,allow_pickle=True) as data:
            return path,{key:data[key] for key in data.files}

    def upstream(self):
        path,calib=self.run_stage(calibration,'data/path_data/calibrated_paths/*.npz')
        self.cfg.inputs.calibrated_npz=str(path)
        ch_path,ch=self.run_stage(channel,'data/channel_data/*.npz')
        self.cfg.inputs.channels_npz=str(ch_path)
        return path,calib,ch_path,ch

    def output(self):
        return self.run_stage(bdp,'data/ml_datasets/noise_test/*.npz')[1]

    def test_fractional_label_dimension_is_rejected(self):
        self.upstream()
        self.cfg.training_sequence.label_dim = 2.5
        with self.assertRaisesRegex(ValueError, 'label_dim'):
            self.output()

    def test_invalid_counts_and_channel_storage_overflow_are_rejected(self):
        _, _, path, ch = self.upstream()
        counts = ch['num_paths'].copy()
        ch['num_paths'] = counts.astype(float)
        np.savez(path, **ch)
        with self.assertRaisesRegex(ValueError, 'num_paths'):
            self.output()
        ch['num_paths'] = counts
        ch['H'][0, 0] = 1e38
        np.savez(path, **ch)
        with self.assertRaisesRegex(ValueError, 'Nonfinite BDP'):
            self.output()

    def test_output_metadata_records_unnormalized_correlation_noise(self):
        self.upstream()
        meta = self.output()['metadata'].item()
        self.assertAlmostEqual(meta['noise_power_w_per_complex_output_bin'] /
                               meta['receiver_noise']['noise_power_w_per_element'], 31.)
        self.assertIn('unnormalized', meta['correlation_normalization'])

    def test_bdp_rejects_conflicting_window_before_generating_or_saving(self):
        self.upstream()
        for bins in (4, 16):
            with self.subTest(bins=bins), patch.object(bdp, 'bdp_from_H') as generate:
                self.cfg.training_sequence.max_delay_bins = bins
                with self.assertRaisesRegex(ValueError, 'saved channel has 8 taps.*regenerate'):
                    self.output()
                generate.assert_not_called()
                self.assertEqual(list(self.root.glob('data/ml_datasets/noise_test/*.npz')), [])

    def test_bdp_derives_length_and_prefix_from_saved_channel_without_config(self):
        self.upstream()
        expected = self.output()
        del self.cfg.training_sequence.max_delay_bins
        actual = self.output()
        np.testing.assert_array_equal(actual['Z'], expected['Z'])
        sequence_meta = actual['metadata'].item()['training_sequence']
        self.assertEqual(sequence_meta['Lh'], 8)
        self.assertEqual(sequence_meta['Lcp'], 7)

    def test_pilot_must_cover_saved_channel(self):
        self.upstream()
        self.cfg.training_sequence.length = 7
        with self.assertRaisesRegex(ValueError, 'Saved channel tap count must not exceed pilot length'):
            self.output()

    def test_channel_generation_rejects_invalid_tap_counts(self):
        self.upstream()
        for bins in (0, -1, 8.5):
            with self.subTest(bins=bins):
                self.cfg.training_sequence.max_delay_bins = bins
                with self.assertRaisesRegex(ValueError, 'training_sequence.max_delay_bins'):
                    channel.main.__wrapped__(self.cfg)

    def test_derived_gains_replace_manual_values_and_follow_current_pilot(self):
        self.cfg.link_budget.array_gain_db=999
        self.cfg.link_budget.processing_gain_db=999
        _,calib,_,_=self.upstream()
        budget=calib['metadata'].item()['link_budget']
        self.assertAlmostEqual(budget['array_gain_db'],10*np.log10(4))
        self.assertAlmostEqual(budget['processing_gain_db'],10*np.log10(31))
        self.assertGreater(budget['noise_power_w'],0)
        self.cfg.training_sequence.length=127
        out=self.output()
        self.assertEqual(out['Z'].shape, (2, 4, 8, 2))
        self.assertEqual(out['metadata'].item()['training_sequence']['Lcp'], 7)
        np.testing.assert_allclose(out['snr_bf_db'][0,0]-out['snr_raw_db'][0,0],10*np.log10(4),rtol=1e-6)
        np.testing.assert_allclose(out['snr_detector_db'][0,0]-out['snr_bf_db'][0,0],10*np.log10(127),rtol=1e-6)

    def test_missing_links_stay_zero_and_covered_noise_is_reproducible(self):
        _, _, path, ch = self.upstream()
        for rows in (1, 2):
            with self.subTest(rows=rows):
                expanded = dict(ch, H=np.repeat(ch['H'], rows, axis=2))
                np.savez(path, **expanded)
                with patch.object(bdp, 'bdp_from_H', wraps=bdp.bdp_from_H) as generate:
                    first = self.output()
                self.assertEqual(generate.call_count, 3)  # only the three covered links
                np.testing.assert_array_equal(first['has_paths'], [[True, False], [True, True]])
                np.testing.assert_array_equal(first['tx_idx'], [0, 1])  # fully uncovered TX excluded
                np.testing.assert_array_equal(first['Z'][0, ..., 1], 0)
                self.assertEqual(first['pilot_signal_power_w'][0, 1], 0)
                self.assertEqual(first['realized_noise_power_w'][0, 1], 0)
                self.assertTrue(np.all(first['realized_noise_power_w'][first['has_paths']] > 0))
                meta = first['metadata'].item()
                self.assertEqual(meta['receiver_noise']['missing_link'], 'zero')
                self.assertEqual(meta['zero_padded'], {'bs0': 0, 'bs1': 1})
                self.assertEqual(meta['noise_only'], {'bs0': 0, 'bs1': 0})
                # Covered observations are identical to the existing noisy DSP core.
                s_cp, spectrum, cp = pilot(31, 3, 8)
                for t, r in ((0, 0), (1, 0), (1, 1)):
                    expected = bdp.bdp_from_H(expanded['H'][r, t].astype(complex),
                        s_cp, spectrum, 8, cp, 31,
                        noise_power_w=meta['receiver_noise']['noise_power_w_per_element'],
                        rng=link_noise_rng(71, self.devices['tx_names'][t], self.devices['rx_names'][r]))
                    if rows == 1:
                        expected = expected[0]
                    np.testing.assert_array_equal(first['Z'][t, ..., r], expected.astype(np.complex64))
                np.testing.assert_array_equal(self.output()['Z'], first['Z'])
                self.cfg.receiver_noise.seed = 72
                changed = self.output()
                np.testing.assert_array_equal(changed['Z'][0, ..., 1], 0)
                self.assertFalse(np.array_equal(changed['Z'], first['Z']))
                self.cfg.receiver_noise.seed = 71
        # Read-only generation leaves the current input intact.
        snapshot = path.read_bytes()
        self.output()
        self.assertEqual(path.read_bytes(), snapshot)

    def test_positive_path_count_with_zero_channel_still_receives_noise(self):
        _, _, path, ch = self.upstream()
        ch['H'][0, 0] = 0
        np.savez(path, **ch)
        out = self.output()
        self.assertTrue(out['has_paths'][0, 0])
        self.assertTrue(np.any(out['Z'][0, ..., 0] != 0))
        self.assertGreater(out['realized_noise_power_w'][0, 0], 0)
        np.testing.assert_array_equal(out['Z'][0, ..., 1], 0)

    def test_noise_disabled_retains_zero_links_and_clean_reference(self):
        _,_,_,ch=self.upstream()
        self.cfg.receiver_noise.enabled=False
        out=self.output()
        np.testing.assert_array_equal(out['Z'][0,:,:,1],0)
        np.testing.assert_array_equal(out['realized_noise_power_w'],0)
        s_cp,spectrum,cp=pilot(31,3,8)
        expected=bdp.bdp_from_H(ch['H'][0,0].astype(complex),s_cp,spectrum,8,cp,31)[0]
        np.testing.assert_array_equal(out['Z'][0,:,:,0],expected.astype(np.complex64))

    def test_embedded_budget_survives_missing_calibration_and_ignores_yaml_drift(self):
        path,calib,_,_=self.upstream()
        expected=calib['metadata'].item()['link_budget']['noise_power_w']
        path.unlink()
        del self.cfg.inputs.calibrated_npz
        self.cfg.link_budget.bandwidth_hz=1
        self.cfg.link_budget.noise_figure_db=99
        out=self.output()
        self.assertEqual(out['metadata'].item()['receiver_noise']['noise_power_w_per_element'],expected)
        self.assertEqual(out['metadata'].item()['snr_source'],'embedded_channel_diagnostics')

    def test_legacy_follows_recorded_calibration_not_newest(self):
        path,calib,ch_path,ch=self.upstream()
        for key in ('snr_raw_db','snr_bf_db','snr_detector_db','p_rx_total_dbm'):
            del ch[key]
        # Simulate legacy metadata storing only dBm.
        del ch['metadata'].item()['source_metadata']['link_budget']['noise_power_w']
        del calib['metadata'].item()['link_budget']['noise_power_w']
        np.savez(path,**calib)
        np.savez(ch_path,**ch)
        wrong=dict(calib)
        wrong['snr_raw_db']=np.full((2,3),90.)
        newest=path.parent/'29990101_000000_calibrated_noise_test.npz'
        np.savez(newest,**wrong)
        del self.cfg.inputs.calibrated_npz
        out=self.output()
        self.assertEqual(out['metadata'].item()['source_calibrated_npz'],path.name)
        self.assertEqual(out['metadata'].item()['snr_source'],'recorded_calibration')
        np.testing.assert_allclose(out['snr_raw_db'],calib['snr_raw_db'][:,:2].T,rtol=1e-6)
        self.cfg.inputs.calibrated_npz=str(newest)
        with self.assertRaisesRegex(ValueError,'recorded calibration'):
            self.output()

    def test_legacy_calibration_device_order_must_match(self):
        path,calib,ch_path,ch=self.upstream()
        for key in ('snr_raw_db','snr_bf_db','snr_detector_db','p_rx_total_dbm'):
            del ch[key]
        np.savez(ch_path,**ch)
        calib['tx_names']=calib['tx_names'][::-1]
        np.savez(path,**calib)
        with self.assertRaisesRegex(ValueError,'device order'):
            self.output()

    def test_reordering_links_preserves_named_link_noise(self):
        _,_,path,ch=self.upstream()
        original=self.output()
        tx_order=[1,0,2];rx_order=[1,0]
        for key in ('H','num_paths','is_los_kept','shift_applied','snr_raw_db','snr_bf_db',
                    'snr_detector_db','p_rx_total_dbm','tau_los_ns'):
            ch[key]=ch[key][rx_order][:,tx_order]
        for key in ('tx_names','tx_pos'):
            ch[key]=ch[key][tx_order]
        for key in ('rx_names','rx_pos','rx_look_at'):
            ch[key]=ch[key][rx_order]
        np.savez(path,**ch)
        reordered=self.output()
        np.testing.assert_array_equal(reordered['Z'],original['Z'][[1,0]][:,:,:,[1,0]])

    def test_same_instant_runs_save_distinct_datasets(self):
        from datetime import datetime
        self.upstream()
        with patch.object(bdp, 'datetime') as clock:
            clock.now.return_value = datetime(2030, 1, 1, 12, 0, 0)
            self.output()
            self.cfg.receiver_noise.seed = 72
            self.output()
        files = list(self.root.glob('data/ml_datasets/noise_test/*.npz'))
        self.assertEqual(len(files), 2)
        outputs = {}
        for path in files:
            with np.load(path, allow_pickle=True) as data:
                outputs[data['metadata'].item()['receiver_noise']['seed']] = data['Z']
        self.assertEqual(set(outputs), {71, 72})
        self.assertFalse(np.array_equal(outputs[71], outputs[72]))

    def test_exact_filename_collision_never_overwrites_existing_dataset(self):
        from datetime import datetime
        self.upstream()
        with patch.object(bdp, 'datetime') as clock, patch.object(bdp.uuid, 'uuid4') as unique:
            clock.now.return_value = datetime(2030, 1, 1, 12, 0, 0)
            unique.return_value.hex = 'a'*32
            self.output()
            path = next(self.root.glob('data/ml_datasets/noise_test/*.npz'))
            original = path.read_bytes()
            self.cfg.receiver_noise.seed = 72
            with self.assertRaises(FileExistsError):
                self.output()
            self.assertEqual(path.read_bytes(), original)

    def test_changed_array_dimensions_produce_new_diagnostic_gain(self):
        with np.load(self.filtered,allow_pickle=True) as data:
            values={key:data[key] for key in data.files}
        values['metadata'].item()['rx_array'].update(num_rows=2,num_cols=3)
        values['tau_per_element_ns']=np.zeros((2,3,2,3,1))
        np.savez(self.filtered,**values)
        self.cfg.rx_array_geometry.num_rows=2
        self.cfg.rx_array_geometry.num_cols=3
        _,calib,_,_=self.upstream()
        self.assertAlmostEqual(calib['metadata'].item()['link_budget']['array_gain_db'],10*np.log10(6))
        out=self.output()
        self.assertEqual(out['Z'].shape,(2,2,3,8,2))
        self.assertAlmostEqual(out['metadata'].item()['diagnostic_gains']['array_gain_db'],10*np.log10(6))


if __name__ == '__main__':
    unittest.main()
