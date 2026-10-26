"""BDP array/BS axis contracts and per-link power diagnostics."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from src.module_6_localization import ml_preprocessing as pp


class BdpPreprocessingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / 'bdp.npz'
        self.cfg = OmegaConf.create(dict(
            project_dir=str(self.root), scenario_name='test',
            inputs={'bdp_npz': str(self.source)},
            ml_preprocessing={'normalization': 'per_sample', 'magnitude_mode': 'linear'}))
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def fixture(self, rows=1, stations=1, powers=True):
        shape = (3,) + ((rows,) if rows > 1 else ()) + (4, 8)
        if stations > 1:
            shape += (stations,)
        z = np.arange(1, np.prod(shape)+1).reshape(shape).astype(np.complex64)
        data = dict(Z=z, Y=np.arange(6).reshape(3, 2),
                    rx_names=np.array([f'bs{i}' for i in range(stations)]),
                    rx_pos=np.zeros((stations, 3)), tx_idx=np.arange(3),
                    has_paths=np.ones((3, stations), dtype=bool),
                    metadata={'num_base_stations': stations, 'array': {'M': rows, 'N': 4}})
        if powers:
            data['pilot_signal_power_w'] = np.arange(1, 3*stations+1).reshape(3, stations)*1e-8
            data['realized_noise_power_w'] = np.arange(11, 3*stations+11).reshape(3, stations)*1e-12
        return data

    def process(self, data):
        np.savez(self.source, **data)
        with patch.object(pp, 'write_sidecar') as sidecar:
            pp.main.__wrapped__(self.cfg)
        path = Path(sidecar.call_args.args[0])
        with np.load(path, allow_pickle=True) as output:
            return {key: output[key] for key in output.files}

    def test_single_bs_ula_and_upa_get_exactly_one_channel_axis(self):
        for rows in (1, 2):
            with self.subTest(rows=rows):
                data = self.fixture(rows=rows)
                out = self.process(data)
                self.assertEqual(out['X'].shape, data['Z'].shape + (1,))
                self.assertEqual(out['metadata'].item()['K'], 1)
                self.assertFalse(out['metadata'].item()['multi_bs'])
                for key in ('pilot_signal_power_w', 'realized_noise_power_w'):
                    np.testing.assert_array_equal(out[key], data[key])

    def test_multi_bs_ula_and_upa_preserve_existing_channel_axis(self):
        for rows in (1, 2):
            with self.subTest(rows=rows):
                data = self.fixture(rows=rows, stations=2)
                out = self.process(data)
                self.assertEqual(out['X'].shape, data['Z'].shape)
                self.assertEqual(out['metadata'].item()['K'], 2)

    def test_powers_follow_bs_selection_and_sample_filtering(self):
        for rows in (1, 2):
            with self.subTest(rows=rows):
                data = self.fixture(rows=rows, stations=2)
                data['snr_detector_db'] = np.array([[5, 15], [25, 5], [30, 20]])
                self.cfg.ml_preprocessing.keep_bs = [1]
                self.cfg.ml_preprocessing.min_snr_detector_db = 10
                out = self.process(data)
                expected_shape = (2,) + data['Z'].shape[1:-1] + (1,)
                self.assertEqual(out['X'].shape, expected_shape)
                self.assertEqual(out['metadata'].item()['K'], 1)
                for key in ('pilot_signal_power_w', 'realized_noise_power_w'):
                    np.testing.assert_array_equal(out[key], data[key][[0, 2]][:, [1]])
                np.testing.assert_array_equal(out['Y'], data['Y'][[0, 2]])
                np.testing.assert_array_equal(out['rx_names'], ['bs1'])

    def test_powers_follow_reordered_bs_selection(self):
        data = self.fixture(stations=3)
        self.cfg.ml_preprocessing.keep_bs = [2, 0]
        out = self.process(data)
        for key in ('pilot_signal_power_w', 'realized_noise_power_w'):
            np.testing.assert_array_equal(out[key], data[key][:, [2, 0]])
        np.testing.assert_array_equal(out['rx_names'], ['bs2', 'bs0'])

    def test_legacy_receiver_table_and_missing_optional_powers(self):
        data = self.fixture(powers=False)
        data['metadata'] = {}
        out = self.process(data)
        self.assertEqual(out['X'].shape, (3, 4, 8, 1))
        self.assertNotIn('realized_noise_power_w', out)

    def test_missing_or_inconsistent_layout_fails_before_transform(self):
        data = self.fixture()
        data['metadata'] = {}
        del data['rx_names']
        with self.assertRaisesRegex(ValueError, 'identify its BS axis'):
            self.process(data)
        data = self.fixture()
        data['metadata']['num_base_stations'] = 2
        with self.assertRaisesRegex(ValueError, 'disagrees with rx_names'):
            self.process(data)
        data = self.fixture(stations=2)
        data['Z'] = data['Z'][..., 0]
        with self.assertRaisesRegex(ValueError, 'shape.*disagrees'):
            self.process(data)

    def test_single_bs_selection_zero_preserves_array_and_metadata(self):
        for rows in (1, 2):
            with self.subTest(rows=rows):
                data = self.fixture(rows=rows)
                self.cfg.ml_preprocessing.pop('keep_bs', None)
                baseline = self.process(data)
                self.cfg.ml_preprocessing.keep_bs = [0]
                selected = self.process(data)
                for key in ('X', 'Y', 'rx_names', 'has_paths',
                            'pilot_signal_power_w', 'realized_noise_power_w'):
                    np.testing.assert_array_equal(selected[key], baseline[key])
                self.assertEqual(selected['metadata'].item()['K'], 1)

    def test_invalid_bs_selections_fail_before_transform_or_output(self):
        for stations in (1, 2):
            data = self.fixture(stations=stations)
            for selection in ([], [0, 0], [-1], [stations], [0.9], [0.0],
                              [True], ['0'], '0', 0, {'index': 0}):
                with self.subTest(stations=stations, selection=selection):
                    self.cfg.ml_preprocessing.keep_bs = selection
                    with patch.object(pp, 'complex_to_cnn_input_batched') as transform:
                        with self.assertRaisesRegex(ValueError, 'ml_preprocessing.keep_bs'):
                            self.process(data)
                        transform.assert_not_called()
        self.assertEqual(list(self.root.glob('data/preprocessed_datasets/test/*.npz')), [])

    def test_empty_filter_result_fails_and_preserves_prior_dataset(self):
        data = self.fixture(stations=2)
        self.process(data)
        original_path = next(self.root.glob('data/preprocessed_datasets/test/*.npz'))
        original_bytes = original_path.read_bytes()
        data['snr_detector_db'] = np.zeros((3, 2))
        data['is_los_kept'] = np.zeros((3, 2), dtype=bool)
        data['has_paths'] = np.zeros((3, 2), dtype=bool)
        for settings in ({'min_snr_detector_db': 10}, {'los_kept_only': True},
                         {'drop_uncovered': True}):
            with self.subTest(settings=settings):
                self.cfg.ml_preprocessing = dict(normalization='per_sample',
                                                 magnitude_mode='linear', **settings)
                with patch.object(pp, 'complex_to_cnn_input_batched') as transform:
                    with self.assertRaisesRegex(ValueError, 'No samples remain.*No dataset was generated'):
                        self.process(data)
                    transform.assert_not_called()
                self.assertEqual(list(original_path.parent.glob('*.npz')), [original_path])
                self.assertEqual(original_path.read_bytes(), original_bytes)

    def test_same_instant_preprocessing_runs_keep_both_outputs(self):
        from datetime import datetime
        data = self.fixture(stations=2)
        with patch.object(pp, 'datetime') as clock:
            clock.now.return_value = datetime(2030, 1, 1, 12, 0, 0)
            first = self.process(data)
            original_path = next(self.root.glob('data/preprocessed_datasets/test/*.npz'))
            original_bytes = original_path.read_bytes()
            data['Z'] = np.ones_like(data['Z'])
            second = self.process(data)
        self.assertEqual(len(list(original_path.parent.glob('*.npz'))), 2)
        self.assertEqual(original_path.read_bytes(), original_bytes)
        self.assertFalse(np.array_equal(first['X'], second['X']))

    def test_forced_filename_collision_preserves_dataset_and_sidecar(self):
        from datetime import datetime
        from src.utils.dataset_io import sidecar_path
        data = self.fixture(stations=2)
        with patch.object(pp, 'datetime') as clock, patch.object(pp.uuid, 'uuid4') as unique:
            clock.now.return_value = datetime(2030, 1, 1, 12, 0, 0)
            unique.return_value.hex = 'a'*32
            self.process(data)
            path = next(self.root.glob('data/preprocessed_datasets/test/*.npz'))
            original = path.read_bytes()
            sidecar = Path(sidecar_path(path))
            sidecar.write_text('{"existing": true}\n')
            data['Z'] = np.ones_like(data['Z'])
            with self.assertRaises(FileExistsError):
                self.process(data)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(sidecar.read_text(), '{"existing": true}\n')

    def test_invalid_transform_settings_fail_before_loading_or_allocating(self):
        invalid = [('magnitude_mode', value) for value in ('lg', '', None)]
        invalid += [('normalization', value) for value in ('none', 'global', 'per_sampel', '', None)]
        invalid += [('dynamic_range_db', value)
                    for value in (0, -10, float('nan'), float('inf'), -float('inf'), None, 'bad')]
        z = np.ones((2, 4, 8, 2), dtype=np.complex64)
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                settings = dict(magnitude_mode='log', normalization='per_sample', dynamic_range_db=50)
                settings[key] = value
                with patch.object(pp.np, 'empty') as allocate:
                    with self.assertRaisesRegex(ValueError, 'ml_preprocessing.' + key):
                        pp.complex_to_cnn_input_batched(z, **settings, channels_present=True)
                    allocate.assert_not_called()
                self.cfg.ml_preprocessing = settings
                with patch.object(pp, 'resolve_input') as load:
                    with self.assertRaisesRegex(ValueError, 'ml_preprocessing.' + key):
                        pp.main.__wrapped__(self.cfg)
                    load.assert_not_called()
        self.assertEqual(list(self.root.glob('data/preprocessed_datasets/test/*.npz')), [])

    def test_supported_transforms_preserve_expected_values(self):
        z = np.array([[2, .2, 0], [20, 2, 0]], dtype=np.complex64).reshape(2, 1, 3, 1)
        cases = (
            ('linear', 'per_sample', [[1, .1, 0], [1, .1, 0]]),
            ('linear', 'per_bs', [[1, .1, 0], [1, .1, 0]]),
            ('log', 'per_sample', [[1, .6, 0], [1, .6, 0]]),
            ('log', 'per_bs', [[1, .6, 0], [1, .6, 0]]),
            (' LOG ', ' PER_SAMPLE ', [[1, .6, 0], [1, .6, 0]]),
        )
        for mode, norm, expected in cases:
            with self.subTest(mode=mode, norm=norm):
                # Dynamic range is unused in linear mode, so zero is allowed there.
                result = pp.complex_to_cnn_input_batched(
                    z, mode, norm, 0 if mode == 'linear' else 50, channels_present=True)
                np.testing.assert_allclose(result.reshape(2, 3), expected, atol=1e-6)
                self.assertTrue(np.isfinite(result).all())

    def test_valid_log_pipeline_saves_finite_values_and_settings(self):
        self.cfg.ml_preprocessing.magnitude_mode = 'log'
        self.cfg.ml_preprocessing.dynamic_range_db = 50
        data = self.fixture(stations=2)
        out = self.process(data)
        self.assertEqual(out['X'].shape, data['Z'].shape)
        self.assertTrue(np.isfinite(out['X']).all())
        self.assertGreaterEqual(float(out['X'].min()), 0)
        self.assertLessEqual(float(out['X'].max()), 1)
        self.assertEqual(out['metadata'].item()['transform'],
                         dict(magnitude_mode='log', normalization='per_sample', dynamic_range_db=50.0))

    def test_transform_requires_explicit_axis_semantics(self):
        with self.assertRaisesRegex(ValueError, 'channels_present'):
            pp.complex_to_cnn_input_batched(np.ones((2, 4, 8)), 'linear', 'per_sample', 50)

    def test_per_bs_reference_is_local_to_each_position_and_station(self):
        z = np.array([[[[4., 1., 0.], [2., .25, 0.]]]], complex)
        for mode in ('linear', 'log'):
            kwargs = dict(magnitude_mode=mode, normalization='per_bs', dynamic_range_db=50,
                          channels_present=True, return_peaks=True)
            alone, peaks = pp.complex_to_cnn_input_batched(z, **kwargs)
            np.testing.assert_array_equal(peaks, [[4., 1., 0.]])
            np.testing.assert_array_equal(alone[..., 2], 0)
            np.testing.assert_array_equal(alone[0, 0, 0], [1., 1., 0.])
            if mode == 'linear':
                np.testing.assert_array_equal(alone[0, 0, 1], [.5, .25, 0.])
            else:
                np.testing.assert_allclose(alone[0, 0, 1],
                    [1+20*np.log10(.5)/50, 1+20*np.log10(.25)/50, 0], atol=1e-7)
            combined = np.concatenate([z, z*100])
            combined[..., 0] *= 10
            for batch_size in (1, 2):
                actual, _ = pp.complex_to_cnn_input_batched(combined, batch_size=batch_size, **kwargs)
                np.testing.assert_array_equal(actual, np.concatenate([alone, alone]))

    def test_per_bs_layout_selection_and_saved_peak_order(self):
        self.cfg.ml_preprocessing.normalization = 'per_bs'
        for rows in (1, 2):
            for stations, selection in ((1, None), (3, None), (3, [2, 0]), (3, [1])):
                with self.subTest(rows=rows, stations=stations, selection=selection):
                    self.cfg.ml_preprocessing.keep_bs = selection
                    data = self.fixture(rows=rows, stations=stations)
                    out = self.process(data)
                    z = data['Z'] if stations > 1 else data['Z'][..., None]
                    if selection is not None:
                        z = z[..., selection]
                    expected_peaks = np.max(abs(z), axis=tuple(range(1, z.ndim-1)))
                    np.testing.assert_array_equal(out['normalization_peak'], expected_peaks)
                    np.testing.assert_allclose(out['X'], z.real/expected_peaks.reshape(
                        (len(z),)+(1,)*(z.ndim-2)+(z.shape[-1],)), atol=1e-7)
                    names = data['rx_names'] if selection is None else data['rx_names'][selection]
                    np.testing.assert_array_equal(out['rx_names'], names)

    def test_per_sample_peak_is_shared_across_selected_base_stations(self):
        z = np.array([[[[4., 1.], [2., .5]]]], complex)
        x, peaks = pp.complex_to_cnn_input_batched(z, 'linear', 'per_sample', 50,
                                                  channels_present=True, return_peaks=True)
        np.testing.assert_array_equal(x, z.real/4)
        np.testing.assert_array_equal(peaks, [4.])
        # Selecting the weaker BS first recomputes the sample peak to 1.
        subset = pp.complex_to_cnn_input_batched(z[..., [1]], 'linear', 'per_sample', 50,
                                                channels_present=True)
        np.testing.assert_array_equal(subset[0, 0, :, 0], [1., .5])

    def test_phase_discard_scale_invariance_and_zero_dynamic_range(self):
        z = np.array([[[1e-25j, -1e-26, 0]], [[0, 0, 0]]], complex)
        for mode, norm in ((m, n) for m in ('linear', 'log') for n in ('per_sample', 'per_bs')):
            kwargs = dict(magnitude_mode=mode, normalization=norm, dynamic_range_db=300,
                          channels_present=False)
            x = pp.complex_to_cnn_input_batched(z, **kwargs)
            np.testing.assert_allclose(x, pp.complex_to_cnn_input_batched(z*1e20, **kwargs), atol=1e-7)
            np.testing.assert_allclose(x, pp.complex_to_cnn_input_batched(z*np.exp(.3j), **kwargs), atol=1e-7)
            np.testing.assert_array_equal(x[1], 0)
            self.assertEqual(x[0, 0, 0, 0], 1.)

    def test_normalization_references_and_filter_settings_are_saved(self):
        data = self.fixture(stations=2)
        self.cfg.ml_preprocessing.keep_bs = [1]
        self.cfg.ml_preprocessing.drop_uncovered = True
        out = self.process(data)
        expected = np.max(abs(data['Z'][..., [1]]), axis=(1, 2, 3))
        np.testing.assert_array_equal(out['normalization_peak'], expected)
        meta = out['metadata'].item()
        self.assertEqual(meta['filters']['keep_bs'], [1])
        self.assertTrue(meta['filters']['drop_uncovered'])
        self.cfg.ml_preprocessing.normalization = 'per_bs'
        out = self.process(data)
        np.testing.assert_array_equal(out['normalization_peak'], expected[:, None])
        reference = out['metadata'].item()['normalization_reference']
        self.assertEqual(reference['scope'], 'sample_per_bs')
        self.assertEqual(reference['peak_axes'], '(samples, selected_bs)')

    def test_invalid_values_and_sample_metadata_fail(self):
        data = self.fixture(stations=2)
        data['Z'][0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'BDP values'):
            self.process(data)
        data = self.fixture(stations=2)
        data['Y'] = data['Y'][:2]
        with self.assertRaisesRegex(ValueError, 'Y must'):
            self.process(data)
        data = self.fixture(stations=2)
        data['has_paths'] = np.ones((3, 3), bool)
        with self.assertRaisesRegex(ValueError, 'has_paths shape'):
            self.process(data)
        self.cfg.ml_preprocessing.min_snr_detector_db = float('nan')
        with self.assertRaisesRegex(ValueError, 'min_snr_detector_db'):
            self.process(self.fixture())


if __name__ == '__main__':
    unittest.main()
