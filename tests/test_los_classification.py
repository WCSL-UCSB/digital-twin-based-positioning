"""Interaction-based LoS classification and propagation through the pipeline."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
from scipy.constants import speed_of_light

from src.utils.path_classification import los_from_interactions, LOS_CLASSIFICATION
from src.module_2_path_refinement import filter_and_adjust_paths as refinement
from src.module_2_path_refinement import apply_link_budget as calibration
from src.module_3_channel_gen import channel_taps_gen as channel


class LosClassificationTests(unittest.TestCase):
    def test_padding_empty_links_and_all_interaction_types(self):
        interactions = np.zeros((2, 1, 6, 3), dtype=np.int16)
        interactions[0, 0, 1:5, 0] = [1, 2, 4, 8]
        counts = np.array([[1, 1, 1, 1, 1, 0]])
        np.testing.assert_array_equal(los_from_interactions(interactions, counts),
                                      [[True, False, False, False, False, False]])

    def test_zero_depth_and_zero_path_axes(self):
        np.testing.assert_array_equal(los_from_interactions(np.zeros((0, 1, 2, 2)),
                                                            np.array([[1, 0]])), [[True, False]])
        np.testing.assert_array_equal(los_from_interactions(np.zeros((2, 1, 2, 0)),
                                                            np.zeros((1, 2), dtype=int)), [[False, False]])

    def test_missing_or_malformed_evidence_fails(self):
        with self.assertRaisesRegex(ValueError, 'raw paths with interactions'):
            los_from_interactions(None, np.array([[1]]))
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            los_from_interactions(np.zeros((1, 1, 1)), np.array([[1]]))
        for counts in (np.array([[3]]), np.array([[-1]]), np.array([[.5]])):
            with self.subTest(counts=counts), self.assertRaisesRegex(ValueError, 'counts'):
                los_from_interactions(np.zeros((2, 1, 1, 2)), counts)


class LosPipelineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.raw = self.root / 'raw.npz'
        self.expected = np.array([[True, False, False, False, False, True]])
        counts = np.array([[2, 1, 1, 2, 0, 2]])
        self.interactions = np.zeros((2, 1, 6, 3), dtype=np.int16)
        self.interactions[0, 0, 0, 1] = 1  # direct + reflected
        self.interactions[:, 0, 1, 0] = 4  # wall transmission
        self.interactions[0, 0, 2, 0] = 1  # short reflection
        self.interactions[0, 0, 3, 1] = 1  # direct path gets filtered out
        self.interactions[0, 0, 5, 0] = 1  # direct path is not the first slot
        shape = (1, 6, 3)
        phi = np.zeros(shape)
        phi[0, 3, 0] = np.pi
        a = (np.arange(3) + 1.) * (np.arange(3) < counts[..., None])
        tau = np.broadcast_to(10 / speed_of_light * 1e9 + np.arange(3)*5., shape).copy()
        tau[0, 2, 0] += .25  # within old LoS tolerance, but is a reflection
        np.savez(self.raw, a=a.astype(complex), tau_ns=tau,
                 theta_r=np.full(shape, np.pi/2), phi_r=phi,
                 theta_t=np.full(shape, np.pi/2), phi_t=np.zeros(shape),
                 num_valid_paths=counts, interactions=self.interactions,
                 rx_names=np.array(['rx']), tx_names=np.array([f'tx{i}' for i in range(6)]),
                 rx_pos=np.array([[0., 0., 0.]]), tx_pos=np.tile([10., 0., 0.], (6, 1)),
                 rx_look_at=np.array([[1., 0., 0.]]), tx_look_at=np.full((6, 3), np.nan),
                 metadata={'frequency_hz':14e9})
        self.cfg = OmegaConf.create(dict(project_dir=str(self.root), scenario_name='los_test', frequency=14e9,
            inputs={'paths_npz':str(self.raw)},
            rx_array_geometry=dict(num_rows=1,num_cols=2,horizontal_spacing=.5,vertical_spacing=.5,
                                   fov_az_deg=120.,fov_el_deg=180.),
            link_budget=dict(tx_power_dbm=30.,bandwidth_hz=300e6,noise_figure_db=5.,temperature_k=290.),
            channel_params=dict(dt=3.33e-9,pulse=dict(type='sinc',Tsym=3.33e-9,beta=0.),
                                include_true_time_delay=True,include_spatial_phase=True),
            training_sequence={'max_delay_bins':8, 'length':31}))
        stdout = contextlib.redirect_stdout(io.StringIO())
        stdout.__enter__()
        self.addCleanup(stdout.__exit__, None, None, None)

    def run_stage(self, module, pattern):
        with patch.object(module, 'write_sidecar'):
            module.main.__wrapped__(self.cfg)
        output = sorted(self.root.glob(pattern))[-1]
        with np.load(output, allow_pickle=True) as data:
            return output, {key:data[key] for key in data.files}

    def upstream(self):
        path, filtered = self.run_stage(refinement, 'data/path_data/refined_paths/*.npz')
        self.cfg.inputs.filtered_npz = str(path)
        path, calibrated = self.run_stage(calibration, 'data/path_data/calibrated_paths/*.npz')
        self.cfg.inputs.calibrated_npz = str(path)
        return filtered, calibrated

    def test_filter_then_calibration_then_channel_preserves_labels(self):
        original = self.raw.read_bytes()
        filtered, calibrated = self.upstream()
        _, channels = self.run_stage(channel, 'data/channel_data/*.npz')
        for result in (filtered, calibrated, channels):
            np.testing.assert_array_equal(result['is_los_kept'], self.expected)
        np.testing.assert_array_equal(filtered['num_paths_after_filter'], [[2,1,1,1,0,2]])
        self.assertEqual(filtered['metadata'].item()['los_classification'], LOS_CLASSIFICATION)
        self.assertEqual(channels['metadata'].item()['los_label_source'], 'upstream_is_los_kept')
        self.assertEqual(channels['metadata'].item()['nlos_only_count'], 3)
        self.assertEqual(channels['shift_applied'][0, 1], 0.)  # NLoS can have zero excess
        self.assertAlmostEqual(channels['shift_applied'][0, 2], .25)
        self.assertEqual(self.raw.read_bytes(), original)

    def test_legacy_interactions_recover_the_same_labels_and_taps(self):
        _, calibrated = self.upstream()
        _, current = self.run_stage(channel, 'data/channel_data/*.npz')
        del calibrated['is_los_kept']
        legacy = self.root / 'legacy.npz'
        np.savez(legacy, **calibrated)
        before = legacy.read_bytes()
        self.cfg.inputs.calibrated_npz = str(legacy)
        _, recovered = self.run_stage(channel, 'data/channel_data/*.npz')
        np.testing.assert_array_equal(recovered['is_los_kept'], self.expected)
        np.testing.assert_array_equal(recovered['H'], current['H'])
        self.assertEqual(recovered['metadata'].item()['los_label_source'], 'legacy_surviving_interactions')
        self.assertEqual(legacy.read_bytes(), before)

    def test_channel_label_changes_do_not_change_taps(self):
        _, calibrated = self.upstream()
        _, correct = self.run_stage(channel, 'data/channel_data/*.npz')
        # Emulate an incorrect upstream label; synthesis must be independent.
        calibrated['is_los_kept'] = calibrated['num_paths_after_filter'] > 0
        no_interactions = self.root / 'labels_only.npz'
        del calibrated['interactions']
        np.savez(no_interactions, **calibrated)
        self.cfg.inputs.calibrated_npz = str(no_interactions)
        _, relabeled = self.run_stage(channel, 'data/channel_data/*.npz')
        np.testing.assert_array_equal(relabeled['H'], correct['H'])
        self.assertFalse(np.array_equal(relabeled['is_los_kept'], correct['is_los_kept']))

    def test_legacy_without_labels_or_interactions_requires_regeneration(self):
        _, calibrated = self.upstream()
        del calibrated['is_los_kept'], calibrated['interactions']
        legacy = self.root / 'legacy.npz'
        np.savez(legacy, **calibrated)
        self.cfg.inputs.calibrated_npz = str(legacy)
        with self.assertRaisesRegex(ValueError, 'raw paths with interactions'):
            self.run_stage(channel, 'data/channel_data/*.npz')
        self.assertFalse((self.root / 'data/channel_data').exists())

    def test_refinement_requires_interactions(self):
        with np.load(self.raw, allow_pickle=True) as data:
            values = {key:data[key] for key in data.files if key != 'interactions'}
        np.savez(self.raw, **values)
        with self.assertRaisesRegex(ValueError, 'raw paths with interactions'):
            self.run_stage(refinement, 'data/path_data/refined_paths/*.npz')
        self.assertFalse((self.root / 'data/path_data/refined_paths').exists())


if __name__ == '__main__':
    unittest.main()
