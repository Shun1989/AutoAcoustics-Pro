import pickle
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np

from autoacoustics.model import (
    AnalysisDetail, AnalysisSettings, BatchItemResult, BatchItemSpec, CalibrationProfile, ChannelInfo,
    ImportMapping, MeasurementContext, SignalData, ValidationError,
)


class ModelTests(unittest.TestCase):
    def signal(self):
        return SignalData(np.zeros((2, 100)), 48000,
                          (ChannelInfo('mic', 'Pa'), ChannelInfo('accel', 'g')),
                          metadata={'nested': {'number': 1}})

    def test_shape_units_and_original_array_are_preserved(self):
        original = np.ones((2, 20))
        signal = SignalData(original, 51200,
                            (ChannelInfo('mic', 'FS'), ChannelInfo('voltage', 'V')))
        original[0, 0] = 99
        self.assertEqual(signal.samples[0, 0], 1)
        self.assertEqual(signal.frames, 20)
        self.assertEqual(signal.channel_count, 2)
        self.assertEqual(signal.channels[1].unit, 'V')
        self.assertAlmostEqual(signal.duration, 20 / 51200)
        with self.assertRaises(ValueError):
            signal.samples[0, 0] = 5

    def test_invalid_shape_rate_channel_metadata_and_samples_fail(self):
        for samples, rate, channels in [
            (np.zeros(10), 48000, (ChannelInfo('mic', 'FS'),)),
            (np.zeros((1, 10)), 0, (ChannelInfo('mic', 'FS'),)),
            (np.zeros((2, 10)), 48000, (ChannelInfo('mic', 'FS'),)),
            (np.array([[np.nan]]), 48000, (ChannelInfo('mic', 'FS'),)),
            (np.zeros((1, 0)), 48000, (ChannelInfo('mic', 'FS'),)),
        ]:
            with self.assertRaises(ValidationError):
                SignalData(samples, rate, channels)
        with self.assertRaises(ValidationError):
            ChannelInfo('mic', 'dB')

    def test_settings_require_explicit_channel_and_original_sample_bounds(self):
        signal = self.signal()
        with self.assertRaises(ValidationError):
            AnalysisSettings().resolve(signal)
        for setting in [AnalysisSettings(channel=2), AnalysisSettings(channel=0, start=-1),
                        AnalysisSettings(channel=0, start=10, end=10),
                        AnalysisSettings(channel=0, end=101)]:
            with self.assertRaises(ValidationError):
                setting.resolve(signal)
        self.assertEqual(AnalysisSettings(channel=1, start=10, end=20).resolve(signal), (1, 10, 20))

    def test_snapshots_are_deeply_immutable_and_worker_picklable(self):
        signal = self.signal()
        with self.assertRaises(TypeError):
            signal.metadata['nested']['number'] = 2
        with self.assertRaises(FrozenInstanceError):
            signal.sample_rate = 1
        restored = pickle.loads(pickle.dumps(signal))
        self.assertEqual(restored.metadata['nested']['number'], 1)
        self.assertEqual(restored.channels[1].unit, 'g')
        self.assertFalse(restored.samples.flags.writeable)

    def test_batch_items_have_identity_for_same_path_and_different_events(self):
        first = BatchItemSpec(Path('same.wav'), AnalysisSettings(channel=0, end=10))
        second = BatchItemSpec(Path('same.wav'), AnalysisSettings(channel=0, start=10, end=20))
        self.assertNotEqual(first.item_id, second.item_id)

    def test_failed_batch_retains_immutable_input_specification(self):
        specification = BatchItemSpec(Path('missing.wav'), AnalysisSettings(channel=1, start=10, end=20),
                                      context=MeasurementContext(supply='12 V'))
        row = BatchItemResult(specification.item_id, specification.path, 'failed',
                              error='missing', specification=specification)
        self.assertEqual(pickle.loads(pickle.dumps(row)).specification.context.supply, '12 V')
        self.assertEqual(row.specification.settings.channel, 1)

    def test_time_unit_mapping_is_explicit_and_validated(self):
        self.assertEqual(ImportMapping(time_column='time', time_unit='ms').time_unit, 'ms')
        self.assertIsNone(ImportMapping().time_unit)
        with self.assertRaises(ValidationError):
            ImportMapping(time_unit='minute')

    def test_source_time_origin_is_preserved_and_must_be_finite(self):
        signal=SignalData(np.zeros((1,100)),48000,(ChannelInfo('mic','Pa'),),
                          metadata={'time_origin_seconds':.25})
        self.assertEqual(signal.time_origin,.25)
        unknown=SignalData(np.zeros((1,100)),48000,(ChannelInfo('mic','FS'),),
                           metadata={'time_origin_seconds':None})
        self.assertEqual(unknown.time_origin,0.)
        with self.assertRaises(ValidationError):
            SignalData(np.zeros((1,100)),48000,(ChannelInfo('mic','Pa'),),
                       metadata={'time_origin_seconds':float('nan')})

    def test_profile_and_context_validate_without_guessing_direction(self):
        context = MeasurementContext(motor_rotation='CW', actual_movement='unknown')
        self.assertEqual(context.actual_movement, 'unknown')
        for value in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValidationError):
                CalibrationProfile('test', 'HEAD Pa reference', value, 'FS')

    def test_detail_public_constructor_isolates_but_internal_owner_moves_arrays(self):
        original = np.arange(20.)
        public = AnalysisDetail({'waveform': original}, {'waveform': 'Pa'})
        original[0] = 99
        self.assertEqual(public.arrays['waveform'][0], 0)
        owned = np.arange(20.)
        detail = AnalysisDetail._from_owned_arrays({'waveform': owned}, {'waveform': 'Pa'})
        self.assertIs(detail.arrays['waveform'], owned)
        self.assertFalse(owned.flags.writeable)
        with self.assertRaises(ValueError):
            detail.arrays['waveform'][0] = 2

    def test_detail_pickle_restores_readonly_without_another_array_copy(self):
        from unittest.mock import patch
        detail = AnalysisDetail({'waveform': np.arange(20.)}, {'waveform': 'Pa'})
        serialized = pickle.dumps(detail, protocol=pickle.HIGHEST_PROTOCOL)
        with patch('autoacoustics.model.readonly_array', side_effect=AssertionError('duplicate copy')):
            restored = pickle.loads(serialized)
        np.testing.assert_array_equal(restored.arrays['waveform'], detail.arrays['waveform'])
        self.assertFalse(restored.arrays['waveform'].flags.writeable)
        self.assertFalse(np.shares_memory(restored.arrays['waveform'], detail.arrays['waveform']))


if __name__ == '__main__':
    unittest.main()
