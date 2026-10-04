import tempfile
import unittest
from pathlib import Path

import numpy as np

from autoacoustics.calibration import (apply_calibration, load_profile,
    make_calibrator_profile, make_sensitivity_profile, save_profile)
from autoacoustics.model import (CalibrationError, CalibrationProfile, ChannelInfo,
    PhysicalSignal, SignalData)


class CalibrationTests(unittest.TestCase):
    def test_fs_conversion_provenance_and_original_are_preserved(self):
        signal = SignalData(np.array([[.1, -.1]]), 48000,
                            (ChannelInfo('mic', 'FS'),), source_hash='known')
        profile = CalibrationProfile('dataset', 'HEAD Pa reference', .8, 'FS', applicable_hashes=('known',))
        physical = apply_calibration(signal, profile)
        self.assertIsInstance(physical, PhysicalSignal)
        np.testing.assert_allclose(physical.samples, [[.08, -.08]], rtol=1e-15, atol=0)
        np.testing.assert_array_equal(signal.samples, [[.1, -.1]])
        self.assertEqual(physical.metadata['calibration']['source'], 'HEAD Pa reference')

    def test_voltage_and_sensor_gain_chain_and_multichannel_selection(self):
        signal = SignalData(np.array([[.1, -.1], [1, -1]]), 48000,
                            (ChannelInfo('mic', 'V'), ChannelInfo('accel', 'g')))
        profile = make_sensitivity_profile('PCB', 50, gain=2, input_unit='V', channel=0)
        physical = apply_calibration(signal, profile)
        np.testing.assert_allclose(physical.samples, [[1, -1]])
        self.assertEqual(signal.channels[1].unit, 'g')
        with self.assertRaises(CalibrationError):
            apply_calibration(signal, CalibrationProfile('x', 'manual', 1, 'V'))
        with self.assertRaises(CalibrationError):
            apply_calibration(signal, profile, channel=1)

    def test_pa_passes_once_and_incompatible_or_wrong_dataset_fails(self):
        pa = SignalData(np.ones((1, 100)), 48000, (ChannelInfo('mic', 'Pa', scaled=True),))
        physical = apply_calibration(pa, CalibrationProfile('x', 'manual FS', 80, 'FS'))
        np.testing.assert_array_equal(physical.samples, pa.samples)
        self.assertEqual(physical.metadata['calibration']['source'], '文件声明 Pa')
        fs = SignalData(np.ones((1, 100)), 48000, (ChannelInfo('mic', 'FS'),), source_hash='other')
        with self.assertRaises(CalibrationError):
            apply_calibration(fs, CalibrationProfile('x', 'dataset', .8, 'FS', applicable_hashes=('known',)))
        with self.assertRaises(CalibrationError):
            apply_calibration(fs, CalibrationProfile('x', 'sensor', 10, 'V'))

    def test_full_scale_and_independent_tone_profile_are_recorded(self):
        fs_profile = make_sensitivity_profile('mic', 50, full_scale_v=2, input_unit='FS')
        self.assertEqual(fs_profile.coefficient, 40)
        rate = 48000
        signal = SignalData((.25 * np.sqrt(2) * np.sin(2*np.pi*1000*np.arange(rate)/rate))[None,:],
                            rate, (ChannelInfo('mic', 'V'),), source_hash='calibrator-audio')
        profile = make_calibrator_profile(signal, known_level_db=93.9794000867, channel=0)
        self.assertAlmostEqual(profile.coefficient, 4, places=8)
        self.assertEqual(profile.details['reference_input_hash'], 'calibrator-audio')
        self.assertAlmostEqual(profile.details['reference_rms'], .25)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)/'校准.json'
            save_profile(profile, target)
            self.assertEqual(load_profile(target), profile)
        silent = SignalData(np.zeros((1, rate)), rate, (ChannelInfo('mic','FS'),))
        with self.assertRaises(CalibrationError):
            make_calibrator_profile(silent, channel=0)
        with self.assertRaises(CalibrationError):
            make_calibrator_profile(SignalData(np.random.default_rng(4).normal(size=(1,rate)),
                rate, (ChannelInfo('mic','V'),)), channel=0)


if __name__ == '__main__':
    unittest.main()
