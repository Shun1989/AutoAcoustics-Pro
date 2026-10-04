"""Existing external numerical curves, never produced by this application."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from autoacoustics.importers import import_signal
from autoacoustics.calibration import apply_calibration
from autoacoustics.model import CalibrationProfile
from autoacoustics.analysis.loudness import compute_loudness,compute_stationary

REF=Path(__file__).parent/'reference/upstream'

def physical(name):
    manifest=json.loads((REF/'manifest.json').read_text(encoding='utf-8'))
    record=next(f for f in manifest['files'] if f['local_file']==name)
    path=REF/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
    signal=import_signal(path)
    profile=CalibrationProfile('upstream reference','MoSQITo v1.2.1 fixed input scaling',
        2*np.sqrt(2)*32768/32767,'FS',applicable_hashes=(signal.source_hash,))
    return apply_calibration(signal,profile).samples[0],signal.sample_rate

def test_stationary_external_pinknoise_total_and_all_240_specific_values():
    x,fs=physical('stationary_pinknoise_60dB.wav')
    result=compute_stationary(x,fs)
    expected=np.loadtxt(REF/'stationary_signal5_specific.csv',skiprows=1,encoding='utf-8-sig')
    assert abs(result['total']-10.498)<=max(.1,.05*10.498)
    assert np.all(np.abs(result['specific']-expected)<=np.maximum(.1,.05*np.abs(expected)))
    np.testing.assert_allclose(result['bark'],np.arange(1,241)*.1,atol=1e-13)

@pytest.mark.parametrize('signal_id,name',[(7,'varying_tone1kHz_30_to_80dB.wav'),(10,'pulse_1kHz_10ms_70dB.wav')])
def test_time_varying_external_total_and_specific_curves_5_and_10_percent(signal_id,name):
    x,fs=physical(name)
    result=compute_loudness(x,fs)
    reference=np.genfromtxt(REF/f'time_varying_signal{signal_id}_reference.csv',delimiter=',',names=True)
    np.testing.assert_allclose(result['time'],reference['time_s'],atol=1e-14,rtol=0)
    n=result['total'];specific=result['specific'][84]
    # Freeze a stricter unshifted comparison. No per-point shifts or stretching.
    for values,prefix in [(n,'N'),(specific,'N_spec')]:
        lower=reference[prefix+'_lower_5'];upper=reference[prefix+'_upper_5']
        assert np.mean((values<lower)|(values>upper))<=.01
        lower10=reference[prefix+'_lower_10'];upper10=reference[prefix+'_upper_10']
        assert not np.any((values<lower10)|(values>upper10))
    expected_peak={7:15.9534,10:4.2998}[signal_id]
    assert abs(np.max(n)-expected_peak)<=max(.1,.05*expected_peak)
