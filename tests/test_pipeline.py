from dataclasses import replace
import numpy as np
import pytest
from autoacoustics.model import (SignalData, ChannelInfo, AnalysisSettings,
    CalibrationProfile, MeasurementContext, ValidationError, CancellationToken, CancelledError)

def source(unit='Pa', fs=48000, seconds=.2):
    x=np.sqrt(2)*np.sin(2*np.pi*1000*np.arange(round(fs*seconds))/fs)
    return SignalData(x[None],fs,(ChannelInfo('mic',unit),),source_hash='testhash')

def test_physical_analysis_units_context_detail_and_summary_are_consistent():
    from autoacoustics.analysis.pipeline import analyze
    sig=source()
    settings=AnalysisSettings(channel=0,compute_loudness=False)
    full=analyze(sig,None,settings,MeasurementContext(supply='12 V'))
    summary=analyze(sig,None,settings,MeasurementContext(supply='12 V'),include_detail=False)
    assert abs(full.summary.metrics['LZeq'].value-93.97940008672)<1e-6
    assert full.summary.metrics['LZeq'].unit=='dB re 20 µPa'
    assert full.context.supply=='12 V' and full.input_hash=='testhash'
    assert full.result_id==summary.result_id and summary.detail is None
    assert dict(full.summary.metrics)==dict(summary.summary.metrics)
    assert len(full.detail.arrays['psd'])==len(full.detail.arrays['psd_frequency'])
    assert full.detail.arrays['stft_db'].shape==(len(full.detail.arrays['stft_frequency']),len(full.detail.arrays['stft_time']))

def test_uncalibrated_and_nonacoustic_inputs_keep_useful_spectrum_without_fake_spl():
    from autoacoustics.analysis.pipeline import analyze
    for unit in ('FS','V','g','unknown'):
        result=analyze(source(unit),None,AnalysisSettings(channel=0))
        assert result.summary.metrics['LAeq'].value is None
        assert result.summary.metrics['Nmax'].value is None
        assert result.summary.metrics['LAeq'].status=='uncalibrated' if unit in ('FS','V') else result.summary.metrics['LAeq'].status=='not_applicable'
        assert result.detail.units['waveform']==unit
        assert np.isfinite(result.detail.arrays['amplitude']).all()

def test_silence_and_short_event_have_explicit_states():
    from autoacoustics.analysis.pipeline import analyze
    sig=SignalData(np.zeros((1,480)),48000,(ChannelInfo('silent','Pa'),))
    result=analyze(sig,None,AnalysisSettings(channel=0))
    assert result.summary.metrics['LAeq'].value==float('-inf')
    assert result.summary.metrics['LAeq'].status=='silent'
    assert result.summary.metrics['Nmax'].value==0
    assert 'stft_db' not in result.detail.arrays
    assert any('STFT' in w for w in result.summary.warnings)

def test_event_uses_full_record_filter_state_and_original_clock():
    from autoacoustics.analysis.pipeline import analyze
    sig=source(seconds=.8)
    settings=AnalysisSettings(channel=0,start=24000,end=30000,compute_loudness=False)
    result=analyze(sig,None,settings)
    from autoacoustics.analysis.spl import compute_spl
    expected=compute_spl(sig.samples[0],sig.sample_rate,24000,30000)
    assert result.summary.metrics['LASmax'].value==expected['LASmax']
    assert result.detail.arrays['wave_time'][0]==.5
    assert result.detail.arrays['spl_time'][0]>=.5
    assert result.provenance['preprocessing']['scope']=='full_recording'

def test_invalid_selected_profile_fails_and_cancellation_is_not_success():
    from autoacoustics.analysis.pipeline import analyze
    sig=source('FS')
    profile=CalibrationProfile('other','sensor',1,'V')
    with pytest.raises(Exception,match='单位'):
        analyze(sig,profile,AnalysisSettings(channel=0))
    token=CancellationToken();token.cancel()
    with pytest.raises(CancelledError):
        analyze(sig,None,AnalysisSettings(channel=0),cancel=token)

def test_nonzero_source_origin_maps_every_output_axis_without_padding_samples():
    from autoacoustics.analysis.pipeline import analyze
    sig=replace(source(seconds=.2),metadata={'time_origin_seconds':.25})
    result=analyze(sig,None,AnalysisSettings(channel=0,start=4800,end=9600))
    arrays=result.detail.arrays
    assert arrays['wave_time'][0]==.35
    assert arrays['spl_time'][0]>=.35
    assert arrays['loudness_time'][0]>=.35
    assert arrays['stft_time'][0]>.35
    assert result.provenance['event_start_time_seconds']==.35
    assert result.settings.start==4800

def test_incomplete_recording_can_be_viewed_but_not_reported_as_valid_measurement():
    from autoacoustics.analysis.pipeline import analyze
    sig=replace(source(),quality=('incomplete_recording',))
    with pytest.raises(ValidationError,match='录制未完成'):
        analyze(sig,None,AnalysisSettings(channel=0))

def test_one_sample_event_retains_level_and_waveform_when_fft_is_unavailable():
    from autoacoustics.analysis.pipeline import analyze
    sig=source(seconds=.1)
    result=analyze(sig,None,AnalysisSettings(channel=0,start=100,end=101,compute_loudness=False))
    assert result.summary.metrics['duration'].value==1/48000
    assert result.detail.arrays['waveform'].shape==(1,)
    assert 'spectrum_db' not in result.detail.arrays
    assert result.summary.metrics['LZeq'].value is not None
    assert any('FFT' in text for text in result.summary.warnings)

def test_loudness_component_failure_keeps_spl_and_other_details(monkeypatch):
    import autoacoustics.analysis.pipeline as module
    def fail(*args,**kwargs):raise RuntimeError('component failed')
    monkeypatch.setattr(module,'compute_loudness',fail)
    result=module.analyze(source(),None,AnalysisSettings(channel=0))
    assert result.summary.metrics['LZeq'].status=='ok'
    assert result.summary.metrics['Nmax'].status=='failed'
    assert result.summary.metrics['Nmax'].value is None
    assert 'spectrum_db' in result.detail.arrays
    assert any('component failed' in text for text in result.summary.warnings)

def test_same_file_different_dataset_unit_rate_and_origin_have_distinct_identity():
    from autoacoustics.analysis.pipeline import analyze
    settings=AnalysisSettings(channel=0,compute_loudness=False)
    first=source()
    alternatives=(replace(first,samples=first.samples*2,metadata={'selected_dataset':'second'}),
                  replace(first,sample_rate=96000),
                  replace(first,metadata={'time_origin_seconds':.5}))
    original=analyze(first,None,settings,include_detail=False)
    assert all(analyze(item,None,settings,include_detail=False).result_id!=original.result_id for item in alternatives)

