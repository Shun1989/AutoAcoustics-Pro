import numpy as np
import pytest
from scipy.signal import sosfreqz
from autoacoustics.analysis.octave import octave_bands, octave_filter, compute_octave


def test_exact_centers_edges_and_response():
    bands=octave_bands(48000)
    assert 1000 in bands['nominal']
    i=list(bands['nominal']).index(1000)
    assert bands['exact'][i]==pytest.approx(1000)
    assert bands['lower'][i]==pytest.approx(1000*10**(-.05))
    sos=octave_filter(bands['exact'][i],48000)
    _,h=sosfreqz(sos,worN=[bands['lower'][i],1000,bands['upper'][i]],fs=48000)
    assert 20*np.log10(abs(h[1]))==pytest.approx(0,abs=.01)
    assert 20*np.log10(abs(h[0]))==pytest.approx(-3.0103,abs=.001)
    assert 20*np.log10(abs(h[2]))==pytest.approx(-3.0103,abs=.001)
    _,adjacent=sosfreqz(sos,worN=[1000*10**(.1)],fs=48000)
    assert 20*np.log10(abs(adjacent[0])) < -20


def test_physical_tone_and_inadequate_sample_rate():
    fs=48000
    x=np.sqrt(2)*np.sin(2*np.pi*1000*np.arange(fs*2)/fs)
    result=compute_octave(x,fs,start=fs,end=fs*2)
    i=list(result['nominal']).index(1000)
    assert result['levels'][i]==pytest.approx(93.9794,abs=.02)
    low=octave_bands(8000)
    assert np.max(low['upper']) < .9*4000


def test_startup_reference_discloses_missing_history_without_trimming_samples():
    fs=48000
    x=np.sqrt(2)*np.sin(2*np.pi*1000*np.arange(2*fs)/fs)
    start=compute_octave(x,fs,start=0,end=fs)
    later=compute_octave(x,fs,start=fs,end=2*fs)
    assert 25 in start['startup_reference']['affected_nominal_hz']
    assert later['startup_reference']['affected_nominal_hz']==()
    assert start['startup_reference']['certified_settling_time'] is False
    assert start['startup_reference']['start_context_seconds']==0
    assert later['startup_reference']['start_context_seconds']==1
    # There is no automatic 5/BW crop or filter reset: keep the complete event.
    from scipy.signal import sosfilt
    band=list(start['nominal']).index(1000)
    expected=10*np.log10(np.mean(sosfilt(octave_filter(1000,fs),x)[:fs]**2)/(20e-6)**2)
    assert start['levels'][band]==pytest.approx(expected,abs=1e-12)


def test_white_noise_1khz_band_matches_independent_order4_equivalent_bandwidth():
    from scipy.integrate import quad
    # Oracle: |H_LP(u)|²=1/(1+u^8), the analogue bandpass substitution,
    # and bilinear prewarping. No butter(), SOS, product filter or product
    # frequency response enters this integration.
    fs=48000
    lower=1000*10**(-.05)
    upper=1000*10**(.05)
    warped_lower=np.tan(np.pi*lower/fs)
    warped_upper=np.tan(np.pi*upper/fs)
    def power_response(frequency):
        if frequency<=0 or frequency>=fs/2:
            return 0.
        warped=np.tan(np.pi*frequency/fs)
        u=(warped**2-warped_lower*warped_upper)/((warped_upper-warped_lower)*warped)
        return 1/(1+u**8)
    bandwidth,error=quad(power_response,0,fs/2,points=[lower,1000,upper],
                         epsabs=1e-8,epsrel=1e-10,limit=200)
    squared_bandwidth,_=quad(lambda f:power_response(f)**2,0,fs/2,
                    points=[lower,1000,upper],epsabs=1e-8,epsrel=1e-10,limit=200)
    assert error<1e-5
    sigma=.02  # Pa RMS, independent white samples; one-sided PSD=2*sigma²/fs.
    expected_power=2*sigma**2/fs*bandwidth
    expected_level=10*np.log10(expected_power/(20e-6)**2)
    samples=np.random.Generator(np.random.PCG64(532104)).normal(0,sigma,fs*16)
    # Filter the whole input, then discard 2 s of startup. Four sigma in the
    # 14 s Gaussian power average is <0.30 dB using integral |H|^4; keep this
    # preselected statistical gate rather than tuning it after observing data.
    relative_sigma=np.sqrt(squared_bandwidth/(14*bandwidth**2))
    assert 10*np.log10(1+4*relative_sigma)<.30
    result=compute_octave(samples,fs,start=fs*2,end=fs*16)
    index=list(result['nominal']).index(1000)
    assert result['levels'][index]==pytest.approx(expected_level,abs=.30)
