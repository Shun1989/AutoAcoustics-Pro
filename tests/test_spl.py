import numpy as np
import pytest
from scipy.signal import freqz
from autoacoustics.analysis.spl import a_weighting_amplitude, a_weighting_fir, compute_spl, time_weighted_power


def test_known_pa_rms_and_silence():
    fs=48000
    x=np.sqrt(2)*np.sin(2*np.pi*1000*np.arange(fs*2)/fs)
    result=compute_spl(x,fs)
    assert result['LZeq']==pytest.approx(93.9794000867,abs=.0001)
    assert result['LAeq']==pytest.approx(93.9794000867,abs=.02)
    assert compute_spl(np.full(48000,20e-6),fs)['LZeq']==pytest.approx(0,abs=1e-12)
    silent=compute_spl(np.zeros(fs),fs)
    assert np.isneginf(silent['LZeq']) and np.isneginf(silent['LAeq'])


def test_nominal_a_curve_and_digital_response():
    frequencies=np.array([20,31.5,63,125,250,500,1000,2000,4000,8000,16000,20000])
    nominal=np.array([-50.5,-39.4,-26.2,-16.1,-8.6,-3.2,0,1.2,1.0,-1.1,-6.6,-9.3])
    assert np.max(np.abs(20*np.log10(a_weighting_amplitude(frequencies))-nominal))<.16
    taps=a_weighting_fir(48000)
    _,response=freqz(taps,worN=frequencies,fs=48000)
    # Product engineering gate, not a claim of IEC instrument classification.
    assert np.max(np.abs(20*np.log10(np.abs(response)/a_weighting_amplitude(frequencies))))<.15


@pytest.mark.parametrize('tau',[.125,1.0])
def test_fast_slow_step_and_full_context(tau):
    fs=48000
    power=time_weighted_power(np.ones(fs*2),fs,tau)
    times=(np.arange(len(power))+1)/fs
    np.testing.assert_allclose(power,1-np.exp(-times/tau),rtol=1e-10,atol=1e-11)
    full=compute_spl(np.ones(fs*2),fs)
    segment=compute_spl(np.ones(fs*2),fs,start=fs,end=fs*2)
    assert segment['LZeq']==full['LZeq']
    np.testing.assert_array_equal(segment['laf'],full['laf'][full['time']>=1])


def test_white_noise_laeq_matches_independent_analogue_a_frequency_energy():
    # Catches unweighted/partial-band LAeq, a power-vs-amplitude error, or
    # accidental use of Fast/Slow. Evaluate the nominal analogue A transfer
    # via its complex poles, not a_weighting_amplitude()/FIR/product PSD.
    fs=48000
    sigma=.02
    period=np.random.Generator(np.random.PCG64(532107)).normal(0,sigma,fs)
    frequencies=np.fft.fftfreq(len(period),1/fs)
    poles=2*np.pi*np.array([20.598997,107.65265,737.86223,12194.217])
    def analogue_transfer(frequency):
        s=2j*np.pi*frequency
        return poles[3]**2*s**4/((s+poles[0])**2*(s+poles[1])
                              *(s+poles[2])*(s+poles[3])**2)
    gain=np.abs(analogue_transfer(frequencies))/abs(analogue_transfer(1000))
    # Two-sided frequency-domain energy avoids one-sided endpoint factors.
    reference_power=np.sum(np.abs(np.fft.fft(period))**2*gain**2)/len(period)**2
    reference_level=10*np.log10(reference_power/(20e-6)**2)
    # Periodic random broadband input gives an exact finite spectral oracle.
    # Select two middle periods, 1 s from either edge (>0.3334 s FIR support),
    # eliminating outside-record zero padding/startup from this comparison.
    samples=np.tile(period,4)
    result=compute_spl(samples,fs,start=fs,end=3*fs)
    assert result['metadata']['boundary_context_seconds']<1
    assert result['LAeq']==pytest.approx(reference_level,abs=.05)
    assert result['LZeq']==pytest.approx(10*np.log10(np.mean(period**2)/(20e-6)**2),abs=1e-10)
    # 0.05 dB broadband energy is an additional gate. The existing 12-frequency
    # 0.15 dB response gate above remains unchanged; neither is IEC certification.
