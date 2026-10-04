import numpy as np
import pytest
from autoacoustics.analysis.spectrum import compute_spectrum, compute_stft


@pytest.mark.parametrize('n', [4095, 4096])
def test_fft_single_sided_peak_dc_and_parseval(n):
    fs = float(n)
    t = np.arange(n)/fs
    x = 2*np.sin(2*np.pi*100*t)+.3
    result = compute_spectrum(x, fs, window='boxcar')
    assert result['amplitude'][100] == pytest.approx(2, abs=1e-12)
    assert result['amplitude'][0] == pytest.approx(.3, abs=1e-12)
    assert np.sum(result['energy_psd'])*(fs/n) == pytest.approx(np.mean(x*x), rel=1e-12)


def test_nyquist_is_not_doubled_and_padding_does_not_change_amplitude():
    x = (-1.)**np.arange(1000)
    result = compute_spectrum(x, 1000, window='boxcar')
    assert result['amplitude'][-1] == pytest.approx(1)
    assert compute_spectrum(x,1000,fft_size=2000,window='boxcar')['amplitude'][-1] == pytest.approx(1)


def test_stft_original_time_and_short_event_are_explicit():
    result = compute_stft(np.ones(4096),48000,1024,512,time_origin=.2)
    assert result['time'][0] == pytest.approx(.2+512/48000)
    assert result['window_seconds'] == pytest.approx(1024/48000)
    assert result['bin_spacing_hz'] == pytest.approx(48000/1024)
    with pytest.raises(ValueError):
        compute_stft(np.ones(20),48000,1024,512)


def test_silence_is_negative_infinity_not_zero():
    result = compute_spectrum(np.zeros(100),1000)
    assert np.isneginf(result['db']).all()


@pytest.mark.parametrize('window,gain,enbw',[('boxcar',1.,1.),('hann',.5,1.5),('hamming',.54,1.3628257887517146)])
def test_window_noise_bandwidth_uses_physical_window_not_zero_padded_grid(window,gain,enbw):
    # NI 322194C-01 Table10-3: rectangular1, Hann1.5, Hamming~1.36.
    # Zero padding refines the grid, but cannot reduce physical noise bandwidth.
    n=1024;fs=48000
    x=np.sin(2*np.pi*100*np.arange(n)/n)
    original=compute_spectrum(x,fs,window=window)
    padded=compute_spectrum(x,fs,fft_size=4*n,window=window)
    for result in (original,padded):
        assert result['window_coherent_gain']==pytest.approx(gain,abs=1e-14)
        assert result['window_enbw_bins']==pytest.approx(enbw,abs=1e-12)
        assert result['window_enbw_hz']==pytest.approx(enbw*fs/n,abs=1e-10)
        assert result['physical_window_bin_hz']==fs/n
    assert padded['bin_spacing_hz']==original['bin_spacing_hz']/4
    assert padded['amplitude'][400]==pytest.approx(original['amplitude'][100],rel=1e-12)


def test_noninteger_cycle_tone_has_dirichlet_leakage_and_conserves_power():
    # Catches a rounded-frequency/bin-only spectrum, doubled DC/Nyquist,
    # or a peak-amplitude normalization used for energy PSD.
    fs=n=4096
    frequency=137.35
    amplitude=.75
    phase=.41
    samples=amplitude*np.sin(2*np.pi*frequency*np.arange(n)/fs+phase)
    result=compute_spectrum(samples,fs,window='boxcar')
    peak=int(np.argmax(result['amplitude']))
    assert result['frequency'][peak]==137
    assert abs(result['frequency'][peak]-frequency)<=fs/(2*n)
    assert result['amplitude'][peak]<amplitude
    assert result['amplitude'][peak+1]>.15*amplitude

    # Analytic finite geometric sums, not an FFT or production helper.
    # The negative-frequency sine term matters even for these positive bins.
    def geometric_sum(angle):
        return np.exp(1j*angle*(n-1)/2)*np.sin(n*angle/2)/np.sin(angle/2)
    for bin_index in (peak-1,peak,peak+1):
        omega=2*np.pi*frequency/fs
        bin_omega=2*np.pi*bin_index/n
        coefficient=amplitude/(2j)*(np.exp(1j*phase)*geometric_sum(omega-bin_omega)
                        -np.exp(-1j*phase)*geometric_sum(-omega-bin_omega))
        assert result['amplitude'][bin_index]==pytest.approx(2*abs(coefficient)/n,rel=1e-11)
    assert np.sum(result['energy_psd'])*(fs/n)==pytest.approx(np.mean(samples**2),rel=1e-12)


@pytest.mark.parametrize('window',['boxcar','hann'])
def test_fixed_white_noise_full_and_welch_psd_integrate_to_time_energy(window):
    # No spectrum or Welch helper supplies the oracle. Parseval relates the
    # complete spectrum to window-weighted time energy, and averaging preserves
    # that identity for each Welch segment. Detects wrong fs/window-energy/2x.
    fs=48000
    n=262144
    segment=1024
    sigma=.2
    samples=np.random.Generator(np.random.PCG64(20261003)).normal(0,sigma,n)
    result=compute_spectrum(samples,fs,fft_size=segment,window=window)
    def weights(size):
        return np.ones(size) if window=='boxcar' else .5-.5*np.cos(2*np.pi*np.arange(size)/size)
    full_weights=weights(n)
    full_reference=np.dot(samples**2,full_weights**2)/np.dot(full_weights,full_weights)
    segment_weights=weights(segment)
    welch_reference=np.mean([np.dot(samples[start:start+segment]**2,segment_weights**2)
                 /np.dot(segment_weights,segment_weights)
                 for start in range(0,n-segment+1,segment//2)])
    full_integral=np.sum(result['energy_psd'])*(fs/n)
    welch_integral=np.sum(result['psd'])*(fs/segment)
    assert full_integral==pytest.approx(full_reference,rel=2e-12)
    assert welch_integral==pytest.approx(welch_reference,rel=2e-12)
    # Fixed Gaussian variance also bounds the physically expected integrated
    # density. The 2% gate is set in advance, comfortably above finite-record
    # variance fluctuations (~0.3% here), and is not fitted to product output.
    assert full_integral==pytest.approx(sigma**2,rel=.02)
    assert welch_integral==pytest.approx(sigma**2,rel=.02)
    assert len(result['psd_frequency'])==segment//2+1
    assert result['psd_frequency'][-1]==fs/2
    assert result['frequency'][1]-result['frequency'][0]==fs/n
