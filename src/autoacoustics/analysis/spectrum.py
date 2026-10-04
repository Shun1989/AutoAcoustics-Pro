"""Peak amplitude FFT and distinct energy-normalized PSD/STFT calculations."""
import numpy as np
from scipy.signal import get_window, stft, welch


def db_amplitude(amplitude, reference=1.0):
    with np.errstate(divide='ignore'):
        return 20*np.log10(np.asarray(amplitude)/reference)


def _window_metadata(weights, sample_rate):
    count=len(weights)
    noise_bandwidth=float(np.sum(weights**2)/np.sum(weights)**2)
    return {'window_coherent_gain':float(np.mean(weights)),
            'window_enbw_bins':count*noise_bandwidth,
            'window_enbw_hz':sample_rate*noise_bandwidth,
            'physical_window_bin_hz':sample_rate/count,
            'window_enbw_definition':'bins relative to fs/window_samples; Hz independent of FFT zero padding'}


def compute_spectrum(samples, sample_rate, fft_size=None, window='hann', reference=1.0):
    x=np.asarray(samples,dtype=float)
    if x.ndim!=1 or len(x)<2 or sample_rate<=0:
        raise ValueError('频谱需要至少两个样本和正采样率。')
    n=len(x)
    nfft=max(n,fft_size or n)
    weights=get_window(window,n,fftbins=True)
    transformed=np.fft.rfft(x*weights,n=nfft)
    factors=np.full(len(transformed),2.)
    factors[0]=1.
    if nfft%2==0:
        factors[-1]=1.
    amplitude=np.abs(transformed)*factors/np.sum(weights)
    energy_psd=(np.abs(transformed)**2)*factors/(sample_rate*np.sum(weights**2))
    segment=min(fft_size or 4096,n)
    psd_frequency,psd=welch(x,sample_rate,window=window,nperseg=segment,
                            noverlap=segment//2,detrend=False,scaling='density')
    return {'frequency':np.fft.rfftfreq(nfft,1/sample_rate),'amplitude':amplitude,
            'db':db_amplitude(amplitude,reference),'energy_psd':energy_psd,
            'psd_frequency':psd_frequency,'psd':psd,'fft_size':nfft,
            'window_samples':n,'bin_spacing_hz':sample_rate/nfft,
            'physical_window_seconds':n/sample_rate,'psd_segment_samples':segment,
            **_window_metadata(weights,sample_rate)}


def compute_stft(samples,sample_rate,window_size=1024,hop=512,window='hann',
                 time_origin=0.,reference=1.0):
    x=np.asarray(samples,dtype=float)
    if len(x)<window_size or not 0<hop<=window_size:
        raise ValueError('所选事件不足一个 STFT 窗；请选择更短窗口。')
    frequencies,times,z=stft(x,sample_rate,window=window,nperseg=window_size,
                            noverlap=window_size-hop,boundary=None,padded=False,
                            detrend=False,scaling='spectrum')
    factors=np.full(len(frequencies),2.)
    factors[0]=1.
    if window_size%2==0:
        factors[-1]=1.
    amplitude=np.abs(z)*factors[:,None]
    return {'frequency':frequencies,'time':times+time_origin,'amplitude':amplitude,
            'db':db_amplitude(amplitude,reference),'window_seconds':window_size/sample_rate,
            'bin_spacing_hz':sample_rate/window_size,'hop_seconds':hop/sample_rate,
            'window_samples':window_size,
            **_window_metadata(get_window(window,window_size,fftbins=True),sample_rate)}
