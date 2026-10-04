"""Nominal A-weighting, physical Leq and continuous exponential time weighting."""
from functools import lru_cache

import numpy as np
from scipy.signal import fftconvolve, firwin2, lfilter

P0=20e-6


def level_from_power(power):
    with np.errstate(divide='ignore'):
        return 10*np.log10(np.asarray(power)/(P0*P0))


def a_weighting_amplitude(frequency):
    f=np.asarray(frequency,dtype=float)
    squared=f*f
    denominator=(squared+20.598997**2)*np.sqrt((squared+107.65265**2)*(squared+737.86223**2))*(squared+12194.217**2)
    raw=12194.217**2 * squared**2 / denominator
    # Normalize the prescribed analogue target exactly to 0 dB at 1 kHz.
    s=1000.**2
    normalization=12194.217**2*s*s/((s+20.598997**2)*np.sqrt((s+107.65265**2)*(s+737.86223**2))*(s+12194.217**2))
    return raw/normalization


@lru_cache(maxsize=8)
def a_weighting_fir(sample_rate):
    # Offline linear-phase FIR avoids bilinear high-frequency warping.
    # Full-record context is used; group delay is removed, zero extension is documented.
    taps=max(1025,2*int(np.ceil(sample_rate/3))+1)
    frequency=np.linspace(0,sample_rate/2,32769)
    coefficients=firwin2(taps,frequency,a_weighting_amplitude(frequency),
                         fs=sample_rate,window=('kaiser',8))
    coefficients.flags.writeable=False
    return coefficients


def time_weighted_power(samples,sample_rate,tau):
    alpha=np.exp(-1/(sample_rate*tau))
    return lfilter([1-alpha],[1,-alpha],np.asarray(samples,dtype=float)**2)


def compute_spl(samples_pa,sample_rate,start=0,end=None,history_step_s=.01):
    x=np.asarray(samples_pa,dtype=float)
    end=len(x) if end is None else end
    if x.ndim!=1 or sample_rate<=0 or not 0<=start<end<=len(x):
        raise ValueError('SPL 输入或事件区间无效。')
    coefficients=a_weighting_fir(float(sample_rate))
    delay=(len(coefficients)-1)//2
    weighted=fftconvolve(x,coefficients,mode='full')[delay:delay+len(x)]
    fast=np.maximum(time_weighted_power(weighted,sample_rate,.125),0)
    slow=np.maximum(time_weighted_power(weighted,sample_rate,1.),0)
    hop=max(1,round(sample_rate*history_step_s))
    indices=np.arange(((start+hop-1)//hop)*hop,end,hop,dtype=int)
    return {'LZeq':float(level_from_power(np.mean(x[start:end]**2))),
            'LAeq':float(level_from_power(np.mean(weighted[start:end]**2))),
            'LAFmax':float(level_from_power(np.max(fast[start:end]))),
            'LASmax':float(level_from_power(np.max(slow[start:end]))),
            'time':indices/sample_rate,'laf':level_from_power(fast[indices]),
            'las':level_from_power(slow[indices]),
            'metadata':{'a_method':'nominal-A linear-phase FIR, Kaiser beta=8',
                'a_taps':len(coefficients),'removed_group_delay_samples':delay,
                'boundary_extension':'zero outside full recording',
                'boundary_context_seconds':delay/sample_rate,
                'time_weighting_initial_power':0.,'history_step_seconds':hop/sample_rate}}
