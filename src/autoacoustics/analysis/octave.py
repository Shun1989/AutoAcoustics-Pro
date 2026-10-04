"""Base-10 one-third octave Butterworth bands, without instrument-class claims."""
import numpy as np
from scipy.signal import butter, sosfilt

from .spl import level_from_power

NOMINAL=np.array([25,31.5,40,50,63,80,100,125,160,200,250,315,400,500,630,800,
                  1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,12500,16000])


def octave_bands(sample_rate):
    exact=1000*10**(np.arange(-16,13)/10)
    lower=exact*10**(-.05)
    upper=exact*10**(.05)
    valid=upper<.9*sample_rate/2
    return {'nominal':NOMINAL[valid],'exact':exact[valid],'lower':lower[valid],'upper':upper[valid]}


def octave_filter(center,sample_rate):
    return butter(4,[center*10**(-.05),center*10**(.05)],btype='bandpass',
                  fs=sample_rate,output='sos')


def compute_octave(samples_pa,sample_rate,start=0,end=None,cancel=None):
    x=np.asarray(samples_pa,dtype=float)
    end=len(x) if end is None else end
    if not 0<=start<end<=len(x):
        raise ValueError('倍频程事件区间无效。')
    bands=octave_bands(sample_rate)
    levels=[]
    for center in bands['exact']:
        if cancel is not None:
            cancel.check()
        filtered=sosfilt(octave_filter(center,sample_rate),x)
        levels.append(float(level_from_power(np.mean(filtered[start:end]**2))))
    reference_times=5/(bands['upper']-bands['lower'])
    startup={'rule':'5 / band width (Hz)', 'source':'NI 322194C-01 (2004), section 9-7; engineering comparison reference',
             'certified_settling_time':False,'start_context_seconds':start/sample_rate,
             'maximum_reference_seconds':float(np.max(reference_times)) if len(reference_times) else None,
             'affected_nominal_hz':tuple(float(center) for center in bands['nominal'][start/sample_rate<reference_times]),
             'automatic_trim_or_reset':False}
    return {**bands,'levels':np.asarray(levels),'method':'base-10, Butterworth prototype order 4',
            'initial_state':'zero at recording start','certified_instrument_class':None,'startup_reference':startup}
