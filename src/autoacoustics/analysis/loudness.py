"""Physical loudness adapter: full recording context, original event clock."""
from fractions import Fraction
import numpy as np
from scipy.signal import resample_poly
from mosqito.sq_metrics import loudness_zwst
from ._loudness_engine import compute_loudness as _engine
from ..model import CancelledError, ValidationError

def _cancel_check(cancel):
    if cancel is not None:
        cancel.check()

def compute_stationary(samples_pa, sample_rate, field_type='free'):
    samples=np.asarray(samples_pa,dtype=float)
    estimated=sample_rate<48000
    if sample_rate < 48000:
        ratio=Fraction(48000/sample_rate).limit_denominator(100000)
        samples=resample_poly(samples,ratio.numerator,ratio.denominator)
        sample_rate=48000
    total,specific,bark=loudness_zwst(samples,sample_rate,field_type)
    return {'total':float(total),'specific':np.asarray(specific),'bark':np.asarray(bark),
            'status':'estimated' if estimated else 'ok'}

def compute_loudness(samples_pa, sample_rate, start=0, end=None,
                     field_type='free', stationary=False, cancel=None, include_specific=True):
    samples=np.asarray(samples_pa,dtype=float)
    end=len(samples) if end is None else end
    if not 0<=start<end<=len(samples):
        raise ValidationError('响度事件区间必须位于原录音中。')
    _cancel_check(cancel)
    try:
        result=_engine(samples,sample_rate,field_type,stationary=False,
                       cancel=(lambda: cancel.cancelled) if cancel is not None else None,
                       include_specific=include_specific)
    except InterruptedError as exc:
        raise CancelledError('响度计算已取消。') from exc
    times=result['time']
    selected=(times>=start/sample_rate)&(times<end/sample_rate)
    result['time']=times[selected]
    result['total']=result['total'][selected]
    if result['specific'] is not None:
        result['specific']=result['specific'][:,selected]
    result['status']='estimated' if sample_rate<48000 else 'ok'
    result['metadata']['event_start_sample']=start
    result['metadata']['event_end_sample']=end
    result['metadata']['processing_context']='full_recording'
    if stationary:
        _cancel_check(cancel)
        steady=compute_stationary(samples[start:end],sample_rate,field_type)
        result['stationary_total']=steady['total']
        result['stationary_specific']=steady['specific']
        result['metadata']['stationary_scope']='selected_event'
    _cancel_check(cancel)
    return result
