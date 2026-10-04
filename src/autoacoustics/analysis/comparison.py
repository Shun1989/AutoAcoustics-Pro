"""Right minus left comparisons, preserving unavailable and physical states."""
from dataclasses import fields
import math
from ..model import ComparisonResult,MetricResult

def compare_results(left,right):
    differences={}
    for key in sorted(set(left.summary.metrics)|set(right.summary.metrics)):
        a=left.summary.metrics.get(key);b=right.summary.metrics.get(key)
        unit='dB' if a and a.unit.startswith('dB') and a.unit==getattr(b,'unit',None) else getattr(a,'unit',getattr(b,'unit',''))
        if a is None or b is None or a.value is None or b.value is None:
            difference=MetricResult(None,unit,'unavailable','right_minus_left','两侧没有可用的同名数值。')
        elif a.unit!=b.unit:
            difference=MetricResult(None,unit,'incompatible','right_minus_left','两侧单位不同。')
        elif not math.isfinite(a.value) and a.value==b.value:
            difference=MetricResult(None,unit,'undefined','right_minus_left','两侧均为真静音，差值未定义。')
        else:
            difference=MetricResult(b.value-a.value,unit,'estimated' if 'estimated' in (a.status,b.status) else 'ok',
                                    'right_minus_left')
        differences[key]=difference
    conditions=[]
    for item in fields(left.context):
        a=getattr(left.context,item.name);b=getattr(right.context,item.name)
        if a!=b: conditions.append(f'{item.name}: {a} → {b}')
        elif item.name!='notes' and a in ('unknown',''): conditions.append(f'{item.name}: unknown (两侧均未记录)')
    for name in ('sample_rate','time_origin_seconds','channel_name','source_unit'):
        a=left.provenance.get(name);b=right.provenance.get(name)
        if a!=b: conditions.append(f'{name}: {a} → {b}')
    if left.settings!=right.settings: conditions.append('analysis_settings: 两侧分析配置或事件不同。')
    if left.calibration!=right.calibration: conditions.append('calibration: 两侧校准来源或系数不同。')
    if left.provenance.get('versions')!=right.provenance.get('versions'): conditions.append('versions: 两侧算法依赖版本不同。')
    return ComparisonResult(left,right,differences,tuple(conditions))
