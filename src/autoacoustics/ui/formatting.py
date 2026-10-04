import math

STATUS_LABELS = {'ok': '有效', 'silent': '静音', 'uncalibrated': '未校准',
    'failed': '失败', 'estimated': '估计', 'not_applicable': '不适用', 'cancelled': '已取消',
    'disabled':'未启用','unavailable':'不可用','incompatible':'条件不兼容','undefined':'未定义'}

METRIC_LABELS = {'LAeq': '平均 A 声级', 'LZeq': '平均 Z 声级',
    'LAFmax': '最大 Fast A 声级', 'LASmax': '最大 Slow A 声级',
    'Nmean': '时变响度时间平均', 'Nmax': '时变响度最大值',
    'Nstationary': '稳态响度', 'duration': '事件时长', 'RMS': '均方根', 'dBFS': '数字 RMS','RMS_dBFS':'数字 RMS'}

QUALITY_LABELS={'possible_clipping':'可能出现数字触轨','lossy_source':'有损编码来源',
    'unknown_frontend_overload_status':'前端过载状态未记录',
    'incomplete_recording':'录制不完整；仅供恢复查看，不能作为完整测量',
    'floating_full_scale_unknown':'浮点录音未声明采集满量程','unknown_source_unit':'源物理单位未知',
    'decoded_amplitude_exceeds_nominal_fs':'解码幅度超过约定数字满量程'}

def quality_text(values):return '；'.join(QUALITY_LABELS.get(value,value) for value in values)

def metric_text(metric):
    status = STATUS_LABELS.get(metric.status, metric.status)
    if metric.value is None or metric.status in {'failed', 'uncalibrated', 'not_applicable', 'cancelled'}:
        return f'— {status}'
    if math.isinf(metric.value):
        value = '−∞' if metric.value < 0 else '+∞'
    elif math.isnan(metric.value):
        return '— 失败'
    else:
        value = f'{metric.value:.4g}'
    return f'{value} {metric.unit} · {status}'
