"""Apply exactly one traceable scale conversion to the selected acoustic channel."""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .model import (CalibrationError, CalibrationProfile, ChannelInfo,
                    PhysicalSignal, SignalData, to_plain)
from .project import write_json_atomic


def _channel(signal, profile=None, channel=None):
    if channel is None and profile is not None:
        channel = profile.channel
    if channel is None:
        if signal.channel_count != 1:
            raise CalibrationError('多通道校准必须明确选通道。')
        channel = 0
    if not isinstance(channel, int) or not 0 <= channel < signal.channel_count:
        raise CalibrationError('校准通道不存在。')
    return channel


def apply_calibration(signal: SignalData, profile: CalibrationProfile | None,
                      *, channel: int | None = None) -> PhysicalSignal:
    channel = _channel(signal, profile, channel)
    info = signal.channels[channel]
    if info.unit == 'Pa':
        coefficient = 1.0
        provenance = {'source': '文件声明 Pa', 'coefficient': 1.0, 'input_unit': 'Pa',
                      'scaled_once': True, 'ignored_profile': profile.profile_id if profile else None}
    else:
        if info.unit not in {'FS', 'V'}:
            raise CalibrationError('所选通道不是可换算声压的 FS/V/Pa。')
        if profile is None:
            raise CalibrationError('未选择有来源的校准配置，不能计算绝对声压。')
        if profile.input_unit != info.unit:
            raise CalibrationError('校准输入单位与通道不匹配。')
        if profile.channel is not None and profile.channel != channel:
            raise CalibrationError('校准配置的通道编号不匹配。')
        if profile.channel_name and profile.channel_name != info.name:
            raise CalibrationError('校准配置的通道名称不匹配。')
        if profile.applicable_hashes and signal.source_hash not in profile.applicable_hashes:
            raise CalibrationError('本录音不属于该校准配置的适用数据集。')
        coefficient = profile.coefficient
        provenance = to_plain(profile)
        provenance['scaled_once'] = True
    physical_info = ChannelInfo(info.name, 'Pa', 'sound_pressure', True,
        info.full_scale * coefficient if info.full_scale is not None else None,
        info.sensor_ref, info.metadata)
    metadata = {**signal.metadata, 'calibration': provenance, 'original_channel_index': channel}
    return PhysicalSignal(signal.samples[channel:channel+1] * coefficient, signal.sample_rate,
        (physical_info,), signal.path, signal.source_hash, metadata, signal.quality)


def make_sensitivity_profile(name: str, sensitivity_mv_pa: float, gain: float = 1.0,
                             full_scale_v: float | None = None, input_unit: str = 'V',
                             channel: int | None = None, channel_name: str | None = None,
                             **details) -> CalibrationProfile:
    if not math.isfinite(sensitivity_mv_pa) or sensitivity_mv_pa <= 0 or not math.isfinite(gain) or gain <= 0:
        raise CalibrationError('灵敏度与线性增益必须为有限正数。')
    if input_unit not in {'FS', 'V'}:
        raise CalibrationError('传感器链换算只接受 V 或已知 V/FS 的数字信号。')
    if input_unit == 'FS' and (full_scale_v is None or not math.isfinite(full_scale_v) or full_scale_v <= 0):
        raise CalibrationError('数字信号须填写采集链的 V/full-scale。')
    scale = (full_scale_v if input_unit == 'FS' else 1.0) / (sensitivity_mv_pa * .001 * gain)
    return CalibrationProfile(name, '传感器链换算', scale, input_unit,
        channel=channel, channel_name=channel_name, mode='sensor_chain',
        date=datetime.now(timezone.utc).isoformat(), details={**details,
            'sensitivity_mv_pa': sensitivity_mv_pa, 'linear_gain': gain,
            'full_scale_v': full_scale_v, 'formula': 'Pa = raw × V/raw ÷ (mV/Pa × 0.001 × gain)'})


def make_calibrator_profile(signal: SignalData, known_level_db: float = 94.0,
                             channel: int | None = None, start: int = 0,
                             end: int | None = None, frequency: float = 1000.0,
                             name: str = '声校准音配置') -> CalibrationProfile:
    channel = _channel(signal, channel=channel)
    end = signal.frames if end is None else end
    if not 0 <= start < end <= signal.frames or (end-start) / signal.sample_rate < .5:
        raise CalibrationError('校准音须选择至少 0.5 秒的有效样本区间。')
    if not math.isfinite(known_level_db) or not 0 < frequency < signal.sample_rate/2:
        raise CalibrationError('参考声级或频率无效。')
    info = signal.channels[channel]
    if info.unit not in {'FS', 'V', 'Pa'}:
        raise CalibrationError('此物理量不适用声校准。')
    if 'possible_clipping' in signal.quality:
        raise CalibrationError('校准录音存在数字触轨，请检查量程后重新记录。')
    raw = signal.samples[channel, start:end]
    centered = raw - np.mean(raw)
    rms = float(np.sqrt(np.mean(centered ** 2)))
    if rms <= np.finfo(float).tiny:
        raise CalibrationError('校准音为静音或没有交流分量。')
    power = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
    axis = np.fft.rfftfreq(len(centered), 1 / signal.sample_rate)
    peak = int(np.argmax(power[1:])) + 1
    fraction = float(np.sum(power[max(1, peak-2):peak+3]) / np.sum(power[1:]))
    if abs(axis[peak]-frequency) > max(5, frequency*.02) or fraction < .7:
        raise CalibrationError('所选录音不是指定频率的稳定校准音。')
    reference_pa = 20e-6 * 10 ** (known_level_db/20)
    coefficient = 1.0 if info.unit == 'Pa' else reference_pa / rms
    return CalibrationProfile(name, '独立声校准检查' if info.unit == 'Pa' else '独立声校准音换算',
        coefficient, info.unit, channel=channel, channel_name=info.name,
        date=datetime.now(timezone.utc).isoformat(), mode='calibrator', details={
            'known_level_db': known_level_db, 'reference_frequency_hz': frequency,
            'reference_rms': rms, 'reference_input_hash': signal.source_hash,
            'start': start, 'end': end, 'dc_removed_for_reference': True,
            'measured_level_db': 20*np.log10(rms/20e-6) if info.unit == 'Pa' else None,
            'tone_fraction': fraction,
            'tone_guard': '0.5 s minimum, ±max(5 Hz, 2%), main-lobe energy ≥70%; engineering input guard'})


def save_profile(profile: CalibrationProfile, destination: Path) -> Path:
    return write_json_atomic({'schema_version': 1, 'profile': to_plain(profile)}, destination)


def load_profile(path: Path) -> CalibrationProfile:
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if data['schema_version'] != 1:
            raise CalibrationError('不支持此校准配置版本。')
        return CalibrationProfile(**data['profile'])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CalibrationError(f'校准配置读取失败：{exc}') from exc
