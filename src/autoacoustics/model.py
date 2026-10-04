"""Shared, immutable measurement contracts. Samples always use channel × frame."""
from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import numpy as np


class AcousticError(Exception):
    """A user-visible error with a stable category."""
    code = 'acoustic_error'


class ValidationError(AcousticError):
    code = 'invalid_input'


class ImportError(AcousticError):
    code = 'import_failed'


class CalibrationError(AcousticError):
    code = 'calibration_failed'


class CancelledError(AcousticError):
    code = 'cancelled'


class MappingRequired(ImportError):
    code = 'mapping_required'

    def __init__(self, message, options=None):
        super().__init__(message)
        self.options = FrozenMap(options or {})


def readonly_array(value):
    array = np.array(value, copy=True)
    array.flags.writeable = False
    return array


class FrozenMap(Mapping):
    def __init__(self, values=None):
        self._values = {str(k): freeze(v) for k, v in (values or {}).items()}

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __reduce__(self):
        # Unpickling has already allocated independent arrays. Freezing those
        # buffers in place avoids a second copy of large worker results.
        return _restore_owned_map, (self._values,)


def _freeze_owned(value):
    """Freeze private buffers whose owner is transferring them to a snapshot."""
    if isinstance(value, Mapping):
        return _restore_owned_map(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_owned(v) for v in value)
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    return value


def _restore_owned_map(values):
    result = FrozenMap.__new__(FrozenMap)
    result._values = {str(k): _freeze_owned(v) for k, v in values.items()}
    return result


def freeze(value):
    if isinstance(value, Mapping):
        return FrozenMap(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, np.ndarray):
        return readonly_array(value)
    return value


def to_plain(value):
    if is_dataclass(value):
        return {f.name: to_plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        raise ValidationError('项目清单不得包含波形或图表数组；请保存详情缓存引用。')
    if isinstance(value, np.generic):
        return value.item()
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(to_plain(value), sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


UNITS = {'FS', 'V', 'Pa', 'm/s²', 'g', 'unknown'}


@dataclass(frozen=True)
class ChannelInfo:
    name: str
    unit: str = 'unknown'
    physical_quantity: str = 'unknown'
    scaled: bool = False
    full_scale: float | None = None
    sensor_ref: str | None = None
    metadata: Mapping = field(default_factory=FrozenMap)

    def __post_init__(self):
        if not self.name or self.unit not in UNITS:
            raise ValidationError('通道名称或单位无效。')
        if self.full_scale is not None and (not math.isfinite(self.full_scale) or self.full_scale <= 0):
            raise ValidationError('ADC 满量程必须为正数。')
        object.__setattr__(self, 'metadata', FrozenMap(self.metadata))


@dataclass(frozen=True)
class SignalData:
    samples: np.ndarray
    sample_rate: float
    channels: tuple[ChannelInfo, ...]
    path: Path | None = None
    source_hash: str = ''
    metadata: Mapping = field(default_factory=FrozenMap)
    quality: tuple[str, ...] = ()

    def __post_init__(self):
        array = np.asarray(self.samples)
        if array.ndim != 2 or min(array.shape) < 1:
            raise ValidationError('信号形状必须为非空的 (channels, frames)。')
        if not math.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValidationError('采样率必须为正数。')
        if len(self.channels) != array.shape[0] or not all(isinstance(c, ChannelInfo) for c in self.channels):
            raise ValidationError('通道描述与数据形状不一致。')
        if not np.isfinite(array).all():
            raise ValidationError('信号包含 NaN 或 Infinity，不能进行声学计算。')
        try:
            declared_origin = self.metadata.get('time_origin_seconds', 0.)
            origin = 0. if declared_origin is None else float(declared_origin)
        except (TypeError, ValueError):
            raise ValidationError('源时间起点必须为有限秒数。') from None
        if not math.isfinite(origin):
            raise ValidationError('源时间起点必须为有限秒数。')
        object.__setattr__(self, 'samples', readonly_array(array.astype(float, copy=False)))
        object.__setattr__(self, 'channels', tuple(self.channels))
        object.__setattr__(self, 'path', Path(self.path) if self.path else None)
        object.__setattr__(self, 'metadata', FrozenMap(self.metadata))
        object.__setattr__(self, 'quality', tuple(self.quality))

    @property
    def frames(self):
        return self.samples.shape[1]

    @property
    def channel_count(self):
        return self.samples.shape[0]

    @property
    def duration(self):
        return self.frames / self.sample_rate

    @property
    def time_origin(self):
        value=self.metadata.get('time_origin_seconds', 0.)
        return 0. if value is None else float(value)

    def __reduce__(self):
        return type(self), (self.samples, self.sample_rate, self.channels, self.path,
                            self.source_hash, self.metadata, self.quality)


@dataclass(frozen=True)
class PhysicalSignal(SignalData):
    def __post_init__(self):
        super().__post_init__()
        if any(c.unit != 'Pa' for c in self.channels):
            raise ValidationError('声学物理信号仅接受 Pa；其他物理量请使用对应频谱。')


@dataclass(frozen=True)
class CalibrationProfile:
    name: str
    source: str
    coefficient: float
    input_unit: str
    profile_id: str = field(default_factory=lambda: str(uuid4()))
    channel: int | None = None
    channel_name: str | None = None
    applicable_hashes: tuple[str, ...] = ()
    date: str = ''
    mode: str = 'coefficient'
    details: Mapping = field(default_factory=FrozenMap)

    def __post_init__(self):
        if not self.name or not self.source or self.input_unit not in {'FS', 'V', 'Pa'}:
            raise ValidationError('校准须有名称、来源和明确输入单位。')
        if not math.isfinite(self.coefficient) or self.coefficient <= 0:
            raise ValidationError('校准换算系数必须为有限正数。')
        if self.channel is not None and self.channel < 0:
            raise ValidationError('校准通道无效。')
        object.__setattr__(self, 'applicable_hashes', tuple(self.applicable_hashes))
        object.__setattr__(self, 'details', FrozenMap(self.details))


@dataclass(frozen=True)
class ImportMapping:
    sample_rate: float | None = None
    unit: str = 'unknown'
    channel_units: tuple[str, ...] = ()
    dataset: str | None = None
    channels: tuple[str, ...] = ()
    time_column: str | int | None = None
    data_columns: tuple[str | int, ...] = ()
    channel_axis: int | None = None
    audio_stream_index: int | None = None
    delimiter: str | None = None
    time_unit: str | None = None

    def __post_init__(self):
        if self.unit not in UNITS or any(unit not in UNITS for unit in self.channel_units):
            raise ValidationError('导入映射单位无效。')
        if self.sample_rate is not None and (not math.isfinite(self.sample_rate) or self.sample_rate <= 0):
            raise ValidationError('映射采样率必须为有限正数。')
        if self.channel_axis not in (None, 0, 1):
            raise ValidationError('通道轴只能为 0 或 1。')
        if self.time_unit not in (None, 's', 'ms', 'us'):
            raise ValidationError('时间列单位须明确为 s、ms 或 us。')
        for name in ('channel_units', 'channels', 'data_columns'):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class AnalysisSettings:
    channel: int | None = None
    start: int = 0
    end: int | None = None
    event_label: str = 'full'
    remove_dc: bool = False
    fft_size: int = 4096
    stft_size: int = 1024
    stft_hop: int = 512
    window: str = 'hann'
    field_type: str = 'free'
    spl_time_weighting: str = 'Fast'
    history_step_s: float = 0.01
    compute_loudness: bool = True
    stationary_loudness: bool = False

    def __post_init__(self):
        if self.field_type not in {'free', 'diffuse'} or self.spl_time_weighting not in {'Fast', 'Slow'}:
            raise ValidationError('声场或 SPL 时间计权无效。')
        if self.fft_size < 2 or self.stft_size < 2 or not 0 < self.stft_hop <= self.stft_size:
            raise ValidationError('FFT/STFT 窗长和步长无效。')
        if self.history_step_s <= 0 or not math.isfinite(self.history_step_s):
            raise ValidationError('历程输出间隔必须为正数。')

    def resolve(self, signal):
        end = signal.frames if self.end is None else self.end
        if self.channel is None or isinstance(self.channel, bool) or not isinstance(self.channel, int):
            raise ValidationError('必须明确选择一个分析通道。')
        if not 0 <= self.channel < signal.channel_count:
            raise ValidationError('所选通道不存在。')
        if not isinstance(self.start, int) or not isinstance(end, int) or not 0 <= self.start < end <= signal.frames:
            raise ValidationError('事件区间须为原录音样本索引 [start, end)，且不得越界。')
        return self.channel, self.start, end


@dataclass(frozen=True)
class MeasurementContext:
    specimen: str = 'unknown'
    version: str = 'unknown'
    test_level: str = 'unknown'
    mechanism: str = 'unknown'
    supply: str = 'unknown'
    supply_source: str = 'unknown'
    load: str = 'unknown'
    load_layout: str = 'unknown'
    fixture: str = 'unknown'
    fastening: str = 'unknown'
    environment: str = 'unknown'
    temperature: str = 'unknown'
    mic_position: str = 'unknown'
    mic_distance: str = 'unknown'
    mic_direction: str = 'unknown'
    background: str = 'unknown'
    actual_movement: str = 'unknown'
    motor_rotation: str = 'unknown'
    rotation_view: str = 'unknown'
    stroke: str = 'unknown'
    cycle: str = 'unknown'
    repeat: str = 'unknown'
    notes: str = ''


@dataclass(frozen=True)
class MetricResult:
    value: float | None
    unit: str
    status: str = 'ok'
    method: str = ''
    message: str = ''


@dataclass(frozen=True)
class AnalysisSummary:
    metrics: Mapping[str, MetricResult] = field(default_factory=FrozenMap)
    warnings: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'metrics', FrozenMap(self.metrics))
        object.__setattr__(self, 'warnings', tuple(self.warnings))


@dataclass(frozen=True)
class AnalysisDetail:
    arrays: Mapping[str, np.ndarray] = field(default_factory=FrozenMap)
    units: Mapping[str, str] = field(default_factory=FrozenMap)

    def __post_init__(self):
        object.__setattr__(self, 'arrays', FrozenMap(self.arrays))
        object.__setattr__(self, 'units', FrozenMap(self.units))

    @classmethod
    def _from_owned_arrays(cls, arrays, units):
        """Consume pipeline/cache-owned arrays; callers must relinquish writes.

        Public construction still copies caller buffers. Only internal producers
        use this move so a ten-minute detail does not briefly double its memory.
        """
        result = cls.__new__(cls)
        object.__setattr__(result, 'arrays', _restore_owned_map(arrays))
        object.__setattr__(result, 'units', FrozenMap(units))
        return result


@dataclass(frozen=True)
class AnalysisResult:
    result_id: str
    input_hash: str
    path: Path | None
    settings: AnalysisSettings
    context: MeasurementContext
    calibration: CalibrationProfile | None
    summary: AnalysisSummary
    detail: AnalysisDetail | None = None
    provenance: Mapping = field(default_factory=FrozenMap)

    def __post_init__(self):
        object.__setattr__(self, 'provenance', FrozenMap(self.provenance))


@dataclass(frozen=True)
class BatchItemSpec:
    path: Path
    settings: AnalysisSettings
    profile: CalibrationProfile | None = None
    context: MeasurementContext = field(default_factory=MeasurementContext)
    mapping: ImportMapping | None = None
    input_hash: str = ''
    item_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        object.__setattr__(self, 'path', Path(self.path))


@dataclass(frozen=True)
class BatchItemResult:
    item_id: str
    path: Path
    status: str
    result: AnalysisResult | None = None
    error: str = ''
    specification: BatchItemSpec | None = None


@dataclass(frozen=True)
class BatchResult:
    items: tuple[BatchItemResult, ...]


@dataclass(frozen=True)
class ComparisonResult:
    left: AnalysisResult
    right: AnalysisResult
    differences: Mapping[str, MetricResult]
    condition_differences: tuple[str, ...]
    direction: str = 'right_minus_left'

    def __post_init__(self):
        object.__setattr__(self, 'differences', FrozenMap(self.differences))


@dataclass(frozen=True)
class ResultSnapshot:
    result: AnalysisResult
    detail_cache: str = ''
    detail_hash: str = ''


@dataclass(frozen=True)
class MeasurementProject:
    name: str
    items: tuple[BatchItemSpec, ...] = ()
    results: tuple[ResultSnapshot, ...] = ()
    reports: tuple[str, ...] = ()
    project_id: str = field(default_factory=lambda: str(uuid4()))
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or not self.name:
            raise ValidationError('项目版本或名称无效。')
        for name in ('items', 'results', 'reports'):
            object.__setattr__(self, name, tuple(getattr(self, name)))


class CancellationToken:
    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    @property
    def cancelled(self):
        return self._event.is_set()

    def check(self):
        if self.cancelled:
            raise CancelledError('任务已取消。')


ProgressCallback = Callable[[int, int, str], None]
