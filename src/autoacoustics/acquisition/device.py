from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from autoacoustics.model import AcousticError, freeze


class AcquisitionError(AcousticError):
    def __init__(self, message, *, code=None, flag=None):
        super().__init__(message)
        self.code, self.flag = code, flag


class AcquisitionStatus(str, Enum):
    IDLE = "idle"
    CONFIGURED = "configured"
    RECORDING = "recording"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class DeviceCapabilities:
    device_id: str
    name: str
    backend: str
    physical_channels: tuple[str, ...] = ()
    driver_version: str | None = None
    model: str | None = None
    supports_realtime_samples: bool = False
    supports_controlled_recording: bool = False
    is_simulated: bool = False
    metadata: object = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "physical_channels", tuple(self.physical_channels))
        object.__setattr__(self, "metadata", freeze(self.metadata))


@dataclass(frozen=True)
class DiscoveryResult:
    available: bool
    devices: tuple[DeviceCapabilities, ...] = ()
    reason: str = ""
    driver_version: str | None = None
    python_package_version: str | None = None


@dataclass(frozen=True)
class ChannelConfig:
    physical_channel: str
    name: str
    unit: str = "V"
    measurement_type: str = "voltage"
    input_min: float | None = None
    input_max: float | None = None
    coupling: str | None = None
    excitation_source: str = "none"
    excitation_current_a: float | None = None
    sensitivity_mv_pa: float | None = None
    max_sound_pressure_db: float | None = None
    sensor: object = field(default_factory=dict)
    calibration_snapshot: object = field(default_factory=dict)

    def __post_init__(self):
        if not self.physical_channel or not self.name or self.unit not in {"FS", "V", "Pa", "g", "m/s²", "unknown"}:
            raise AcquisitionError("通道路径、名称或源单位无效。")
        if self.measurement_type not in {"voltage", "microphone", "physical"}:
            raise AcquisitionError("须明确通道测量类型。")
        if (self.input_min is None) != (self.input_max is None):
            raise AcquisitionError("输入范围须同时给出上下限。")
        if self.input_min is not None and (not math.isfinite(self.input_min) or not math.isfinite(self.input_max)
                                           or self.input_min >= self.input_max):
            raise AcquisitionError("输入范围无效。")
        if self.coupling not in {None, "AC", "DC"} or self.excitation_source not in {"none", "internal", "external"}:
            raise AcquisitionError("耦合或激励来源无效。")
        for value in (self.excitation_current_a, self.sensitivity_mv_pa):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise AcquisitionError("激励电流和灵敏度须为有限正数。")
        if self.excitation_source != "none" and self.excitation_current_a is None:
            raise AcquisitionError("激励来源已选择，须明确填写电流 A。")
        if self.max_sound_pressure_db is not None and not math.isfinite(self.max_sound_pressure_db):
            raise AcquisitionError("最大声压级无效。")
        object.__setattr__(self, "sensor", freeze(self.sensor))
        object.__setattr__(self, "calibration_snapshot", freeze(self.calibration_snapshot))


@dataclass(frozen=True)
class AcquisitionConfig:
    channels: tuple[ChannelConfig, ...]
    requested_sample_rate: float
    block_frames: int = 4096
    queue_blocks: int = 8
    queue_timeout: float = 1.0
    read_timeout: float = 0.25
    preview_frames: int = 48000
    target_frames: int | None = None
    minimum_free_bytes: int = 64 * 1024 * 1024
    measurement_context: object = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "channels", tuple(self.channels))
        if not self.channels or len({c.physical_channel for c in self.channels}) != len(self.channels):
            raise AcquisitionError("须选择至少一个通道，且物理通道不能重复。")
        if len({c.name for c in self.channels}) != len(self.channels):
            raise AcquisitionError("通道显示名称不能重复。")
        if not math.isfinite(self.requested_sample_rate) or self.requested_sample_rate <= 0:
            raise AcquisitionError("请求采样率须为有限正数。")
        for name in ("block_frames", "queue_blocks", "preview_frames"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise AcquisitionError(f"{name} 须为正整数。")
        if self.target_frames is not None and (not isinstance(self.target_frames, int)
                                               or isinstance(self.target_frames, bool) or self.target_frames <= 0):
            raise AcquisitionError("固定目标帧数须为正整数。")
        if any(not math.isfinite(v) or v <= 0 for v in (self.queue_timeout, self.read_timeout)):
            raise AcquisitionError("读取及队列超时须为有限正数。")
        if not isinstance(self.minimum_free_bytes, int) or self.minimum_free_bytes < 0:
            raise AcquisitionError("磁盘保留空间无效。")
        object.__setattr__(self, "measurement_context", freeze(self.measurement_context))


@dataclass(frozen=True)
class ConfiguredAcquisition:
    device: DeviceCapabilities
    requested_sample_rate: float
    actual_sample_rate: float
    channels: tuple[ChannelConfig, ...]
    metadata: object = field(default_factory=dict)

    def __post_init__(self):
        if not math.isfinite(self.actual_sample_rate) or self.actual_sample_rate <= 0:
            raise AcquisitionError("驱动回读的实际采样率无效。")
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(self, "metadata", freeze(self.metadata))


@dataclass(frozen=True)
class SampleBlock:
    samples: np.ndarray
    first_sample_index: int
    timestamp_seconds: float | None = None
    flags: tuple[str, ...] = ()
    error_code: int | None = None
    error_message: str | None = None

    def __post_init__(self):
        value = np.array(self.samples, dtype=np.float64, copy=True)
        if value.ndim != 2 or not value.shape[0] or not value.shape[1] or not np.isfinite(value).all():
            raise AcquisitionError("设备样本块须为有限实数 channels×frames。", flag="invalid_sample_block")
        if not isinstance(self.first_sample_index, int) or self.first_sample_index < 0:
            raise AcquisitionError("样本块首帧序号无效。", flag="sample_order")
        if self.timestamp_seconds is not None and not math.isfinite(self.timestamp_seconds):
            raise AcquisitionError("样本块时间戳无效。", flag="invalid_timestamp")
        value.flags.writeable = False
        object.__setattr__(self, "samples", value)
        object.__setattr__(self, "flags", tuple(self.flags))

    @property
    def frames(self):
        return self.samples.shape[1]


@dataclass(frozen=True)
class AcquisitionEvent:
    kind: str
    sample_index: int
    time_seconds: float
    direction: str | None = None
    note: str = ""
    recorded_at_utc: str = ""


@dataclass(frozen=True)
class PreviewData:
    samples: np.ndarray
    sample_rate: float | None
    first_sample_index: int

    def __post_init__(self):
        value = np.array(self.samples, dtype=np.float64, copy=True)
        value.flags.writeable = False
        object.__setattr__(self, "samples", value)


@dataclass(frozen=True)
class RecordingSession:
    session_id: str
    status: AcquisitionStatus
    backend: str
    is_simulated: bool
    manifest_path: Path
    data_path: Path | None = None
    requested_sample_rate: float | None = None
    actual_sample_rate: float | None = None
    frames_expected: int | None = None
    frames_read: int = 0
    frames_written: int = 0
    flags: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@runtime_checkable
class StreamBackend(Protocol):
    def configure(self, config: AcquisitionConfig) -> ConfiguredAcquisition: ...
    def start(self) -> None: ...
    def read_block(self, timeout: float) -> SampleBlock | None: ...
    def stop_acquisition(self) -> int: ...
    def drain_blocks(self): ...
    def close(self) -> None: ...


@runtime_checkable
class ManagedRecorderBackend(Protocol):
    supports_realtime_samples: bool
    def discover(self) -> DiscoveryResult: ...
    def configure(self, config=None): ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def completed_files(self) -> tuple[Path, ...]: ...
