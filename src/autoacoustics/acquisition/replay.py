"""Explicitly simulated deterministic input for software and failure testing."""
import time

import numpy as np

from .device import AcquisitionError, ConfiguredAcquisition, DeviceCapabilities, DiscoveryResult, SampleBlock


class ReplayBackend:
    def __init__(self, *, actual_sample_rate=None, generator=None, pace_seconds=0,
                 stop_tail_frames=0, fault=None, fault_at_block=2):
        self.actual_sample_rate = actual_sample_rate
        self.generator = generator or self._default_pattern
        self.pace_seconds, self.stop_tail_frames = pace_seconds, stop_tail_frames
        self.fault, self.fault_at_block = fault, fault_at_block
        self.closed, self.first, self.blocks, self.stopped = False, 0, 0, False

    @staticmethod
    def _default_pattern(first, count, channels):
        raw = (np.arange(first, first + count, dtype=np.int64) % 4096) / 4096
        return np.vstack([raw + index for index in range(channels)])

    def configure(self, config):
        self.config = config
        self.rate = self.actual_sample_rate or config.requested_sample_rate
        return ConfiguredAcquisition(DeviceCapabilities("REPLAY", "确定性回放（模拟）", "REPLAY",
            tuple(c.physical_channel for c in config.channels), supports_realtime_samples=True,
            supports_controlled_recording=True, is_simulated=True, metadata={"hardware_verified": False}),
            config.requested_sample_rate, self.rate, config.channels, {"clock": "simulated_sample_index"})

    def start(self):
        if not hasattr(self, "config"):
            raise AcquisitionError("回放尚未配置。")
        self.closed = False

    def _block(self, count, index=None):
        first = self.first
        result = SampleBlock(self.generator(first, count, len(self.config.channels)),
                             first if index is None else index, first / self.rate)
        self.first += count
        self.blocks += 1
        return result

    def read_block(self, timeout):
        if self.config.target_frames is not None and self.first >= self.config.target_frames:
            return None
        if self.pace_seconds:
            time.sleep(min(timeout, self.pace_seconds))
        fault_now = self.blocks == self.fault_at_block
        if fault_now and self.fault in {"disconnect", "overflow"}:
            flag = "device_disconnected" if self.fault == "disconnect" else "device_overflow"
            raise AcquisitionError(f"Injected replay {self.fault}", code="SIMULATED", flag=flag)
        count = self.config.block_frames
        if self.config.target_frames is not None:
            count = min(count, self.config.target_frames-self.first)
        index = None
        if fault_now and self.fault == "gap":
            index = self.first + 1
        elif fault_now and self.fault == "reorder":
            index = self.first - 1
        return self._block(count, index)

    def stop_acquisition(self):
        self.stopped = True
        if self.fault == "stop_failure":
            raise AcquisitionError("Injected replay stop failure", flag="stop_failed")
        remaining = self.stop_tail_frames
        if self.config.target_frames is not None:
            remaining = min(remaining, max(0, self.config.target_frames-self.first))
        self.tail = remaining
        return self.first + remaining

    def drain_blocks(self):
        while self.tail:
            count = min(self.tail, self.config.block_frames)
            self.tail -= count
            yield self._block(count)

    def close(self):
        self.closed = True


class SimulatedManagedRecorder:
    """Contract-only fake recorder over existing test files, never HEAD control."""
    supports_realtime_samples = False

    def __init__(self, files, *, fault=None):
        from pathlib import Path
        self.files = tuple(Path(path).resolve() for path in files)
        self.fault, self.status = fault, "idle"

    def discover(self):
        return DiscoveryResult(True, (DeviceCapabilities("SIMULATED_MANAGED", "受控录制模拟器", "SIMULATED_MANAGED",
            supports_controlled_recording=True, is_simulated=True, metadata={"hardware_verified": False}),))

    def configure(self, config=None):
        if self.status not in {"idle", "configured"}:
            raise AcquisitionError("模拟会话不能在当前状态重新配置。")
        self.configuration = config
        self.status = "configured"
        return self.discover().devices[0]

    def start(self):
        if self.fault in {"disconnect", "license"}:
            self.status = "failed"
            raise AcquisitionError(f"Injected managed {self.fault}", flag=f"simulated_{self.fault}")
        if self.status == "complete":
            raise AcquisitionError("已终结的模拟会话不能重新开始。")
        if self.status not in {"configured", "recording"}:
            raise AcquisitionError("模拟受控录制须先明确配置。")
        self.status = "recording"

    def stop(self):
        if self.status == "complete":
            return
        if self.fault == "stop":
            self.status = "failed"
            raise AcquisitionError("Injected managed stop failure", flag="simulated_stop_failed")
        if self.status != "recording":
            raise AcquisitionError("模拟受控录制尚未开始。")
        if any(not path.is_file() for path in self.files):
            self.status = "failed"
            raise AcquisitionError("模拟设备文件缺失。")
        self.status = "complete"

    def completed_files(self):
        return self.files if self.status == "complete" else ()
