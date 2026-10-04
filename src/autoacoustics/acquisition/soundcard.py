"""Computer microphone/sound-card input without application gain or resampling.

The PortAudio callback only copies selected raw channels into a bounded queue.
Its exceptions never reach the caller, so failures are explicitly handed to the
RecordingEngine reader. A failed capture retains its recoverable raw prefix.
"""
from __future__ import annotations

import importlib
import json
import math
import os
from pathlib import Path
import queue
import threading
import time
from uuid import uuid4

import numpy as np

from .device import (AcquisitionError, ConfiguredAcquisition, DeviceCapabilities,
                     DiscoveryResult, SampleBlock)


def _sounddevice(module=None):
    return module if module is not None else importlib.import_module("sounddevice")


def _channel_indices(indices):
    try:
        indices = tuple(indices)
    except TypeError as error:
        raise AcquisitionError("声卡输入通道须为非负整数序列。") from error
    if (not indices or any(type(index) is not int or index < 0 for index in indices)
            or len(set(indices)) != len(indices)):
        raise AcquisitionError("声卡输入通道须为非空、不重复的非负整数序列。")
    return indices


def discover_soundcards(*, sd_module=None):
    """Enumerate genuine PortAudio input devices; no simulated fallback."""
    version = None
    try:
        sd = _sounddevice(sd_module)
        version = getattr(sd, "__version__", None)
        host_apis = sd.query_hostapis()
        try:
            default_input = int(sd.default.device[0])
        except (AttributeError, IndexError, TypeError, ValueError):
            default_input = -1
        devices = []
        for index, info in enumerate(sd.query_devices()):
            inputs = int(info["max_input_channels"])
            if inputs <= 0:
                continue
            host_api = str(host_apis[int(info["hostapi"])]["name"])
            default_rate = float(info["default_samplerate"])
            if not math.isfinite(default_rate) or default_rate <= 0:
                raise AcquisitionError(f"声卡 {info['name']} 未提供有效默认采样率。")
            identifier = f"soundcard:{index}"
            devices.append(DeviceCapabilities(identifier, str(info["name"]), "SOUND_CARD",
                tuple(f"{identifier}/input{channel}" for channel in range(inputs)),
                supports_realtime_samples=True, supports_controlled_recording=True,
                is_simulated=sd_module is not None,
                metadata={"device_index": index, "host_api": host_api,
                    "host_api_index": int(info["hostapi"]), "default_samplerate": default_rate,
                    "max_input_channels": inputs, "is_default_input": index == default_input,
                    "source_unit": "FS", "operating_system_processing": "unconfirmed",
                    "application_gain_applied": False, "application_resampling_applied": False}))
        return DiscoveryResult(True, tuple(devices),
            "" if devices else "PortAudio 可用，但未发现麦克风或声卡输入；请核对设备连接和系统录音权限。",
            python_package_version=version)
    except Exception as error:
        return DiscoveryResult(False, reason=f"声卡输入不可用：{error}。请核对录音设备和系统麦克风权限。",
                               python_package_version=version)


def check_soundcard_input_settings(device_index, sample_rate, channel_indices=(0,), *, sd_module=None):
    """Ask the driver about this exact input format; unsupported formats fail."""
    indices = _channel_indices(channel_indices)
    if type(device_index) is not int or device_index < 0:
        raise AcquisitionError("声卡设备序号无效。")
    if isinstance(sample_rate, bool) or not math.isfinite(sample_rate) or sample_rate <= 0:
        raise AcquisitionError("声卡采样率须为有限正数。")
    try:
        _sounddevice(sd_module).check_input_settings(device=device_index, channels=max(indices)+1,
            samplerate=sample_rate, dtype="float32")
    except Exception as error:
        raise AcquisitionError(f"声卡不支持所选输入通道或采样率：{error}",
                               flag="unsupported_input_settings") from error


class SoundCardBackend:
    """Single-use StreamBackend. Counts mean callback-delivered input frames.

    PortAudio has no device hardware frame counter. Overflow invalidates the
    capture even when delivered/read/written counts match. OS/microphone AGC,
    enhancement, ADC headroom and clock accuracy require external confirmation.
    """
    def __init__(self, device_index: int, channel_indices: tuple[int, ...] = (0,), *, sd_module=None):
        if type(device_index) is not int or device_index < 0:
            raise AcquisitionError("声卡设备序号无效。")
        self.device_index, self.channel_indices = device_index, _channel_indices(channel_indices)
        self.sd = _sounddevice(sd_module)
        self._injected = sd_module is not None
        self.stream = self.configured = None
        self.started = self.closed = self.stopped = False
        self._lock = threading.RLock()
        self._stop_requested, self._finished = threading.Event(), threading.Event()
        self._received_frames = 0
        self._error, self._error_reported = None, False

    def configure(self, config):
        if self.stream is not None or self.closed:
            raise AcquisitionError("声卡会话不能重新配置；请创建新的录制会话。")
        discovery = discover_soundcards(sd_module=self.sd if self._injected else None)
        if not discovery.available:
            raise AcquisitionError(discovery.reason, flag="driver_unavailable")
        devices = [device for device in discovery.devices
                   if device.metadata["device_index"] == self.device_index]
        if len(devices) != 1:
            raise AcquisitionError("所选声卡输入已不可用，请刷新录音设备。", flag="device_disconnected")
        device = devices[0]
        paths = tuple(f"soundcard:{self.device_index}/input{index}" for index in self.channel_indices)
        if (tuple(channel.physical_channel for channel in config.channels) != paths
                or any(path not in device.physical_channels for path in paths)):
            raise AcquisitionError("所选声卡输入通道或通道顺序与录制配置不一致。")
        for channel in config.channels:
            if channel.unit != "FS" or channel.measurement_type != "physical":
                raise AcquisitionError("声卡只能保存原始 FS 数字幅值；声压换算请在录制后校准。")
            if (channel.coupling is not None or channel.excitation_source != "none"
                    or channel.excitation_current_a is not None or channel.sensitivity_mv_pa is not None
                    or channel.max_sound_pressure_db is not None):
                raise AcquisitionError("声卡输入不能声明未由驱动确认的耦合、激励或麦克风物理缩放。")
            if channel.input_min is not None and (channel.input_min != -1 or channel.input_max != 1):
                raise AcquisitionError("声卡 FS 标称输入范围应为 -1 到 1，不能改变原始幅值。")
        check_soundcard_input_settings(self.device_index, config.requested_sample_rate,
                                      self.channel_indices, sd_module=self.sd)
        self.config = config
        self._queue = queue.Queue(maxsize=config.queue_blocks)
        try:
            self.stream = self.sd.InputStream(device=self.device_index,
                samplerate=config.requested_sample_rate, channels=max(self.channel_indices)+1,
                blocksize=config.block_frames, dtype="float32", callback=self._callback,
                finished_callback=self._finished.set)
            self.rate = float(self.stream.samplerate)
            self.configured = ConfiguredAcquisition(device, config.requested_sample_rate,
                self.rate, config.channels, {"clock": "PortAudio reported input stream sample rate",
                    "python_package_version": discovery.python_package_version,
                    "portaudio_input_channels": max(self.channel_indices)+1,
                    "selected_input_channels": self.channel_indices,
                    "sample_dtype": "float32", "source_unit": "FS", "nominal_full_scale": 1.0,
                    "frame_count_scope": "PortAudio callback-delivered frames; no hardware counter",
                    "overflow_invalidates_recording": True,
                    "application_gain_applied": False, "application_resampling_applied": False,
                    "application_calibration_applied": False,
                    "operating_system_processing": "unconfirmed",
                    "analog_overload_detection": "unavailable",
                    "main_storage": "application_chunk_journal_then_hdf5"})
            return self.configured
        except Exception as error:
            self.close()
            if isinstance(error, AcquisitionError):
                raise
            raise AcquisitionError(f"配置声卡录音失败：{error}", flag="configuration_failed") from error

    def _set_error(self, error):
        with self._lock:
            if self._error is None:
                self._error = error

    def _callback(self, indata, frames, time_info, status):
        if frames == 0:
            if self._stop_requested.is_set():
                raise self.sd.CallbackStop
            return
        try:
            if (type(frames) is not int or frames < 0 or indata.ndim != 2
                    or indata.shape != (frames, max(self.channel_indices)+1)):
                raise AcquisitionError("声卡回调样本形状与输入配置不一致。", flag="invalid_sample_block")
            with self._lock:
                remaining = (self.config.target_frames - self._received_frames
                             if self.config.target_frames is not None else frames)
                count = min(frames, remaining)
                first = self._received_frames
                self._received_frames += count
            if count <= 0:
                raise self.sd.CallbackStop
            flags, message = [], None
            if getattr(status, "input_overflow", False):
                flags.append("device_overflow")
                message = "PortAudio 输入溢出，驱动可能已经丢样；此录制不能作为完整测量。"
            if getattr(status, "input_underflow", False):
                flags.append("device_underflow")
                message = "PortAudio 输入欠载，原始输入样本不可确认。"
            values = indata[:count, self.channel_indices].T
            for offset in range(0, count, self.config.block_frames):
                samples = values[:, offset:offset+self.config.block_frames]
                block = SampleBlock(samples, first+offset, (first+offset)/self.rate,
                                    tuple(flags), error_message=message)
                try:
                    self._queue.put_nowait(block)
                except queue.Full as error:
                    raise AcquisitionError("声卡回调队列已满；原始样本未能全部提交，录制已停止。",
                                           flag="application_backpressure") from error
            if (self._stop_requested.is_set() or (self.config.target_frames is not None
                    and self._received_frames >= self.config.target_frames)):
                raise self.sd.CallbackStop
        except self.sd.CallbackStop:
            raise
        except Exception as error:
            self._set_error(error if isinstance(error, AcquisitionError) else
                AcquisitionError(f"声卡回调失败：{error}", flag="device_read_failed"))
            raise self.sd.CallbackAbort from error

    def start(self):
        if self.configured is None or self.closed or self.started:
            raise AcquisitionError("声卡录制尚未配置或会话已启动/关闭。")
        self.started = True
        try:
            self.stream.start()
        except Exception as error:
            self.close()
            raise AcquisitionError(f"无法开始声卡录音：{error}；请核对系统麦克风权限。",
                                   flag="start_failed") from error

    def _raise_pending_error(self):
        with self._lock:
            if self._error is not None and not self._error_reported:
                self._error_reported = True
                raise self._error

    def read_block(self, timeout):
        if not self.started or self.closed:
            raise AcquisitionError("声卡录制未开始或已关闭。")
        if not math.isfinite(timeout) or timeout <= 0:
            raise AcquisitionError("声卡读取超时须为有限正数。")
        deadline = time.monotonic() + max(timeout, 2*self.config.block_frames/self.rate, 1.0)
        while True:
            try:
                return self._queue.get_nowait()
            except queue.Empty:
                pass
            self._raise_pending_error()
            if self._stop_requested.is_set() or self.stopped:
                return None
            if self._finished.is_set() or not self.stream.active:
                if self.config.target_frames is not None and self._received_frames >= self.config.target_frames:
                    return None
                raise AcquisitionError("声卡输入流意外停止，可能设备断开或系统终止了录音。",
                                       flag="device_disconnected")
            if time.monotonic() >= deadline:
                raise AcquisitionError("声卡输入流在读取期限内未提供样本。", flag="device_read_timeout")
            # The event wakes this wait when the UI requests stop. It never
            # performs a blocking PortAudio stop on the UI thread.
            self._stop_requested.wait(min(timeout, .02))

    def request_stop(self):
        self._stop_requested.set()

    def stop_acquisition(self):
        self._stop_requested.set()
        if self.stream is None:
            raise AcquisitionError("声卡输入未配置。", flag="stop_failed")
        if not self.stopped:
            try:
                self.stream.stop()
            except Exception as error:
                raise AcquisitionError(f"停止声卡录音失败：{error}", flag="stop_failed") from error
            finally:
                self.stopped = True
        with self._lock:
            return self._received_frames

    def drain_blocks(self):
        if not self.stopped:
            raise AcquisitionError("声卡须停止后才能排空剩余原始样本。")
        while True:
            try:
                yield self._queue.get_nowait()
            except queue.Empty:
                break
        self._raise_pending_error()

    def close(self):
        self._stop_requested.set()
        if self.closed:
            return
        self.closed = True
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception as error:
                raise AcquisitionError(f"关闭声卡输入失败：{error}", flag="device_close_failed") from error


def export_soundcard_wav(manifest_path) -> Path:
    """Save an exact float32 copy of a complete FS sound-card session in WAV.

    HDF5 remains the authoritative record including source/quality/channel
    information. WAV is streamed in bounded chunks and bound to that manifest
    by SHA-256. A failed/partial session cannot become an apparently clean WAV.
    """
    import h5py
    import soundfile as sf
    from autoacoustics.importers.common import file_hash
    from .storage import read_manifest, safe_session_file, write_manifest

    path, manifest = read_manifest(manifest_path)
    if (manifest["status"] != "complete" or manifest["backend"] != "SOUND_CARD"
            or any(channel["unit"] != "FS" for channel in manifest["channels"])):
        raise AcquisitionError("仅能导出通过完整性检查的原始 FS 声卡录制。")
    rate = manifest["actual_sample_rate"]
    if rate != round(rate):
        raise AcquisitionError("WAV 只能声明整数采样率；原始非整数时钟请使用 HDF5。")
    data_path = safe_session_file(path, manifest["data_file"])
    if file_hash(data_path) != manifest["data_sha256"]:
        raise AcquisitionError("声卡原始 HDF5 哈希不一致，拒绝导出 WAV。")
    destination = path.parent / "recording.wav"
    transaction_path = path.parent / ".wav-export.json"
    frames, channels = manifest["frames_written"], len(manifest["channels"])
    transaction = None
    if transaction_path.exists():
        try:
            if transaction_path.stat().st_size > 8192:
                raise ValueError("transaction too large")
            transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
            expected = {"kind": "autoacoustics_soundcard_wav_export", "schema_version": 1,
                "session_id": manifest["session_id"], "data_sha256": manifest["data_sha256"],
                "wav_file": destination.name, "frames": frames, "channels": channels,
                "sample_rate": rate, "wav_subtype": "FLOAT", "wav_calibration_applied": False}
            if (not isinstance(transaction, dict)
                    or any(transaction.get(key) != value for key, value in expected.items())
                    or transaction.get("wav_format") not in {"WAV", "RF64"}
                    or not isinstance(transaction.get("wav_sha256"), str)
                    or len(transaction["wav_sha256"]) != 64
                    or not set(transaction["wav_sha256"]) <= set("0123456789abcdef")
                    or not isinstance(transaction.get("temporary_file"), str)
                    or not transaction["temporary_file"].startswith(".recording.")
                    or not transaction["temporary_file"].endswith(".partial")):
                raise ValueError("transaction does not belong to this recording")
            safe_session_file(path, transaction["temporary_file"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise AcquisitionError("WAV 导出事务与当前录制不一致；保留现有文件，不能覆盖。") from error

    def bind_wav(digest, format_name):
        manifest.update(wav_file=destination.name, wav_sha256=digest, wav_subtype="FLOAT",
                        wav_format=format_name, wav_calibration_applied=False)
        write_manifest(path, manifest)

    def commit_wav(staged):
        # Windows rename fails when the destination exists. Other platforms
        # use an exclusive hard-link so an unexpected file is never replaced.
        if os.name == "nt":
            os.rename(staged, destination)
        else:
            os.link(staged, destination)
            staged.unlink()

    def verify_recoverable_wav(candidate):
        if file_hash(candidate) != transaction["wav_sha256"]:
            raise AcquisitionError("事务中的 WAV 已变更，不能恢复绑定或覆盖。")
        with h5py.File(data_path, "r") as source, sf.SoundFile(candidate, "r") as wav:
            samples = source["samples"]
            if (samples.shape != (channels, frames) or samples.attrs["sample_rate"] != rate
                    or wav.channels != channels or len(wav) != frames or wav.samplerate != rate
                    or wav.subtype != "FLOAT" or wav.format != transaction["wav_format"]):
                raise AcquisitionError("待恢复 WAV 的格式或计数与原始 HDF 会话不一致。")
            block_frames = manifest["configuration"]["block_frames"]
            for start in range(0, frames, block_frames):
                count = min(block_frames, frames-start)
                original = samples[:, start:start+count].T
                copied = wav.read(count, dtype="float64", always_2d=True)
                if not np.isfinite(original).all() or not np.array_equal(copied, original):
                    raise AcquisitionError("待恢复 WAV 的样本与原始 HDF 数据不一致。")
        if (file_hash(candidate) != transaction["wav_sha256"]
                or file_hash(data_path) != manifest["data_sha256"]):
            raise AcquisitionError("WAV 或原始 HDF 在恢复核对期间发生变化。")

    if destination.exists():
        if manifest.get("wav_file") == destination.name and file_hash(destination) == manifest.get("wav_sha256"):
            if transaction and transaction["wav_sha256"] == manifest["wav_sha256"]:
                transaction_path.unlink()
            return destination
        if not transaction or manifest.get("wav_file") or manifest.get("wav_sha256"):
            raise AcquisitionError("录制目录已有未绑定或内容已变更的 recording.wav，不能覆盖。")
        try:
            verify_recoverable_wav(destination)
            bind_wav(transaction["wav_sha256"], transaction["wav_format"])
            transaction_path.unlink()
            return destination
        except AcquisitionError:
            raise
        except Exception as error:
            raise AcquisitionError(f"恢复 WAV 导出事务失败：{error}", flag="wav_export_failed") from error
    if transaction:
        staged = safe_session_file(path, transaction["temporary_file"])
        if staged.exists():
            try:
                verify_recoverable_wav(staged)
                commit_wav(staged)
                bind_wav(transaction["wav_sha256"], transaction["wav_format"])
                transaction_path.unlink()
                return destination
            except AcquisitionError:
                raise
            except Exception as error:
                raise AcquisitionError(f"恢复 WAV 导出事务失败：{error}", flag="wav_export_failed") from error
    temporary = path.parent / f".recording.{uuid4().hex}.partial"
    try:
        with h5py.File(data_path, "r") as source:
            samples = source["samples"]
            if samples.shape != (channels, frames) or samples.attrs["sample_rate"] != rate:
                raise AcquisitionError("声卡 HDF5 帧数、通道数或采样率与会话清单不一致。")
            format_name = "RF64" if frames*channels*4 + 4096 >= 2**32 else "WAV"
            with sf.SoundFile(temporary, mode="x", samplerate=int(rate), channels=channels,
                              format=format_name, subtype="FLOAT") as output:
                block_frames = manifest["configuration"]["block_frames"]
                for start in range(0, frames, block_frames):
                    values = samples[:, start:start+block_frames].T
                    if not np.isfinite(values).all() or not np.array_equal(values, values.astype(np.float32)):
                        raise AcquisitionError("原始样本不符合有限 float32 声卡数据，拒绝改变精度。")
                    output.write(values)
                output.flush()
        info = sf.info(temporary)
        if info.frames != frames or info.channels != channels or info.samplerate != rate or info.subtype != "FLOAT":
            raise AcquisitionError("导出的 WAV 格式或样本计数与原始记录不一致。")
        with temporary.open("r+b") as wav:
            os.fsync(wav.fileno())
        digest = file_hash(temporary)
        transaction = {"kind": "autoacoustics_soundcard_wav_export", "schema_version": 1,
            "session_id": manifest["session_id"], "data_sha256": manifest["data_sha256"],
            "wav_file": destination.name, "wav_sha256": digest, "wav_subtype": "FLOAT",
            "wav_format": format_name, "wav_calibration_applied": False,
            "frames": frames, "channels": channels, "sample_rate": rate,
            "temporary_file": temporary.name}
        write_manifest(transaction_path, transaction)
        commit_wav(temporary)
        bind_wav(digest, format_name)
        transaction_path.unlink()
        return destination
    except AcquisitionError:
        raise
    except Exception as error:
        raise AcquisitionError(f"导出声卡 WAV 失败：{error}", flag="wav_export_failed") from error
    finally:
        temporary.unlink(missing_ok=True)
