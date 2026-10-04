"""Separate bounded acquisition and writer threads; preview never drops raw data."""
from __future__ import annotations

from pathlib import Path
import queue
import threading
from uuid import uuid4

import numpy as np

from .device import AcquisitionError, AcquisitionEvent, AcquisitionStatus, PreviewData, RecordingSession
from .storage import SessionWriter, utc_now


class RecordingEngine:
    def __init__(self, backend, config, directory, *, writer_factory=SessionWriter):
        self.backend, self.config, self.directory = backend, config, Path(directory).resolve()
        self.writer_factory = writer_factory
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._producer_done = threading.Event()
        self._finished = threading.Event()
        self._queue = queue.Queue(maxsize=config.queue_blocks)
        self._status = AcquisitionStatus.IDLE
        self._session_id = uuid4().hex
        self._frames_read = self._frames_written = self._next_index = 0
        self._expected = config.target_frames
        self._flags, self._errors, self._events = [], [], []
        self.maximum_queue_depth = 0
        self._preview = np.empty((len(config.channels), 0))
        self._preview_first = 0
        self.configured = self.writer = self._result = None

    def configure(self):
        with self._lock:
            if self._status == AcquisitionStatus.CONFIGURED:
                return self.configured
            if self._status != AcquisitionStatus.IDLE:
                raise AcquisitionError("录制会话不能在当前状态重新配置。")
            self.configured = self.backend.configure(self.config)
            if len(self.configured.channels) != len(self.config.channels):
                raise AcquisitionError("驱动回读通道数与请求不一致。")
            self._status = AcquisitionStatus.CONFIGURED
            return self.configured

    def start(self):
        with self._lock:
            if self._status == AcquisitionStatus.RECORDING:
                return self.snapshot()
            if self._status != AcquisitionStatus.CONFIGURED:
                raise AcquisitionError("开始前须配置；已终结会话请使用新目录与新会话。")
            try:
                self.writer = self.writer_factory(self.directory, self.config, self.configured)
                self._session_id = self.writer.manifest["session_id"]
                self.backend.start()
            except Exception as error:
                self._record_error(error, "start_failed")
                self._status = AcquisitionStatus.FAILED
                try:
                    self.backend.close()
                finally:
                    if self.writer:
                        self._result = self.writer.finalize(complete=False, frames_expected=None,
                            frames_read=0, flags=self._flags, errors=self._errors, events=self._events)
                raise AcquisitionError(f"开始录制失败：{error}", flag="start_failed") from error
            self._status = AcquisitionStatus.RECORDING
            self._writer_thread = threading.Thread(target=self._write_loop, name="DAQ-writer", daemon=True)
            self._capture_thread = threading.Thread(target=self._capture_loop, name="DAQ-capture", daemon=True)
            self._writer_thread.start()
            self._capture_thread.start()
            return self.snapshot()

    def _record_error(self, error, default):
        with self._lock:
            flag = getattr(error, "flag", None) or default
            if flag not in self._flags:
                self._flags.append(flag)
            code = getattr(error, "code", None)
            self._errors.append(f"{error}" + (f" [code={code}]" if code is not None else ""))

    def _accept(self, block):
        with self._lock:
            self._frames_read += block.frames
            if block.samples.shape[0] != len(self.config.channels) or block.frames > self.config.block_frames:
                raise AcquisitionError("设备块通道数或长度与配置不一致。", flag="invalid_sample_block")
            if block.first_sample_index != self._next_index:
                flag = "sample_gap" if block.first_sample_index > self._next_index else "sample_order"
                raise AcquisitionError(f"样本块序号应为 {self._next_index}，实际 {block.first_sample_index}。", flag=flag)
            if block.timestamp_seconds is not None and not np.isclose(block.timestamp_seconds,
                    block.first_sample_index / self.configured.actual_sample_rate, rtol=0, atol=1e-9):
                raise AcquisitionError("设备块时间戳与样本序号不一致。", flag="invalid_timestamp")
            self._next_index += block.frames
            for flag in block.flags:
                self._record_error(AcquisitionError(block.error_message or f"设备块质量标记：{flag}",
                                                   code=block.error_code, flag=flag), flag)
        try:
            self._queue.put(block, timeout=self.config.queue_timeout)
        except queue.Full as error:
            raise AcquisitionError("落盘队列背压超时，原样本未能全部提交。", flag="application_backpressure") from error
        with self._lock:
            self.maximum_queue_depth = max(self.maximum_queue_depth, self._queue.qsize())
            limit = self.config.preview_frames
            if block.frames >= limit:
                self._preview = block.samples[:, -limit:].copy()
            else:
                self._preview = np.concatenate((self._preview, block.samples), axis=1)[:, -limit:].copy()
            self._preview_first = block.first_sample_index + block.frames - self._preview.shape[1]

    def _capture_loop(self):
        try:
            while not self._stop.is_set():
                block = self.backend.read_block(self.config.read_timeout)
                if block is None:
                    break
                self._accept(block)
        except Exception as error:
            self._record_error(error, "device_read_failed")
            self._stop.set()
        finally:
            try:
                expected = self.backend.stop_acquisition()
                if not isinstance(expected, int) or expected < 0:
                    raise AcquisitionError("停止后未取得可靠设备样本计数。", flag="unknown_device_count")
                with self._lock:
                    self._expected = expected
                for block in self.backend.drain_blocks():
                    self._accept(block)
            except Exception as error:
                self._record_error(error, "stop_failed")
            finally:
                try:
                    self.backend.close()
                except Exception as error:
                    self._record_error(error, "device_close_failed")
                with self._lock:
                    self._status = AcquisitionStatus.FINALIZING
                self._producer_done.set()

    def _write_loop(self):
        write_failed = False
        try:
            while not self._producer_done.is_set() or not self._queue.empty():
                try:
                    block = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    if not write_failed:
                        self.writer.append(block)
                        with self._lock:
                            self._frames_written = self.writer.frames_written
                except Exception as error:
                    write_failed = True
                    self._record_error(error, "disk_write_failed")
                    self._stop.set()
                finally:
                    self._queue.task_done()
            with self._lock:
                if self.config.target_frames is not None and self._expected != self.config.target_frames and not self._errors:
                    self._record_error(AcquisitionError("固定帧数录制在目标帧数之前停止。"), "stopped_before_target")
                complete = (not self._errors and self._expected is not None
                            and self._expected == self._frames_read == self._frames_written and self._frames_written > 0)
                if not complete and not self._errors:
                    self._record_error(AcquisitionError("设备已采、驱动已读与落盘样本计数不一致。"), "recording_integrity_failed")
                self._status = AcquisitionStatus.FINALIZING
                final_arguments = dict(complete=complete, frames_expected=self._expected,
                    frames_read=self._frames_read, flags=tuple(self._flags), errors=tuple(self._errors),
                    events=tuple(self._events))
            result = self.writer.finalize(**final_arguments)
            with self._lock:
                self._result, self._status = result, result.status
                self._flags, self._errors = list(result.flags), list(result.errors)
        except Exception as error:
            self._record_error(error, "disk_finalize_failed")
            with self._lock:
                self._status = AcquisitionStatus.FAILED
            self.writer.abort()
        finally:
            self._finished.set()

    def request_stop(self):
        self._stop.set()
        wake_backend = getattr(self.backend, "request_stop", None)
        if callable(wake_backend):
            wake_backend()

    def stop(self, timeout=10):
        if self._status in {AcquisitionStatus.COMPLETE, AcquisitionStatus.FAILED}:
            return self.snapshot()
        if self._status not in {AcquisitionStatus.RECORDING, AcquisitionStatus.FINALIZING}:
            raise AcquisitionError("当前没有正在进行的录制。")
        self.request_stop()
        return self.wait(timeout)

    def wait(self, timeout=10):
        if not self._finished.wait(timeout):
            raise AcquisitionError("录制仍在进行或终结，等待超时；请检查状态，不能当作已完成。")
        return self.snapshot()

    def snapshot(self):
        with self._lock:
            if self._result:
                return self._result
            return RecordingSession(self._session_id, self._status,
                self.configured.device.backend if self.configured else type(self.backend).__name__,
                bool(self.configured and self.configured.device.is_simulated), self.directory / "session.json",
                self.writer.raw_path if self.writer else None, self.config.requested_sample_rate,
                self.configured.actual_sample_rate if self.configured else None, self._expected,
                self._frames_read, self._frames_written, tuple(self._flags), tuple(self._errors))

    def preview(self):
        with self._lock:
            return PreviewData(self._preview, self.configured.actual_sample_rate if self.configured else None,
                               self._preview_first)

    def mark_event(self, kind, *, direction=None, note="", sample_index=None):
        with self._lock:
            if self._status not in {AcquisitionStatus.CONFIGURED, AcquisitionStatus.RECORDING}:
                raise AcquisitionError("事件只能在已配置或录制状态添加。")
            index = self._frames_read if sample_index is None else sample_index
            if not isinstance(index, int) or not 0 <= index <= self._frames_read or not kind:
                raise AcquisitionError("事件名称或样本序号无效。")
            event = AcquisitionEvent(kind, index, index / self.configured.actual_sample_rate, direction, note, utc_now())
            self._events.append(event)
            return event
