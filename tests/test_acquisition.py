from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pytest

from autoacoustics.acquisition import storage

from autoacoustics.acquisition.device import (AcquisitionConfig, AcquisitionError, AcquisitionEvent,
                                            AcquisitionStatus, ChannelConfig, SampleBlock)
from autoacoustics.acquisition.engine import RecordingEngine
from autoacoustics.acquisition.replay import ReplayBackend
from autoacoustics.acquisition.storage import SessionWriter, open_acquisition_session, recover_session
from autoacoustics.calibration import apply_calibration
from autoacoustics.model import CalibrationProfile


def config(*, frames=128, block=16, **kwargs):
    return AcquisitionConfig(channels=(ChannelConfig("Replay/ai0", "麦克风电压", "V"),
                                       ChannelConfig("Replay/ai1", "已是声压", "Pa")),
                             requested_sample_rate=48000, target_frames=frames, block_frames=block, **kwargs)


def pattern(first, count, channel_count):
    values = (np.arange(first, first + count, dtype=np.int64) % 4096) / 4096
    return np.vstack([values + index for index in range(channel_count)])


def test_replay_recording_reopens_through_product_import_with_native_rate_units_hash_and_events(tmp_path):
    conf = config()
    backend = ReplayBackend(actual_sample_rate=51200, generator=pattern)
    engine = RecordingEngine(backend, conf, tmp_path / "session")
    assert engine.snapshot().status == AcquisitionStatus.IDLE
    configured = engine.configure()
    assert configured.actual_sample_rate == 51200
    engine.mark_event("开始", direction="CW", note="观察基准已登记")
    engine.start()
    result = engine.wait(timeout=5)
    assert result.status == AcquisitionStatus.COMPLETE and result.is_simulated
    assert result.frames_expected == result.frames_read == result.frames_written == 128
    assert backend.closed
    signal = open_acquisition_session(result.manifest_path)
    np.testing.assert_array_equal(signal.samples, pattern(0, 128, 2))
    assert signal.sample_rate == 51200 and [c.unit for c in signal.channels] == ["V", "Pa"]
    assert signal.metadata["acquisition_session"]["events"][0]["direction"] == "CW"
    assert signal.source_hash == hashlib.sha256(result.manifest_path.read_bytes()).hexdigest()
    assert signal.metadata["acquisition_data_sha256"] == hashlib.sha256(result.data_path.read_bytes()).hexdigest()
    profile = CalibrationProfile("test", "test reference", 20, "V", channel=0)
    np.testing.assert_array_equal(apply_calibration(signal, profile).samples, signal.samples[:1] * 20)
    np.testing.assert_array_equal(apply_calibration(signal, profile, channel=1).samples, signal.samples[1:])
    assert engine.stop().frames_written == 128
    with pytest.raises(AcquisitionError):
        engine.start()


def test_stop_drains_backend_tail_and_all_queued_blocks_before_complete(tmp_path):
    backend = ReplayBackend(generator=pattern, pace_seconds=0.005, stop_tail_frames=7)
    engine = RecordingEngine(backend, config(frames=None, block=16), tmp_path / "stop")
    engine.configure()
    engine.start()
    while engine.snapshot().frames_read < 32:
        time.sleep(0.002)
    engine.mark_event("停止", direction="CCW")
    result = engine.stop(timeout=5)
    assert result.status == AcquisitionStatus.COMPLETE
    assert result.frames_expected == result.frames_read == result.frames_written
    assert result.frames_written % 16 == 7
    np.testing.assert_array_equal(open_acquisition_session(result.manifest_path).samples,
                                  pattern(0, result.frames_written, 2))
    assert len(engine.preview().samples[0]) <= engine.config.preview_frames


@pytest.mark.parametrize("fault,flag", [("gap", "sample_gap"), ("reorder", "sample_order"),
                                     ("disconnect", "device_disconnected"), ("overflow", "device_overflow"),
                                     ("stop_failure", "stop_failed")])
def test_integrity_faults_are_failed_and_never_silently_padded(tmp_path, fault, flag):
    engine = RecordingEngine(ReplayBackend(generator=pattern, fault=fault, fault_at_block=2),
                             config(), tmp_path / fault)
    engine.configure()
    engine.start()
    result = engine.wait(5)
    assert result.status == AcquisitionStatus.FAILED and flag in result.flags
    with pytest.raises(AcquisitionError, match="完整"):
        open_acquisition_session(result.manifest_path)
    if result.frames_written:
        recovered = recover_session(result.manifest_path)
        assert recovered.status == AcquisitionStatus.FAILED
        signal = open_acquisition_session(result.manifest_path, allow_partial=True)
        assert "incomplete_recording" in signal.quality
        np.testing.assert_array_equal(signal.samples, pattern(0, result.frames_written, 2))


def test_bounded_queue_failure_is_visible_and_retains_only_committed_prefix(tmp_path):
    class SlowWriter(SessionWriter):
        def append(self, block):
            time.sleep(0.025)
            return super().append(block)
    engine = RecordingEngine(ReplayBackend(generator=pattern),
                             config(frames=4096, block=8, queue_blocks=1, queue_timeout=0.001),
                             tmp_path / "backpressure", writer_factory=SlowWriter)
    engine.configure()
    engine.start()
    result = engine.wait(5)
    assert result.status == AcquisitionStatus.FAILED and "application_backpressure" in result.flags
    assert engine.maximum_queue_depth <= 1
    assert 0 < result.frames_written < result.frames_expected


def test_disk_write_fault_keeps_journal_prefix_and_recovers_without_complete_marker(tmp_path):
    class BrokenWriter(SessionWriter):
        def append(self, block):
            if self.frames_written >= 16:
                raise OSError("Injected disk full")
            return super().append(block)
    engine = RecordingEngine(ReplayBackend(generator=pattern), config(), tmp_path / "disk",
                             writer_factory=BrokenWriter)
    engine.configure()
    engine.start()
    result = engine.wait(5)
    assert result.status == AcquisitionStatus.FAILED and "disk_write_failed" in result.flags
    assert result.frames_written == 16
    recover_session(result.manifest_path)
    signal = open_acquisition_session(result.manifest_path, allow_partial=True)
    np.testing.assert_array_equal(signal.samples, pattern(0, 16, 2))


def test_crash_recovery_ignores_uncommitted_tail_and_rejects_corrupt_committed_chunks(tmp_path):
    backend = ReplayBackend(generator=pattern)
    conf = config()
    prepared = backend.configure(conf)
    writer = SessionWriter(tmp_path / "crash", conf, prepared)
    writer.append(SampleBlock(pattern(0, 16, 2), 0, 0))
    writer.abort()
    with writer.raw_path.open("ab") as output:
        output.write(b"uncommitted tail")
    recovered = recover_session(writer.manifest_path)
    assert recovered.status == AcquisitionStatus.FAILED and recovered.frames_written == 16
    np.testing.assert_array_equal(open_acquisition_session(writer.manifest_path, allow_partial=True).samples,
                                  pattern(0, 16, 2))
    with writer.raw_path.open("r+b") as output:
        output.write(b"broken!!")
    with pytest.raises(AcquisitionError, match="校验"):
        recover_session(writer.manifest_path)


def test_complete_session_refuses_replaced_file_or_path_traversal(tmp_path):
    engine = RecordingEngine(ReplayBackend(generator=pattern), config(), tmp_path / "tamper")
    engine.configure()
    engine.start()
    result = engine.wait(5)
    with result.data_path.open("ab") as output:
        output.write(b"changed")
    with pytest.raises(AcquisitionError, match="哈希"):
        open_acquisition_session(result.manifest_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["data_file"] = "../elsewhere.h5"
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AcquisitionError, match="目录"):
        open_acquisition_session(result.manifest_path)


@pytest.mark.parametrize("kwargs", [{"requested_sample_rate": 0}, {"requested_sample_rate": float("nan")},
                                    {"block_frames": 0}, {"queue_blocks": 0}, {"target_frames": -1}])
def test_invalid_configuration_is_rejected(kwargs):
    values = {"requested_sample_rate": 48000, "channels": (ChannelConfig("x", "Mic", "V"),)}
    with pytest.raises(AcquisitionError):
        AcquisitionConfig(**{**values, **kwargs})


def test_finite_capture_stopped_before_target_is_incomplete(tmp_path):
    engine = RecordingEngine(ReplayBackend(generator=pattern, pace_seconds=0.01),
                             config(frames=1024), tmp_path / "early-stop")
    engine.configure()
    engine.start()
    while engine.snapshot().frames_read < 16:
        time.sleep(0.001)
    result = engine.stop(5)
    assert result.status == AcquisitionStatus.FAILED and "stopped_before_target" in result.flags
    assert result.frames_expected == result.frames_read == result.frames_written < 1024


def test_disk_precheck_rejects_start_and_closes_backend(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    backend = ReplayBackend()
    engine = RecordingEngine(backend, config(), tmp_path / "no-space")
    engine.configure()
    with pytest.raises(AcquisitionError, match="磁盘"):
        engine.start()
    assert engine.snapshot().status == AcquisitionStatus.FAILED
    assert "disk_space_low" in engine.snapshot().flags and backend.closed


def test_finalization_failure_preserves_recovery_and_never_sets_complete(tmp_path, monkeypatch):
    original = storage._build_hdf5
    def failure(*args, **kwargs):
        raise OSError("Injected finalization failure")
    monkeypatch.setattr(storage, "_build_hdf5", failure)
    engine = RecordingEngine(ReplayBackend(generator=pattern), config(), tmp_path / "finalization")
    engine.configure()
    engine.start()
    result = engine.wait(5)
    assert result.status == AcquisitionStatus.FAILED and "disk_finalize_failed" in result.flags
    assert json.loads(result.manifest_path.read_text(encoding="utf-8"))["status"] == "failed"
    monkeypatch.setattr(storage, "_build_hdf5", original)
    recover_session(result.manifest_path)
    np.testing.assert_array_equal(open_acquisition_session(result.manifest_path, allow_partial=True).samples,
                                  pattern(0, 128, 2))


def test_partial_journal_tail_is_recovered_only_as_verified_prefix(tmp_path):
    conf = config()
    prepared = ReplayBackend().configure(conf)
    writer = SessionWriter(tmp_path / "journal-tail", conf, prepared)
    writer.append(SampleBlock(pattern(0, 16, 2), 0, 0))
    writer.abort()
    with writer.journal_path.open("ab") as output:
        output.write(b'{"frames":')
    recovered = recover_session(writer.manifest_path)
    assert recovered.frames_written == 16 and "journal_tail_incomplete" in recovered.flags
