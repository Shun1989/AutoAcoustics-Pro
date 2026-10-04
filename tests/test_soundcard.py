"""PortAudio is injected because hardware/failure cases cannot be deterministic.

RecordingEngine, durable SessionWriter, HDF/session import and WAV decoding are
real. Independent sample fixtures catch mapping, trimming, gap and scale errors.
"""
from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from autoacoustics.acquisition.device import (AcquisitionConfig, AcquisitionError,
    AcquisitionStatus, ChannelConfig)
from autoacoustics.acquisition.engine import RecordingEngine
from autoacoustics.acquisition.storage import open_acquisition_session, recover_session
from autoacoustics.importers import import_signal


class CallbackStop(Exception):
    pass


class CallbackAbort(Exception):
    pass


class FakeStream:
    def __init__(self, api, **settings):
        self.api, self.settings = api, settings
        self.samplerate = api.actual_rate
        self.active = self.closed = False
        self.stop_event = threading.Event()
        self.callback_lock = threading.Lock()

    def start(self):
        if self.api.fault == "start":
            raise RuntimeError("microphone permission denied")
        self.active = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        first = 0
        try:
            for values in self.api.blocks:
                if self.stop_event.wait(self.api.interval):
                    break
                if self.api.fault == "disconnect" and first:
                    break
                with self.callback_lock:
                    status = SimpleNamespace(input_overflow=self.api.fault == "overflow" and not first,
                                             input_underflow=False)
                    try:
                        self.settings["callback"](values, len(values), SimpleNamespace(
                            inputBufferAdcTime=10.0+first/self.samplerate), status)
                    except (CallbackStop, CallbackAbort):
                        break
                first += len(values)
            while self.api.keep_active and not self.stop_event.wait(.005):
                pass
        finally:
            self.active = False
            self.settings["finished_callback"]()

    def stop(self):
        self.stop_event.set()
        if hasattr(self, "thread"):
            self.thread.join(2)
        self.active = False
        if self.api.fault == "stop":
            raise RuntimeError("PortAudio stop failed")

    def close(self):
        self.stop_event.set()
        if hasattr(self, "thread"):
            self.thread.join(2)
        self.active, self.closed = False, True


class FakeSoundDevice:
    """Only the external driver boundary is replaced; input arrays are literal."""
    __version__ = "fixture-driver"
    CallbackStop, CallbackAbort = CallbackStop, CallbackAbort
    default = SimpleNamespace(device=(1, 2))

    def __init__(self, *, blocks=None, actual_rate=48000, fault=None, interval=.01,
                 keep_active=False):
        self.blocks = blocks or [np.array([[.125, -.25], [.25, -.5], [.375, -.75],
                                          [.5, -.125]], dtype=np.float32),
                                np.array([[.625, -.25], [.75, -.375], [.875, -.5],
                                          [.5, -.625]], dtype=np.float32)]
        self.actual_rate, self.fault, self.interval = actual_rate, fault, interval
        self.keep_active, self.stream = keep_active, None

    def query_devices(self):
        return [dict(name="Output only", max_input_channels=0, default_samplerate=44100, hostapi=0),
                dict(name="Microphone array", max_input_channels=2, default_samplerate=48000, hostapi=1)]

    def query_hostapis(self):
        return [dict(name="MME"), dict(name="Windows WASAPI")]

    def check_input_settings(self, *, device, channels, samplerate, dtype):
        if device != 1 or not 1 <= channels <= 2 or samplerate not in (44100, 48000) or dtype != "float32":
            raise ValueError("Unsupported input settings")

    def InputStream(self, **settings):
        if self.fault == "configure":
            raise RuntimeError("device unavailable")
        self.stream = FakeStream(self, **settings)
        return self.stream


def configuration(*, target=6, channel_indices=(1, 0), queue_blocks=8, block_frames=4):
    return AcquisitionConfig(tuple(ChannelConfig(f"soundcard:1/input{index}", f"Mic {index+1}",
        "FS", measurement_type="physical", input_min=-1, input_max=1) for index in channel_indices),
        48000, block_frames=block_frames, queue_blocks=queue_blocks, read_timeout=.02,
        target_frames=target, minimum_free_bytes=0)


def backend(api, indices=(1, 0)):
    from autoacoustics.acquisition.soundcard import SoundCardBackend
    return SoundCardBackend(1, indices, sd_module=api)


def test_channel_config_accepts_fs_without_physical_pressure_relabelling():
    channel = ChannelConfig("soundcard:1/input0", "Mic", "FS", measurement_type="physical")
    assert channel.unit == "FS" and channel.sensitivity_mv_pa is None


def test_discovery_excludes_output_only_devices_and_keeps_actual_host_api():
    from autoacoustics.acquisition.soundcard import discover_soundcards
    result = discover_soundcards(sd_module=FakeSoundDevice())
    assert result.available and len(result.devices) == 1
    device = result.devices[0]
    assert device.device_id == "soundcard:1"
    assert device.physical_channels == ("soundcard:1/input0", "soundcard:1/input1")
    assert device.metadata["host_api"] == "Windows WASAPI"
    assert device.metadata["is_default_input"] is True
    assert device.metadata["default_samplerate"] == 48000
    assert device.is_simulated and device.supports_realtime_samples


def test_missing_portaudio_returns_a_reason_and_never_invents_a_microphone():
    from autoacoustics.acquisition.soundcard import discover_soundcards
    class Missing(FakeSoundDevice):
        def query_devices(self):
            raise OSError("PortAudio not available")
    result = discover_soundcards(sd_module=Missing())
    assert not result.available and result.devices == () and "PortAudio" in result.reason


def test_finite_soundcard_session_keeps_channel_mapping_fs_and_exact_target(tmp_path):
    from autoacoustics.acquisition.soundcard import export_soundcard_wav
    api = FakeSoundDevice(actual_rate=44100)
    engine = RecordingEngine(backend(api), configuration(), tmp_path / "麦克风采集")
    configured = engine.configure()
    assert configured.actual_sample_rate == 44100
    engine.start()
    result = engine.wait(5)
    assert result.status == AcquisitionStatus.COMPLETE
    assert result.frames_expected == result.frames_read == result.frames_written == 6
    signal = import_signal(result.manifest_path)
    expected = np.array([[-.25, -.5, -.75, -.125, -.25, -.375],
                         [.125, .25, .375, .5, .625, .75]], dtype=np.float32)
    np.testing.assert_array_equal(signal.samples, expected)
    assert signal.sample_rate == 44100 and [channel.unit for channel in signal.channels] == ["FS", "FS"]
    assert signal.metadata["scaling"]["applied_by_importer"] is False
    assert all(not channel.scaled for channel in signal.channels)
    assert signal.metadata["acquisition_session"]["calibration_applied_to_raw"] is False
    wav = export_soundcard_wav(result.manifest_path)
    values, fs = sf.read(wav, always_2d=True)
    np.testing.assert_array_equal(values.T, expected)
    assert fs == 44100 and sf.info(wav).subtype == "FLOAT"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["wav_file"] == wav.name and len(manifest["wav_sha256"]) == 64
    assert api.stream.closed


def test_continuous_stop_preserves_all_callback_delivered_frames_and_drains(tmp_path):
    api = FakeSoundDevice(keep_active=True)
    engine = RecordingEngine(backend(api), configuration(target=None), tmp_path / "continuous")
    engine.configure(); engine.start()
    deadline = time.monotonic() + 2
    while engine.snapshot().frames_read < 8 and time.monotonic() < deadline:
        time.sleep(.005)
    result = engine.stop(5)
    assert result.status == AcquisitionStatus.COMPLETE
    assert result.frames_read == result.frames_written == result.frames_expected == 8
    np.testing.assert_array_equal(open_acquisition_session(result.manifest_path).samples,
        [[-.25, -.5, -.75, -.125, -.25, -.375, -.5, -.625],
         [.125, .25, .375, .5, .625, .75, .875, .5]])
    assert api.stream.closed


def test_callback_queue_backpressure_is_failed_with_recoverable_prefix(tmp_path):
    api = FakeSoundDevice(interval=0)
    device = backend(api)
    config = configuration(target=None, queue_blocks=1)
    device.configure(config); device.start()
    deadline = time.monotonic() + 2
    while api.stream.active and time.monotonic() < deadline:
        time.sleep(.005)
    first = device.read_block(.02)
    assert first.frames == 4
    with pytest.raises(AcquisitionError) as caught:
        device.read_block(.02)
    assert caught.value.flag == "application_backpressure"
    assert device.stop_acquisition() == 8
    assert tuple(device.drain_blocks()) == ()
    device.close()


def test_recording_engine_backpressure_keeps_a_recoverable_sample_prefix(tmp_path):
    from autoacoustics.acquisition.storage import SessionWriter
    class SlowWriter(SessionWriter):
        def append(self, block):
            time.sleep(.1)
            super().append(block)
    api = FakeSoundDevice(interval=.005)
    api.blocks *= 8
    engine = RecordingEngine(backend(api), configuration(target=64, queue_blocks=1),
        tmp_path / "backpressure", writer_factory=SlowWriter)
    engine.configure(); engine.start()
    result = engine.wait(5)
    assert result.status == AcquisitionStatus.FAILED and "application_backpressure" in result.flags
    assert 0 < result.frames_written < result.frames_expected and api.stream.closed
    recover_session(result.manifest_path)
    recovered = open_acquisition_session(result.manifest_path, allow_partial=True)
    assert recovered.frames == result.frames_written
    np.testing.assert_array_equal(recovered.samples[:, :4],
                                 [[-.25, -.5, -.75, -.125], [.125, .25, .375, .5]])


@pytest.mark.parametrize("fault,flag", [("overflow", "device_overflow"),
    ("disconnect", "device_disconnected"), ("stop", "stop_failed")])
def test_driver_quality_failures_cannot_be_opened_as_complete_measurement(tmp_path, fault, flag):
    api = FakeSoundDevice(fault=fault)
    engine = RecordingEngine(backend(api), configuration(), tmp_path / fault)
    engine.configure(); engine.start()
    result = engine.wait(5)
    assert result.status == AcquisitionStatus.FAILED and flag in result.flags
    assert result.frames_written > 0 and api.stream.closed
    with pytest.raises(AcquisitionError):
        open_acquisition_session(result.manifest_path)
    recover_session(result.manifest_path)
    recovered = open_acquisition_session(result.manifest_path, allow_partial=True)
    assert "incomplete_recording" in recovered.quality


@pytest.mark.parametrize("fault", ["configure", "start"])
def test_configure_or_start_failure_releases_opened_stream(tmp_path, fault):
    api = FakeSoundDevice(fault=fault)
    engine = RecordingEngine(backend(api), configuration(), tmp_path / fault)
    with pytest.raises(AcquisitionError):
        engine.configure(); engine.start()
    assert api.stream is None or api.stream.closed


def test_empty_active_stream_times_out_instead_of_hanging_capture_thread(tmp_path):
    api = FakeSoundDevice(blocks=[np.empty((0, 2), dtype=np.float32)], keep_active=True)
    engine = RecordingEngine(backend(api), configuration(target=None), tmp_path / "stalled")
    engine.configure(); engine.start()
    result = engine.wait(5)
    assert result.status == AcquisitionStatus.FAILED
    assert "device_read_timeout" in result.flags and api.stream.closed


@pytest.mark.parametrize("indices", [(2,), (0, 0), (), (-1,), (True,)])
def test_unavailable_or_repeated_input_channels_are_rejected(indices):
    with pytest.raises(AcquisitionError):
        device = backend(FakeSoundDevice(), indices)
        device.configure(configuration(channel_indices=indices))


def test_soundcard_cannot_accept_an_unconfirmed_pa_unit_or_sensor_scale():
    device = backend(FakeSoundDevice(), (0,))
    config = AcquisitionConfig((ChannelConfig("soundcard:1/input0", "Mic", "Pa", measurement_type="physical"),),
                               48000)
    with pytest.raises(AcquisitionError):
        device.configure(config)


def test_callback_larger_than_requested_blocks_is_split_without_losing_frames(tmp_path):
    values = np.array([[.125, -.25], [.25, -.5], [.375, -.75], [.5, -.125],
                       [.625, -.25], [.75, -.375]], dtype=np.float32)
    api = FakeSoundDevice(blocks=[values])
    engine = RecordingEngine(backend(api), configuration(block_frames=4), tmp_path / "varying-block")
    engine.configure(); engine.start()
    result = engine.wait(5)
    assert result.status == AcquisitionStatus.COMPLETE and result.frames_written == 6
    np.testing.assert_array_equal(import_signal(result.manifest_path).samples,
                                 [[-.25, -.5, -.75, -.125, -.25, -.375],
                                  [.125, .25, .375, .5, .625, .75]])


def test_stop_request_keeps_the_callback_buffer_already_delivered_by_driver():
    api = FakeSoundDevice()
    device = backend(api)
    device.configure(configuration(target=None))
    device.request_stop()
    # Stop is requested at an arbitrary time. A buffer already handed to the
    # callback still belongs to this capture and must be counted and drained.
    with pytest.raises(CallbackStop):
        device._callback(api.blocks[0], 4, SimpleNamespace(inputBufferAdcTime=10),
                         SimpleNamespace(input_overflow=False, input_underflow=False))
    assert device.stop_acquisition() == 4
    pending = tuple(device.drain_blocks())
    assert len(pending) == 1 and pending[0].frames == 4
    np.testing.assert_array_equal(pending[0].samples,
                                 [[-.25, -.5, -.75, -.125], [.125, .25, .375, .5]])
    device.close()


def test_wav_retry_recovers_our_completed_file_after_manifest_commit_failure(tmp_path, monkeypatch):
    from autoacoustics.acquisition.soundcard import export_soundcard_wav
    from autoacoustics.acquisition import storage
    engine = RecordingEngine(backend(FakeSoundDevice()), configuration(), tmp_path / "wav-transaction")
    engine.configure(); engine.start()
    session = engine.wait(5)
    original_write = storage.write_manifest
    def denied_manifest(path, data):
        if path == session.manifest_path:
            raise PermissionError("temporary session manifest lock")
        return original_write(path, data)
    monkeypatch.setattr(storage, "write_manifest", denied_manifest)
    with pytest.raises(AcquisitionError):
        export_soundcard_wav(session.manifest_path)
    wav_path = session.manifest_path.parent / "recording.wav"
    assert wav_path.is_file()
    original_bytes = wav_path.read_bytes()
    monkeypatch.setattr(storage, "write_manifest", original_write)
    assert export_soundcard_wav(session.manifest_path) == wav_path
    assert wav_path.read_bytes() == original_bytes
    manifest = json.loads(session.manifest_path.read_text(encoding="utf-8"))
    assert manifest["wav_file"] == "recording.wav" and manifest["wav_calibration_applied"] is False


def test_wav_retry_never_overwrites_a_changed_file_from_the_transaction(tmp_path, monkeypatch):
    from autoacoustics.acquisition.soundcard import export_soundcard_wav
    from autoacoustics.acquisition import storage
    engine = RecordingEngine(backend(FakeSoundDevice()), configuration(), tmp_path / "wav-changed")
    engine.configure(); engine.start()
    session = engine.wait(5)
    original_write = storage.write_manifest
    def denied_manifest(path, data):
        if path == session.manifest_path:
            raise PermissionError("temporary session manifest lock")
        return original_write(path, data)
    monkeypatch.setattr(storage, "write_manifest", denied_manifest)
    with pytest.raises(AcquisitionError):
        export_soundcard_wav(session.manifest_path)
    wav_path = session.manifest_path.parent / "recording.wav"
    wav_path.write_bytes(b"This is a changed file. Do not overwrite it.")
    monkeypatch.setattr(storage, "write_manifest", original_write)
    with pytest.raises(AcquisitionError):
        export_soundcard_wav(session.manifest_path)
    assert wav_path.read_bytes() == b"This is a changed file. Do not overwrite it."


def test_wav_export_never_claims_an_unknown_existing_file_even_when_samples_match(tmp_path):
    from autoacoustics.acquisition.soundcard import export_soundcard_wav
    engine = RecordingEngine(backend(FakeSoundDevice()), configuration(), tmp_path / "wav-unknown")
    engine.configure(); engine.start()
    session = engine.wait(5)
    wav_path = session.manifest_path.parent / "recording.wav"
    samples = open_acquisition_session(session.manifest_path).samples
    sf.write(wav_path, samples.T, 48000, subtype="FLOAT")
    original = wav_path.read_bytes()
    with pytest.raises(AcquisitionError):
        export_soundcard_wav(session.manifest_path)
    assert wav_path.read_bytes() == original
