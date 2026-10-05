"""Recording UI regressions; real engine/storage with an injected input boundary."""
from dataclasses import replace
import json
import time

import numpy as np
import pytest

from autoacoustics.acquisition.device import DeviceCapabilities, DiscoveryResult
from autoacoustics.acquisition.replay import ReplayBackend
from autoacoustics.acquisition.storage import open_acquisition_session
from autoacoustics.ui.soundcard_panel import SoundCardPanel


def input_device(index, name, *, default=False):
    return DeviceCapabilities(f"soundcard:{index}", name, "SOUND_CARD",
        tuple(f"soundcard:{index}/input{i}" for i in range(2)),
        supports_realtime_samples=True, supports_controlled_recording=True,
        is_simulated=True, metadata={"device_index": index, "host_api": "Windows WASAPI",
            "default_samplerate": 48000., "max_input_channels": 2,
            "is_default_input": default})


def wait_for(app, predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    app.processEvents()
    assert predicate(), "Recording panel did not finish its asynchronous task."


class CapturedInput(ReplayBackend):
    """Only the microphone boundary is simulated; raw storage and WAV are real."""
    def __init__(self, device, *, actual_rate=None):
        super().__init__(actual_sample_rate=actual_rate, pace_seconds=.01,
            generator=lambda first, count, channels:
                np.tile(.1*np.sin((first+np.arange(count))/20), (channels, 1)).astype(np.float32))
        self.device, self.start_count = device, 0

    def configure(self, config):
        return replace(super().configure(config), device=self.device)

    def start(self):
        self.start_count += 1
        return super().start()


def test_refresh_keeps_explicit_device_channel_and_requested_rate(qt_application_lifetime):
    devices = [input_device(1, "Built-in microphone"), input_device(2, "USB measurement input")]
    panel = SoundCardPanel(discovery_fn=lambda: DiscoveryResult(True, tuple(devices)))
    panel.device_combo.setCurrentIndex(1)
    panel.channel_combo.setCurrentIndex(1)
    panel.rate_combo.setCurrentText("44100")
    panel.refresh_devices()
    assert panel.device_combo.currentData().name == "USB measurement input"
    assert panel.channel_combo.currentData() == (1,)
    assert panel.rate_combo.currentText() == "44100"


def test_refresh_follows_device_identity_after_index_reordering(qt_application_lifetime):
    devices = [input_device(1, "Built-in microphone"), input_device(2, "USB measurement input")]
    panel = SoundCardPanel(discovery_fn=lambda: DiscoveryResult(True, tuple(devices)))
    panel.device_combo.setCurrentIndex(1)
    panel.channel_combo.setCurrentIndex(2)
    panel.rate_combo.setCurrentText("96000")
    devices[:] = [input_device(2, "Built-in microphone"), input_device(7, "USB measurement input")]
    panel.refresh_devices()
    assert panel.device_combo.currentData().metadata["device_index"] == 7
    assert panel.device_combo.currentData().name == "USB measurement input"
    assert panel.channel_combo.currentData() == (0, 1)
    assert panel.rate_combo.currentText() == "96000"


def test_refresh_removed_device_requires_explicit_selection(qt_application_lifetime):
    devices = [input_device(1, "Built-in microphone"), input_device(2, "USB measurement input")]
    panel = SoundCardPanel(discovery_fn=lambda: DiscoveryResult(True, tuple(devices)))
    panel.device_combo.setCurrentIndex(1)
    devices[:] = [input_device(1, "Built-in microphone")]
    panel.refresh_devices()
    assert panel.device_combo.currentData() is None
    assert not panel.start_button.isEnabled()
    assert "重新选择" in panel.status_label.text()
    panel.device_combo.setCurrentIndex(0)
    assert panel.start_button.isEnabled()


def test_refresh_ambiguous_same_name_inputs_requires_explicit_selection(qt_application_lifetime):
    devices = [input_device(1, "Built-in microphone"), input_device(2, "USB measurement input")]
    panel = SoundCardPanel(discovery_fn=lambda: DiscoveryResult(True, tuple(devices)))
    panel.device_combo.setCurrentIndex(1)
    devices[:] = [input_device(7, "USB measurement input"), input_device(8, "USB measurement input")]
    panel.refresh_devices()
    assert panel.device_combo.currentData() is None
    assert not panel.start_button.isEnabled()
    assert "重新选择" in panel.status_label.text()


def test_reconnect_after_empty_refreshes_restores_input_and_rate(qt_application_lifetime):
    devices = [input_device(1, "Built-in microphone"), input_device(2, "USB measurement input")]
    panel = SoundCardPanel(discovery_fn=lambda: DiscoveryResult(True, tuple(devices)))
    panel.device_combo.setCurrentIndex(1)
    panel.channel_combo.setCurrentIndex(1)
    panel.rate_combo.setCurrentText("44100")
    devices[:] = [input_device(1, "Built-in microphone")]
    panel.refresh_devices()
    assert panel.device_combo.currentData() is None and not panel.start_button.isEnabled()
    devices.clear()
    panel.refresh_devices()
    panel.refresh_devices()
    assert panel.device_combo.currentData() is None and panel.channel_combo.count() == 0
    assert not panel.start_button.isEnabled()
    devices[:] = [input_device(1, "Built-in microphone"), input_device(7, "USB measurement input")]
    panel.refresh_devices()
    assert panel.device_combo.currentData().metadata["device_index"] == 7
    assert panel.channel_combo.currentData() == (1,)
    assert panel.rate_combo.currentText() == "44100"
    assert panel.start_button.isEnabled()


def test_shorter_second_recording_resets_and_bounds_audition(qt_application_lifetime, tmp_path):
    device = input_device(2, "USB measurement input")
    panel = SoundCardPanel(discovery_fn=lambda: DiscoveryResult(True, (device,)),
        backend_factory=lambda index, channels: CapturedInput(device))
    panel.save_parent.setText(str(tmp_path))
    panel.target_duration.setValue(.25)
    panel.start_recording()
    wait_for(qt_application_lifetime, lambda: panel.listen_button.isEnabled())
    first_manifest = panel.completed_manifest
    panel.listen_start.setValue(.2)
    panel.target_duration.setValue(.05)
    panel.start_recording()
    wait_for(qt_application_lifetime, lambda: panel.listen_button.isEnabled())
    assert panel.completed_manifest != first_manifest
    assert panel.completed_wav.is_file()
    assert panel.listen_start.value() == 0
    assert panel.listen_end.value() == pytest.approx(.05)
    assert panel.listen_start.maximum() == pytest.approx(.05)
    assert panel.listen_end.maximum() == pytest.approx(.05)
    assert open_acquisition_session(panel.completed_manifest).frames == 2400


def test_fixed_duration_rejects_clock_difference_before_start_then_can_retry(qt_application_lifetime, tmp_path):
    device = input_device(2, "USB measurement input")
    backends = []
    def create(index, channels):
        backend = CapturedInput(device, actual_rate=44100)
        backends.append(backend)
        return backend
    panel = SoundCardPanel(discovery_fn=lambda: DiscoveryResult(True, (device,)),
        backend_factory=create)
    panel.save_parent.setText(str(tmp_path))
    panel.target_duration.setValue(.1)
    panel.start_recording()
    wait_for(qt_application_lifetime, lambda: not panel.is_recording)
    assert panel.completed_manifest is None and panel.completed_wav is None
    assert backends[0].start_count == 0 and backends[0].closed
    assert "48000" in panel.status_label.text() and "44100" in panel.status_label.text()
    assert "固定时长" in panel.status_label.text()
    assert not tuple(tmp_path.rglob("session.json"))
    panel.rate_combo.setCurrentText("44100")
    panel.start_recording()
    wait_for(qt_application_lifetime, lambda: panel.listen_button.isEnabled())
    signal = open_acquisition_session(panel.completed_manifest)
    manifest = json.loads(panel.completed_manifest.read_text(encoding="utf-8"))
    assert signal.frames == manifest["frames_expected"] == 4410
    assert signal.frames/signal.sample_rate == pytest.approx(.1)
    assert manifest["configuration"]["target_frames"] == 4410
    assert manifest["requested_sample_rate"] == manifest["actual_sample_rate"] == 44100


def test_continuous_recording_keeps_driver_actual_clock(qt_application_lifetime, tmp_path):
    device = input_device(2, "USB measurement input")
    panel = SoundCardPanel(discovery_fn=lambda: DiscoveryResult(True, (device,)),
        backend_factory=lambda index, channels: CapturedInput(device, actual_rate=44100))
    panel.save_parent.setText(str(tmp_path))
    panel.start_recording()
    wait_for(qt_application_lifetime,
        lambda: panel._engine is not None and panel._engine.snapshot().frames_read > 0)
    panel.stop_recording()
    wait_for(qt_application_lifetime, lambda: panel.listen_button.isEnabled())
    signal = open_acquisition_session(panel.completed_manifest)
    assert signal.sample_rate == 44100
    assert signal.metadata["acquisition_session"]["requested_sample_rate"] == 48000
    assert signal.frames == signal.metadata["acquisition_session"]["frames_written"]
