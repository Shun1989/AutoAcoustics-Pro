import json
import os
from pathlib import Path
import time
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QRect,QPoint

from autoacoustics.acquisition.device import AcquisitionStatus, DeviceCapabilities, DiscoveryResult
from autoacoustics.acquisition.replay import ReplayBackend
from autoacoustics.acquisition.storage import open_acquisition_session
from autoacoustics.ui import acquisition_dialog as ui


@pytest.fixture
def application():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


def wait_for(app, predicate, timeout=5):
    deadline = time.monotonic()+timeout
    while not predicate():
        app.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for DAQ UI")
        time.sleep(0.002)
    app.processEvents()


def simulated_device(monkeypatch, backend):
    device = DeviceCapabilities("Dev1", "假驱动设备（模拟）", "NI_DAQMX", ("Dev1/ai0", "Dev1/ai1"),
                                supports_realtime_samples=True, is_simulated=True)
    monkeypatch.setattr(ui, "discover_ni", lambda: DiscoveryResult(True, (device,), python_package_version="FAKE"))
    monkeypatch.setattr(ui, "NiDaqmxBackend", lambda: backend)


def fill_voltage(dialog, output, *, finite=None):
    dialog.add_channel_row()
    table = dialog.channel_table
    table.cellWidget(0, 0).setCurrentIndex(1)
    table.cellWidget(0, 1).setText("测试电压")
    table.cellWidget(0, 2).setCurrentIndex(1)
    table.cellWidget(0, 3).setText("-5")
    table.cellWidget(0, 4).setText("5")
    dialog.requested_rate.setText("48000")
    dialog.output_directory.setText(str(output))
    dialog.target_frames.setText(str(finite) if finite is not None else "")


def test_default_dialog_displays_missing_driver_and_head_control_reason_without_mock(application, monkeypatch):
    monkeypatch.setattr(ui, "discover_ni", lambda: DiscoveryResult(False, reason="NI-DAQmx installation missing"))
    dialog = ui.AcquisitionDialog()
    wait_for(application, lambda: not dialog._busy)
    assert "NI-DAQmx" in dialog.device_status.text()
    assert not dialog.configure_button.isEnabled() and not dialog.start_button.isEnabled()
    assert dialog.device_combo.count() == 0
    assert "ASX" in dialog.head_control_status.text() and "SDK" in dialog.head_control_status.text()
    assert "电机电源" in dialog.stop_notice.text()
    dialog.close()


def test_ni_dialog_explicit_config_actual_rate_complete_signal_and_event_metadata(application, monkeypatch, tmp_path):
    simulated_device(monkeypatch, ReplayBackend(actual_sample_rate=51200, pace_seconds=.003))
    dialog = ui.AcquisitionDialog()
    ready = []
    dialog.recordingReady.connect(ready.append)
    wait_for(application, lambda: not dialog._busy)
    fill_voltage(dialog, tmp_path)
    dialog.cw_reference.setText("从轴输出端看")
    dialog.configure_recording()
    wait_for(application, lambda: not dialog._busy)
    assert "51200" in dialog.rate_status.text() and "模拟" in dialog.rate_status.text()
    dialog.start_recording()
    wait_for(application, lambda: dialog._engine and dialog._engine.snapshot().frames_read >= 4096)
    assert not dialog.channel_table.isEnabled() and not dialog.requested_rate.isEnabled()
    dialog.event_kind.setCurrentText("稳定")
    dialog.event_direction.setCurrentText("CW")
    dialog.add_event()
    dialog.stop_recording()
    wait_for(application, lambda: bool(ready))
    manifest_path = ready[0]
    assert isinstance(manifest_path, Path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and manifest["actual_sample_rate"] == 51200
    assert manifest["measurement_context"]["cw_reference"] == "从轴输出端看"
    assert manifest["events"][0]["kind"] == "稳定" and manifest["events"][0]["direction"] == "CW"
    assert open_acquisition_session(manifest_path).channels[0].unit == "V"
    dialog.close()


def test_failed_acquisition_never_emits_success_and_partial_recovery_only_previews(application, monkeypatch, tmp_path):
    simulated_device(monkeypatch, ReplayBackend(actual_sample_rate=48000, fault="gap", fault_at_block=2))
    dialog = ui.AcquisitionDialog()
    ready = []
    dialog.recordingReady.connect(ready.append)
    wait_for(application, lambda: not dialog._busy)
    fill_voltage(dialog, tmp_path, finite=32768)
    dialog.configure_recording()
    wait_for(application, lambda: not dialog._busy)
    dialog.start_recording()
    wait_for(application, lambda: dialog._engine and dialog._engine.snapshot().status == AcquisitionStatus.FAILED)
    dialog.poll_recording()
    assert not ready and dialog.recover_button.isEnabled() and "失败" in dialog.recording_status.text()
    dialog.recover_partial()
    wait_for(application, lambda: not dialog._busy)
    assert "不完整" in dialog.recording_status.text() and not ready
    assert dialog._engine.snapshot().status == AcquisitionStatus.FAILED
    dialog.close()


def test_head_stopped_file_handoff_emits_only_verified_session(application, monkeypatch, tmp_path):
    monkeypatch.setattr(ui, "discover_ni", lambda: DiscoveryResult(False, reason="NI missing"))
    with zipfile.ZipFile(Path(__file__).resolve().parents[1]/"压缩.zip") as archive:
        blob = archive.read("HDF/01-01CW.hdf")
    source = tmp_path/"现场录制.hdf"
    source.write_bytes(blob)
    dialog = ui.AcquisitionDialog()
    ready = []
    dialog.recordingReady.connect(ready.append)
    wait_for(application, lambda: not dialog._busy)
    dialog.head_source.setText(str(source))
    dialog.head_directory.setText(str(tmp_path))
    dialog.import_head_recording()
    assert not ready and "停止" in dialog.head_file_status.text()
    dialog.head_stopped.setChecked(True)
    dialog.import_head_recording()
    wait_for(application, lambda: bool(ready))
    signal = open_acquisition_session(ready[0])
    assert signal.channels[0].unit == "Pa" and source.read_bytes() == blob
    assert signal.metadata["acquisition_session"]["controlled_recording"] is False
    dialog.close()


def test_closing_recording_dialog_requests_stop_and_waits_for_drained_completion(application, monkeypatch, tmp_path):
    simulated_device(monkeypatch, ReplayBackend(pace_seconds=.01, stop_tail_frames=3))
    dialog = ui.AcquisitionDialog()
    dialog.show()
    wait_for(application, lambda: not dialog._busy)
    fill_voltage(dialog, tmp_path)
    dialog.configure_recording()
    wait_for(application, lambda: not dialog._busy)
    dialog.start_recording()
    wait_for(application, lambda: dialog._engine and dialog._engine.snapshot().frames_read >= 4096)
    dialog.close()
    wait_for(application, lambda: not dialog.isVisible())
    result = dialog._engine.snapshot()
    assert result.status == AcquisitionStatus.COMPLETE
    assert result.frames_expected == result.frames_read == result.frames_written
    assert result.frames_written % 4096 == 3


def test_editing_a_configured_sensor_or_rate_requires_fresh_driver_configuration(application, monkeypatch, tmp_path):
    backend = ReplayBackend(actual_sample_rate=51200)
    simulated_device(monkeypatch, backend)
    dialog = ui.AcquisitionDialog()
    wait_for(application, lambda: not dialog._busy)
    fill_voltage(dialog, tmp_path)
    dialog.configure_recording()
    wait_for(application, lambda: not dialog._busy)
    assert dialog.start_button.isEnabled()
    dialog.channel_table.cellWidget(0, 10).setText("新传感器记录")
    assert not dialog.start_button.isEnabled() and backend.closed
    assert "重新" in dialog.rate_status.text()
    dialog.close()


def test_escape_reject_cannot_leave_an_active_recording_hidden(application, monkeypatch, tmp_path):
    simulated_device(monkeypatch, ReplayBackend(pace_seconds=.01, stop_tail_frames=3))
    dialog = ui.AcquisitionDialog()
    dialog.show()
    wait_for(application, lambda: not dialog._busy)
    fill_voltage(dialog, tmp_path)
    dialog.configure_recording()
    wait_for(application, lambda: not dialog._busy)
    dialog.start_recording()
    wait_for(application, lambda: dialog._engine and dialog._engine.snapshot().frames_read >= 4096)
    dialog.reject()
    wait_for(application, lambda: not dialog.isVisible())
    assert dialog._engine.snapshot().status == AcquisitionStatus.COMPLETE
    assert dialog._engine.snapshot().frames_written % 4096 == 3

def test_small_screen_scroll_keeps_stop_and_close_reachable_after_show(application,monkeypatch):
    from autoacoustics.ui import window_geometry
    from PyQt6.QtWidgets import QScrollArea
    available=QRect(0,0,1280,680)
    monkeypatch.setattr(window_geometry,'available_geometry',lambda window:available)
    monkeypatch.setattr(ui,'discover_ni',lambda:DiscoveryResult(False,reason='NI driver unavailable'))
    dialog=ui.AcquisitionDialog();dialog.show();wait_for(application,lambda:not dialog._busy)
    assert isinstance(dialog.ni_scroll,QScrollArea) and dialog.minimumSizeHint().height()<600
    assert available.contains(dialog.frameGeometry())
    stop=dialog.stop_button;close=dialog.findChild(ui.QPushButton,'daqClose')
    assert close is not None and stop.isVisible() and close.isVisible()
    stop_top=stop.mapToGlobal(QPoint(0,0));close_top=close.mapToGlobal(QPoint(0,0))
    assert available.contains(QRect(stop_top,stop.size())) and available.contains(QRect(close_top,close.size()))
    dialog.ni_scroll.verticalScrollBar().setValue(dialog.ni_scroll.verticalScrollBar().maximum());application.processEvents()
    assert stop.mapToGlobal(QPoint(0,0))==stop_top
    assert close.mapToGlobal(QPoint(0,0))==close_top
    dialog.tabs.setCurrentIndex(1);application.processEvents();assert available.contains(dialog.frameGeometry())
    dialog.close()
