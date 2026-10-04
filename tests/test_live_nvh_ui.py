"""Live NVH panels use real algorithms; acquisition hardware is explicitly injected."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from autoacoustics.acquisition.device import DeviceCapabilities, DiscoveryResult, PreviewData
from autoacoustics.acquisition.replay import ReplayBackend
from autoacoustics.model import ChannelInfo, SignalData


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def wait_for(app, predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    app.processEvents()
    assert predicate(), 'Qt panel did not reach its expected terminal state.'


def card(index=9, host='Windows WASAPI', default=False):
    return DeviceCapabilities(f'soundcard:{index}', 'Injected input for software test', 'SOUND_CARD',
        tuple(f'soundcard:{index}/input{i}' for i in range(2)),
        supports_realtime_samples=True, supports_controlled_recording=True, is_simulated=True,
        metadata={'device_index':index, 'host_api':host, 'default_samplerate':48000.,
                  'max_input_channels':2, 'is_default_input':default})


def discovery():
    return DiscoveryResult(True, (card(1, 'MME', True), card()))


class InjectedCapture(ReplayBackend):
    """Exercise real RecordingEngine/storage without claiming physical acquisition."""
    def configure(self, config):
        configured = super().configure(config)
        return replace(configured, device=card())


def test_soundcard_no_device_explains_action_and_never_starts(app, tmp_path):
    from autoacoustics.ui.soundcard_panel import SoundCardPanel
    panel = SoundCardPanel(discovery_fn=lambda:DiscoveryResult(False, reason='麦克风权限被拒绝；请开启 Windows 麦克风访问'))
    panel.save_parent.setText(str(tmp_path))
    assert not panel.start_button.isEnabled()
    assert '权限' in panel.status_label.text()
    assert panel.shutdown() and not panel.is_recording


def test_soundcard_selection_preserves_real_host_api_and_explicit_channels(app):
    from autoacoustics.ui.soundcard_panel import SoundCardPanel
    panel = SoundCardPanel(discovery_fn=discovery)
    assert panel.device_combo.count() == 2
    assert 'WASAPI' in panel.device_combo.currentText()
    assert 'MME' in panel.device_combo.itemText(0)
    assert panel.channel_combo.count() >= 3
    assert 'FS' in panel.metrics_label.text()
    assert panel.shutdown()


def test_soundcard_preview_is_bounded_and_labels_full_scale_not_spl(app):
    from autoacoustics.ui.soundcard_panel import SoundCardPanel
    panel = SoundCardPanel(discovery_fn=discovery)
    samples = np.zeros((2, 48000 * 4))
    samples[0, -100:] = 1.1
    panel._show_preview(PreviewData(samples, 48000, 100))
    assert panel.preview_frames <= 3 * 48000
    assert panel.preview_meters['clipped']
    assert panel.preview_meters['peak_dbfs'] > 0
    assert 'dBFS' in panel.metrics_label.text() and '未校准' in panel.metrics_label.text()
    assert 'SPL' not in panel.metrics_label.text()
    assert panel.shutdown()


def test_soundcard_complete_writes_hdf_and_float_wav_then_emits_only_once(app, tmp_path):
    from autoacoustics.ui.soundcard_panel import SoundCardPanel
    from autoacoustics.importers.registry import import_signal
    import soundfile as sf
    panel = SoundCardPanel(discovery_fn=discovery,
        backend_factory=lambda index, channels:InjectedCapture(pace_seconds=.01,
            generator=lambda first, count, n:np.tile(.1*np.sin(2*np.pi*500*(first+np.arange(count))/48000), (n, 1)).astype(np.float32)))
    panel.save_parent.setText(str(tmp_path))
    panel.target_duration.setValue(.25)
    panel.direction_combo.setCurrentText('CW')
    ready = []
    panel.recordingReady.connect(ready.append)
    panel.start_recording()
    assert panel.is_recording and not panel.device_combo.isEnabled()
    wait_for(app, lambda:panel.send_button.isEnabled())
    assert not ready, 'Completed recording must await the explicit send action.'
    manifest = panel.completed_manifest
    signal = import_signal(manifest)
    assert signal.frames == 12000 and signal.sample_rate == 48000
    assert signal.channels[0].unit == 'FS'
    assert json.loads(manifest.read_text(encoding='utf-8'))['measurement_context']['motor_rotation'] == 'CW'
    assert panel.completed_wav.is_file()
    info = sf.info(panel.completed_wav)
    assert info.subtype == 'FLOAT' and info.frames == 12000 and info.samplerate == 48000
    panel.send_to_analysis();panel.send_to_analysis();panel.poll_recording()
    assert ready == [manifest]
    assert panel.shutdown()


def test_soundcard_stop_shutdown_drains_without_emitting_handoff(app, tmp_path):
    from autoacoustics.ui.soundcard_panel import SoundCardPanel
    panel = SoundCardPanel(discovery_fn=discovery,
        backend_factory=lambda index, channels:InjectedCapture(pace_seconds=.04, fault='disconnect', fault_at_block=1))
    panel.save_parent.setText(str(tmp_path))
    ready = []
    panel.recordingReady.connect(ready.append)
    panel.start_recording()
    assert not panel.shutdown()
    wait_for(app, lambda:not panel.is_recording)
    panel.poll_recording()
    assert not panel.send_button.isEnabled() or panel.completed_manifest is not None
    assert not ready
    assert panel.shutdown()


def test_soundcard_capture_failure_is_not_advertised_as_complete(app, tmp_path):
    from autoacoustics.ui.soundcard_panel import SoundCardPanel
    panel = SoundCardPanel(discovery_fn=discovery,
        backend_factory=lambda index, channels:InjectedCapture(pace_seconds=.02, fault='disconnect', fault_at_block=1))
    panel.save_parent.setText(str(tmp_path))
    panel.start_recording()
    wait_for(app, lambda:not panel.is_recording)
    panel.poll_recording()
    assert panel.completed_manifest is None and panel.completed_wav is None
    assert not panel.send_button.isEnabled()
    assert '不完整' in panel.status_label.text()
    assert panel.shutdown()


def test_soundcard_new_start_failure_clears_previous_preview(app, tmp_path):
    from autoacoustics.ui.soundcard_panel import SoundCardPanel
    from autoacoustics.acquisition.device import AcquisitionError
    class DeniedInput(ReplayBackend):
        def configure(self, config):
            raise AcquisitionError('麦克风权限被拒绝')
    panel = SoundCardPanel(discovery_fn=discovery,
        backend_factory=lambda index, channels:DeniedInput())
    panel.save_parent.setText(str(tmp_path))
    panel._show_preview(PreviewData(np.full((1, 48000), .5), 48000, 0))
    assert panel.preview_frames == 48000
    panel.start_recording()
    wait_for(app, lambda:not panel.is_recording)
    assert panel.preview_frames == 0 and not panel.preview_meters and not panel._last_plot
    assert panel.wave_curve.xData is None and panel.spectrum_curve.xData is None
    assert '等待' in panel.metrics_label.text()
    assert '权限' in panel.status_label.text()
    assert panel.shutdown()


def rotation_signal(origin=10., source_hash='input-hash'):
    fs = 12000
    t = np.arange(fs * 3) / fs
    samples = .2*np.sin(2*np.pi*150*t) + .03*np.sin(2*np.pi*1000*t)
    return SignalData(samples[None, :], fs, (ChannelInfo('Mic', 'FS'),), Path('real-input.wav'),
                      source_hash, {'time_origin_seconds':origin})


def test_rotation_real_synthetic_analysis_keeps_selection_source_and_overlays(app):
    from autoacoustics.ui.rotation_panel import RotationPanel
    panel = RotationPanel()
    panel.set_signal(rotation_signal(), start_seconds=10.5, end_seconds=11.5)
    panel.gear_teeth_spin.setValue(20)
    overlay = []
    panel.overlayReady.connect(overlay.append)
    panel.start_analysis()
    assert panel.is_analyzing and not panel.export_button.isEnabled()
    wait_for(app, lambda:not panel.is_analyzing)
    assert panel.result is not None and panel.result.time_origin == 10.5
    assert panel.result.duration == 1.0 and panel.result.source_unit == 'FS'
    assert any(abs(p.order - 3) < .05 for p in panel.result.peaks)
    payload = overlay[-1]
    assert not payload['stale'] and payload['source_hash'] == 'input-hash'
    assert payload['channel'] == 0 and payload['selection_seconds'] == (10.5, 11.5)
    assert 'rpm_times' in payload and 'rpm_values' in payload
    assert panel.findings_table.rowCount() >= 1
    assert panel.shutdown()


def test_rotation_parameter_and_source_changes_make_results_stale(app, tmp_path):
    from autoacoustics.ui.rotation_panel import RotationPanel
    panel = RotationPanel()
    panel.set_signal(rotation_signal())
    panel.start_analysis()
    wait_for(app, lambda:not panel.is_analyzing)
    panel.rpm_spin.setValue(3100)
    assert panel.result_stale and not panel.export_button.isEnabled()
    with pytest.raises(ValueError, match='过期|重新'):
        panel.export_result(tmp_path/'stale.json')
    panel.start_analysis()
    wait_for(app, lambda:not panel.is_analyzing)
    panel.set_signal(rotation_signal(source_hash='replacement'))
    assert panel.result_stale
    assert panel.shutdown()


def test_rotation_source_change_during_work_rejects_late_result(app, monkeypatch):
    import autoacoustics.ui.rotation_panel as module
    original = module.analyze_rotation
    def slower(*args, **kwargs):
        time.sleep(.12)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'analyze_rotation', slower)
    panel = module.RotationPanel()
    panel.set_signal(rotation_signal())
    panel.start_analysis()
    panel.set_signal(rotation_signal(source_hash='next-source'))
    wait_for(app, lambda:not panel.is_analyzing)
    assert panel.result is None and panel.result_stale
    assert not panel.export_button.isEnabled()
    assert panel.shutdown()


def test_rotation_exports_original_parameters_as_json_and_docx(app, tmp_path):
    from autoacoustics.ui.rotation_panel import RotationPanel
    panel = RotationPanel()
    panel.set_signal(rotation_signal(), start_seconds=10.25, end_seconds=12.25)
    panel.gear_teeth_spin.setValue(20)
    panel.start_analysis()
    wait_for(app, lambda:not panel.is_analyzing)
    json_path = panel.export_result(tmp_path/'rotation.json')
    document = panel.export_result(tmp_path/'rotation.docx')
    data = json.loads(json_path.read_text(encoding='utf-8'))
    assert data['source_hash'] == 'input-hash'
    assert 'gear_teeth' in json_path.read_text(encoding='utf-8')
    assert document.is_file() and document.stat().st_size > 1000
    from docx import Document
    exported='\n'.join(p.text for p in Document(document).paragraphs)
    assert '通道索引：0' in exported
    assert panel.shutdown()


def test_measured_order_map_display_retains_absolute_end_times(app):
    from autoacoustics.ui.rotation_panel import RotationPanel
    from autoacoustics.analysis.rotation import analyze_rotation,RotationConfig
    fs=4000;t=np.arange(fs*3)/fs
    rotations=20*t+5*t*t
    result=analyze_rotation(.3*np.sin(2*np.pi*3*rotations),fs,RotationConfig(),
        rpm_times=[10,13],rpm_values=[1200,3000],time_origin=10)
    panel=RotationPanel();panel._show_result(result)
    data,rect,_=panel._plot_data['map']
    dt=(result.map_times[-1]-result.map_times[0])/(len(result.map_times)-1)
    assert rect.right()==pytest.approx(result.map_times[-1]+dt/2)
    assert '显示时间插值' in panel.map_plot.getAxis('bottom').labelText
    assert panel.shutdown()
