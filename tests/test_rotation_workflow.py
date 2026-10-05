"""NVH export ownership and close guards, without reproducing a native crash."""
from pathlib import Path
import json
import time

import numpy as np
import pytest
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication, QPushButton

from autoacoustics.analysis.rotation import RotationConfig, analyze_rotation
from autoacoustics.model import ChannelInfo, SignalData
from autoacoustics.ui.main_window import MainWindow
from autoacoustics.ui.rotation_panel import RotationPanel, _RotationTask as NativeRotationTask


class ControlledExport(QObject):
    """Keep Qt signal delivery, but never delete a running native QThread."""
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function
        self.running = False
        self.delete_requested = False

    def start(self):
        self.running = True

    def complete(self):
        self.running = False
        self.finished.emit()

    def deleteLater(self):
        # Inspect wrong-owner deletion instead of triggering Windows abort.
        self.delete_requested = True


@pytest.fixture
def export_panel(monkeypatch):
    import autoacoustics.ui.rotation_panel as module
    panel = RotationPanel()
    sample_rate = 6000
    time = np.arange(sample_rate) / sample_rate
    signal = SignalData(np.sin(2*np.pi*50*time)[None], sample_rate,
                        (ChannelInfo('Mic', 'FS'),), Path('recording.wav'), 'source-sha')
    panel.set_signal(signal)
    result = analyze_rotation(signal.samples[0], sample_rate, RotationConfig())
    snapshot = {'generation': panel._generation, 'source_path': str(signal.path),
                'source_hash': signal.source_hash, 'channel': 0,
                'selection_seconds': (0., 1.), 'source_unit': 'FS',
                'sample_rate': sample_rate, 'config': result.config,
                'rpm_mode': 'manual', 'rpm_csv_path': ''}
    panel._analysis_completed((result, snapshot))
    panel._worker_finished()
    monkeypatch.setattr(module, '_RotationTask', ControlledExport)
    monkeypatch.setattr(module.QFileDialog, 'getSaveFileName',
                        lambda *args, **kwargs: ('not-written.docx', ''))
    yield panel
    # Test doubles own no native work. Release retained state for GUI teardown.
    for worker in panel.findChildren(ControlledExport):
        worker.running = False
    panel._export_worker = None


def export_actions(panel):
    return [button for button in panel.findChildren(QPushButton)
            if button.text().startswith('导出诊断报告')]


def test_both_export_actions_follow_current_result_and_worker_state(export_panel):
    panel = export_panel
    assert len(export_actions(panel)) == 2
    assert all(button.isEnabled() for button in export_actions(panel))
    panel._choose_export()
    assert not any(button.isEnabled() for button in export_actions(panel))
    panel._export_worker.complete()
    assert all(button.isEnabled() for button in export_actions(panel))
    panel.rpm_spin.setValue(3100)
    assert not any(button.isEnabled() for button in export_actions(panel))


def test_export_entry_refuses_a_second_job_and_preserves_original_worker(export_panel):
    panel = export_panel
    panel._choose_export()
    original = panel._export_worker
    panel._choose_export()
    assert panel._export_worker is original
    assert len(panel.findChildren(ControlledExport)) == 1
    with pytest.raises(ValueError, match='导出'):
        panel.export_result('not-written.json')
    assert not panel.shutdown()
    original.complete()
    assert panel._export_worker is None
    assert panel.shutdown()


def test_late_finished_signal_only_releases_its_own_export(export_panel):
    panel = export_panel
    panel._choose_export()
    first = panel._export_worker
    first.complete()
    panel._choose_export()
    current = panel._export_worker
    # Simulate an old completion that was queued before a new export started.
    first.finished.emit()
    assert panel._export_worker is current
    assert current.running and not current.delete_requested
    assert not panel.shutdown()
    current.complete()
    assert current.delete_requested and panel.shutdown()


def test_close_during_save_dialog_does_not_start_an_export(export_panel, monkeypatch):
    import autoacoustics.ui.rotation_panel as module
    panel = export_panel

    def close_while_choosing(*args, **kwargs):
        assert panel.shutdown()
        return 'not-written.docx', ''

    monkeypatch.setattr(module.QFileDialog, 'getSaveFileName', close_while_choosing)
    panel._choose_export()
    assert panel._export_worker is None
    assert not panel.findChildren(ControlledExport)


def test_main_window_waits_for_export_then_rejects_new_jobs_during_close(export_panel):
    panel = export_panel
    window = MainWindow()
    panel.setParent(window)
    window.rotation_panel = panel
    window.show()
    QApplication.instance().processEvents()
    panel._choose_export()
    worker = panel._export_worker
    assert window.close() is False
    assert window.isVisible() and worker.running
    panel._choose_export()
    assert panel._export_worker is worker
    worker.complete()
    assert not any(button.isEnabled() for button in export_actions(panel))
    assert window.close()


def test_native_export_finishes_saved_snapshot_before_shutdown(export_panel, monkeypatch, tmp_path):
    import autoacoustics.ui.rotation_panel as module
    panel = export_panel
    destination = tmp_path / 'rotation.json'
    monkeypatch.setattr(module, '_RotationTask', NativeRotationTask)
    monkeypatch.setattr(module.QFileDialog, 'getSaveFileName',
                        lambda *args, **kwargs: (str(destination), ''))
    panel._choose_export()
    worker = panel._export_worker
    try:
        assert not panel.shutdown()
        deadline = time.monotonic() + 5
        while panel._export_worker is not None and time.monotonic() < deadline:
            QApplication.instance().processEvents()
            time.sleep(.005)
        assert panel._export_worker is None
        payload = json.loads(destination.read_text(encoding='utf-8'))
        assert payload['source_hash'] == 'source-sha'
        assert payload['config']['rpm'] == 3000
        assert payload['ui_source']['selection_seconds'] == [0, 1]
        assert not any(button.isEnabled() for button in export_actions(panel))
        assert panel.shutdown()
    finally:
        # Do not destroy a real worker on an assertion failure or slow machine.
        worker.wait(10000)
        QApplication.instance().processEvents()
