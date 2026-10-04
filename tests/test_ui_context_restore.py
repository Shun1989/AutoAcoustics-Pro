"""Restored measurement conditions must agree with the visible workbench."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from dataclasses import asdict, replace

import numpy as np
from PyQt6.QtWidgets import QApplication

from autoacoustics.analysis.pipeline import analyze
from autoacoustics.model import AnalysisSettings, BatchItemSpec, MeasurementContext
from autoacoustics.ui.main_window import MainWindow


def test_history_and_event_restore_show_current_context_without_filename_guess(tmp_path):
    import soundfile as sf
    from autoacoustics.importers import import_signal
    app = QApplication.instance() or QApplication([])
    path = tmp_path / '文件名_CCW.wav'
    sf.write(path, np.zeros(4800), 48000)
    original = import_signal(path)
    first = MeasurementContext(specimen='试件A', test_level='座椅总成',
        actual_movement='座椅向前', mic_position='测点A', notes='记录A')
    second = MeasurementContext(specimen='试件B', test_level='裸电机',
        actual_movement='座椅向后', motor_rotation='CW', rotation_view='从输出轴端观察',
        mic_position='测点B', supply='24 V', load='20 kg', notes='记录B')
    source = replace(original, metadata={'acquisition_session': {
        'measurement_context': asdict(first), 'events': []}})
    settings = AnalysisSettings(channel=0, start=100, end=2000, compute_loudness=False)
    saved = analyze(source, None, settings, second)
    window = MainWindow()
    try:
        window.set_signal(source)
        assert '座椅向前' in window.context_label.text()
        window.results = [saved]
        window.show_history_result(0)
        assert window.context == second
        text = window.context_label.text()
        for value in ('试件B', '裸电机', '座椅向后', 'CW', '从输出轴端观察',
                      '测点B', '24 V', '20 kg', '记录B'):
            assert value in text
        assert '座椅向前' not in text and '记录A' not in text
        item = BatchItemSpec(path, settings, context=first)
        window._restore_item_controls(item)
        assert window.context == first
        assert all(value in window.context_label.text() for value in ('试件A', '座椅向前', '测点A', '记录A'))
        assert '试件B' not in window.context_label.text()
        before = window.context_label.text()
        window.set_signal(original)
        assert window.context == first and window.context_label.text() == before
        assert window.context.motor_rotation == 'unknown'
    finally:
        window.close()
        app.processEvents()
