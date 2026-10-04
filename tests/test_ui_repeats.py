"""Behavioral checks for the condition filter and original result identity."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from dataclasses import fields, replace
from pathlib import Path
import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QComboBox

from autoacoustics.model import (AnalysisResult, AnalysisSettings, AnalysisSummary,
    MeasurementContext, MetricResult)
from autoacoustics.ui.main_window import MainWindow


def recorded_result(index, **context_changes):
    context = MeasurementContext(**{f.name: f'known-{f.name}' for f in fields(MeasurementContext)
        if f.name not in {'repeat', 'notes'}})
    context = replace(context, repeat=str(index), notes=f'record {index}', **context_changes)
    return AnalysisResult(f'result-{index}', f'hash-{index}', Path(f'record-{index}.wav'),
        AnalysisSettings(channel=0, start=10*index, end=1000+index, event_label='running'),
        context, None, AnalysisSummary({'RMS': MetricResult(.1, 'Pa')}), None,
        {'sample_rate':48000, 'channel_name':'Microphone', 'source_unit':'Pa', 'unit':'Pa',
         'method_revision':'test-method', 'versions':{'numpy':'test'},
         'calibration':{'source':'文件声明 Pa', 'coefficient':1., 'input_unit':'Pa'}})


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    yield win
    win.close()
    app.processEvents()


def enable_filter(win):
    combo = win.findChild(QComboBox, 'repeat_condition_filter')
    assert combo is not None, 'The explicit condition filter is missing.'
    combo.setCurrentIndex(combo.findData('matching'))
    return combo


def test_condition_filter_excludes_different_load_and_selects_original_result(window):
    first = recorded_result(1)
    different = recorded_result(2, load='different-load')
    third = recorded_result(3)
    window.results = [first, different, third]
    window.display_result(first)
    window._refresh_history()
    enable_filter(window)
    assert window.history_list.count() == 2
    assert [window.history_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(2)] == [0, 2]
    window.history_list.setCurrentRow(1)
    assert window.current_result.result_id == 'result-3'
    assert window.left_combo.count() == window.right_combo.count() == 3
    window.right_combo.setCurrentIndex(1)
    assert window._result_from_combo(window.right_combo).result_id == 'result-2'


def test_filter_keeps_all_history_available_and_reports_unknown_conditions(window):
    first = replace(recorded_result(1), context=MeasurementContext())
    second = replace(recorded_result(2), context=MeasurementContext(repeat='2'))
    window.results = [first, second, recorded_result(3)]
    window.display_result(first)
    window._refresh_history()
    combo = enable_filter(window)
    assert window.history_list.count() == 2
    assert '仅记录匹配' in window.repeat_filter_status.text()
    assert '重复试验未确认' in window.repeat_filter_status.text()
    assert 'supply' in window.repeat_filter_status.toolTip()
    combo.setCurrentIndex(combo.findData('all'))
    assert window.history_list.count() == 3


def test_legacy_missing_calibration_record_is_displayed_as_unknown(window):
    first = recorded_result(1)
    first = replace(first, provenance={**first.provenance, 'calibration':None})
    window.results = [first]
    window.display_result(first)
    window._refresh_history()
    enable_filter(window)
    assert window.history_list.count() == 1
    assert '仅记录匹配' in window.repeat_filter_status.text()
    assert 'calibration.record' in window.repeat_filter_status.toolTip()


def test_filter_recomputes_against_current_result_without_losing_combo_selection(window):
    first = recorded_result(1)
    different = recorded_result(2, motor_rotation='different-direction')
    window.results = [first, different]
    window.display_result(first)
    window._refresh_history()
    enable_filter(window)
    window.left_combo.setCurrentIndex(1)
    window.display_result(different)
    assert window.history_list.count() == 1
    assert window.history_list.item(0).data(Qt.ItemDataRole.UserRole) == 1
    assert window._result_from_combo(window.left_combo).result_id == 'result-2'


def test_saved_context_supports_the_same_filter_after_project_reopen(window, tmp_path):
    first = recorded_result(1)
    different = recorded_result(2, fixture='different-fixture')
    third = recorded_result(3)
    window.results = [first, different, third]
    window.display_result(first)
    window._refresh_history()
    enable_filter(window)
    project_path = tmp_path / '条件筛选项目.json'
    window.save_project_path(project_path)
    window.open_project_path(project_path)
    # The last saved result is the reference after reopening, and its context
    # still matches first, while the different fixture is excluded.
    assert window.current_result.result_id == 'result-3'
    assert window.history_list.count() == 2
    assert [window.history_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(2)] == [0, 2]


def test_opening_empty_project_drops_the_previous_reference_and_report_selection(window, tmp_path):
    from autoacoustics.analysis.comparison import compare_results
    from autoacoustics.model import MeasurementProject
    from autoacoustics.project import save_project
    first, second = recorded_result(1), recorded_result(2)
    window.results = [first, second]
    window.display_result(first)
    window._refresh_history()
    enable_filter(window)
    window._history_selected = True
    window.comparison = compare_results(first, second)
    window._pending_project_result = first
    empty = tmp_path / '空项目.json'
    save_project(MeasurementProject('Empty'), empty)
    window.open_project_path(empty)
    assert window.current_result is None and not window.current_result_valid
    assert not window._history_selected and window.comparison is None
    assert window._pending_project_result is None
    assert window.history_list.count() == window.metric_table.rowCount() == 0
    assert '尚无当前结果' in window.repeat_filter_status.text()
