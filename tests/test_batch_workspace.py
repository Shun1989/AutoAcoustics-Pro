import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from dataclasses import replace

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLabel, QPlainTextEdit

from autoacoustics.model import (
    AnalysisResult, AnalysisSettings, AnalysisSummary, BatchItemResult,
    BatchItemSpec, BatchResult, ChannelInfo, MetricResult, SignalData,
)
from autoacoustics.ui.batch_panel import BatchItemDialog, BatchPanel


@pytest.fixture
def panel():
    app = QApplication.instance() or QApplication([])
    widget = BatchPanel()
    yield widget
    widget.close()
    app.processEvents()


def specification(path, event='full'):
    return BatchItemSpec(path, AnalysisSettings(channel=0, event_label=event))


def analysis(spec, laeq=61., nmean=2., nmax=3., warnings=()):
    return AnalysisResult(
        f'result-{spec.item_id}', 'source-hash', spec.path, spec.settings,
        spec.context, spec.profile,
        AnalysisSummary({
            'LAeq': MetricResult(laeq, 'dB re 20 µPa'),
            'Nmean': MetricResult(nmean, 'sone'),
            'Nmax': MetricResult(nmax, 'sone'),
        }, warnings),
    )


def count(panel, name):
    label = panel.findChild(QLabel, f'batch_{name}_count')
    assert label is not None, f'Missing visible {name} count'
    return int(label.text())


def test_reordered_results_match_item_identity_for_duplicate_file_events(panel, tmp_path):
    first = specification(tmp_path / 'motor.wav', 'startup')
    second = specification(first.path, 'steady')
    panel.add_item(first)
    panel.add_item(second)
    panel.set_result(BatchResult((
        BatchItemResult(second.item_id, second.path, 'complete', analysis(second, 73., 4., 5.)),
        BatchItemResult(first.item_id, first.path, 'ok', analysis(first)),
    )))

    assert panel.table.columnCount() == 12
    assert panel.table.item(0, 8).text() == '61 dB re 20 µPa · 有效'
    assert panel.table.item(1, 8).text() == '73 dB re 20 µPa · 有效'
    assert panel.table.item(0, 9).text() == '2 sone · 有效'
    assert panel.table.item(1, 10).text() == '5 sone · 有效'
    assert panel.table.item(0, 0).text() == 'motor.wav'
    assert panel.table.item(0, 0).toolTip() == str(first.path)
    assert panel.table.item(0, 0).data(Qt.ItemDataRole.UserRole) == str(first.path)
    assert panel.items()[0].path == first.path
    assert count(panel, 'completed') == 2


def test_fs_metrics_show_missing_calibration_reason_beside_pa_values(panel, tmp_path):
    from autoacoustics.analysis.pipeline import analyze

    pa = specification(tmp_path / 'pa.wav')
    fs = specification(tmp_path / 'digital.wav')
    fs_signal = SignalData(np.ones((1, 2000)) * .01, 48000,
        (ChannelInfo('mic', 'FS'),), path=fs.path, source_hash='fs-source')
    fs_result = analyze(fs_signal, None, fs.settings, include_detail=False)
    panel.add_item(pa)
    panel.add_item(fs)
    panel.set_result(BatchResult((
        BatchItemResult(fs.item_id, fs.path, 'ok', fs_result),
        BatchItemResult(pa.item_id, pa.path, 'ok', analysis(pa)),
    )))

    assert panel.table.columnCount() == 12
    assert panel.table.item(0, 8).text() == '61 dB re 20 µPa · 有效'
    for column in (8, 9, 10):
        text = panel.table.item(1, column).text()
        assert '未校准' in text
        assert not text.startswith('0 ')
    assert '无可用声压换算' in panel.table.item(1, 11).text()
    assert fs_result.detail is None


def test_summary_counts_empty_and_duplicate_paths_as_independent_items(panel, tmp_path):
    assert [count(panel, name) for name in
        ('files', 'items', 'completed', 'partial', 'failed', 'cancelled')] == [0, 0, 0, 0, 0, 0]
    first = specification(tmp_path / 'same.wav', 'startup')
    panel.add_item(first)
    panel.add_item(specification(first.path, 'steady'))
    panel.add_item(specification(tmp_path / 'other.wav'))

    assert count(panel, 'files') == 2
    assert count(panel, 'items') == 3
    assert [count(panel, name) for name in
        ('completed', 'partial', 'failed', 'cancelled')] == [0, 0, 0, 0]


def test_partial_failed_and_cancelled_counts_preserve_completed_results(panel, tmp_path):
    specs = [specification(tmp_path / 'same.wav', str(index)) for index in range(5)]
    for spec in specs:
        panel.add_item(spec)
    partial = replace(analysis(specs[2]), summary=AnalysisSummary({
        'LAeq': MetricResult(65., 'dB re 20 µPa'),
        'Nmean': MetricResult(None, 'sone', 'failed', message='响度内核失败'),
        'Nmax': MetricResult(None, 'sone', 'failed', message='响度内核失败'),
    }))
    unrelated = specification(tmp_path / 'other.wav')
    panel.set_result(BatchResult((
        BatchItemResult(specs[4].item_id, specs[4].path, 'cancelled', error='任务已取消'),
        BatchItemResult(specs[3].item_id, specs[3].path, 'failed', error='文件已损坏'),
        BatchItemResult(specs[2].item_id, specs[2].path, 'partial', partial, '部分指标失败：Nmean, Nmax'),
        BatchItemResult(specs[1].item_id, specs[1].path, 'complete', analysis(specs[1])),
        BatchItemResult(specs[0].item_id, specs[0].path, 'ok', analysis(specs[0])),
        BatchItemResult(unrelated.item_id, unrelated.path, 'failed', error='另一批次'),
    )))

    assert [count(panel, name) for name in
        ('files', 'items', 'completed', 'partial', 'failed', 'cancelled')] == [1, 5, 2, 1, 1, 1]
    assert panel.table.item(0, 8).text() == '61 dB re 20 µPa · 有效'
    assert panel.table.item(2, 8).text() == '65 dB re 20 µPa · 有效'
    assert '失败' in panel.table.item(2, 9).text()
    assert '响度内核失败' in panel.table.item(2, 11).text()
    assert '文件已损坏' in panel.table.item(3, 11).text()
    assert '已取消' in panel.table.item(4, 8).text()
    assert '任务已取消' in panel.table.item(4, 11).text()


def test_edit_refresh_clears_old_values_and_completion_counts(panel, tmp_path, monkeypatch):
    spec = specification(tmp_path / 'edited.wav')
    panel.add_item(spec)
    panel.set_result(BatchResult((BatchItemResult(spec.item_id, spec.path, 'ok', analysis(spec)),)))
    assert panel.table.columnCount() == 12
    assert count(panel, 'completed') == 1

    def accept_edit(dialog):
        dialog.event.setText('startup')
        return 1

    monkeypatch.setattr(BatchItemDialog, 'exec', accept_edit)
    panel.table.setCurrentCell(0, 4)
    panel.edit_current()

    assert panel.items()[0].settings.event_label == 'startup'
    assert panel.items()[0].item_id == spec.item_id
    assert panel.result is None
    assert count(panel, 'completed') == 0
    assert count(panel, 'items') == 1
    assert '已完成' not in panel.table.item(0, 7).text()
    for column in (8, 9, 10):
        assert '有效' not in panel.table.item(0, column).text()
        assert not panel.table.item(0, column).flags() & Qt.ItemFlag.ItemIsEditable
    assert not panel.table.item(0, 11).flags() & Qt.ItemFlag.ItemIsEditable


def test_selecting_row_reads_compact_metrics_without_analysis_detail(panel, tmp_path):
    first = specification(tmp_path / 'same.wav', 'startup')
    second = specification(first.path, 'steady')
    panel.add_item(first)
    panel.add_item(second)
    first_result = analysis(first)
    second_result = analysis(second, 73., 4., 5., warnings=('选段前史不足',))
    panel.set_result(BatchResult((
        BatchItemResult(second.item_id, second.path, 'ok', second_result),
        BatchItemResult(first.item_id, first.path, 'ok', first_result),
    )))
    readout = panel.findChild(QPlainTextEdit, 'batch_result_readout')
    assert readout is not None, 'Selected results must be readable without scrolling to result columns'
    panel.table.setCurrentCell(0, 4)
    assert 'startup' in readout.toPlainText()
    assert '61 dB re 20 µPa' in readout.toPlainText()
    panel.table.setCurrentCell(1, 4)
    assert 'steady' in readout.toPlainText()
    assert '73 dB re 20 µPa' in readout.toPlainText()
    assert '5 sone' in readout.toPlainText()
    assert '选段前史不足' in readout.toPlainText()
    assert readout.isReadOnly()
    assert first_result.detail is None and second_result.detail is None
    panel.refresh()
    assert '73 dB re 20 µPa' not in readout.toPlainText()
