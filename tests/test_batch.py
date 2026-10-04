from dataclasses import replace
from pathlib import Path
import csv

import numpy as np
import pytest

from autoacoustics.model import (AnalysisDetail, AnalysisResult, AnalysisSettings,
    AnalysisSummary, BatchItemSpec, CancellationToken, ChannelInfo,
    MeasurementContext, MetricResult, SignalData, ValidationError)
from autoacoustics import batch


def setup_engine(monkeypatch, failure_channel=None):
    signal = SignalData(np.ones((2, 30)), 48000,
                        (ChannelInfo('左', 'Pa'), ChannelInfo('右', 'Pa')),
                        source_hash='current-hash')
    calls = []
    monkeypatch.setattr(batch, 'import_signal', lambda path, mapping=None: signal)

    def fake_analyze(signal, profile, settings, context=None, *, include_detail=True, cancel=None):
        settings.resolve(signal)
        calls.append((settings.channel, settings.start, settings.end, include_detail))
        if settings.channel == failure_channel:
            raise ValueError('指定通道失败')
        return AnalysisResult(f'result-{len(calls)}', signal.source_hash, Path('same.wav'),
            settings, context, profile,
            AnalysisSummary({'LZeq': MetricResult(81.25, 'dB', method='test method')}),
            AnalysisDetail({'waveform': np.ones(30)}) if include_detail else None,
            {'sample_rate': 48000, 'input_unit': 'Pa', 'algorithm_version': 'test'})
    monkeypatch.setattr(batch, 'analyze', fake_analyze)
    return calls

def test_component_failure_is_partial_row_with_successful_metrics_retained(monkeypatch):
    setup_engine(monkeypatch)
    original=batch.analyze
    def partial(*args,**kwargs):
        result=original(*args,**kwargs)
        return replace(result,summary=AnalysisSummary({**result.summary.metrics,
            'Nmax':MetricResult(None,'sone','failed',message='kernel error')}))
    monkeypatch.setattr(batch,'analyze',partial)
    row=batch.run_batch((BatchItemSpec(Path('same.wav'),AnalysisSettings(channel=0)),)).items[0]
    assert row.status=='partial'
    assert row.result.summary.metrics['LZeq'].value==81.25
    assert 'Nmax' in row.error and row.specification is not None


def test_same_file_two_channels_three_events_are_six_independent_results(monkeypatch):
    calls = setup_engine(monkeypatch)
    items = [BatchItemSpec(Path('same.wav'), AnalysisSettings(channel=channel, start=start,
             end=start + 10, event_label=f'事件{start}')) for channel in (0, 1) for start in (0, 10, 20)]
    progress = []
    result = batch.run_batch(items, lambda *values: progress.append(values))
    assert len(result.items) == 6
    assert len({row.item_id for row in result.items}) == 6
    assert len({row.result.result_id for row in result.items}) == 6
    assert all(row.status == 'ok' and row.result.detail is None for row in result.items)
    assert all(not call[3] for call in calls)
    assert progress[-1][:2] == (6, 6)


def test_invalid_channel_event_and_changed_hash_fail_per_row_without_default(monkeypatch):
    setup_engine(monkeypatch)
    items = [BatchItemSpec(Path('a.wav'), AnalysisSettings(channel=0, end=20)),
             BatchItemSpec(Path('a.wav'), AnalysisSettings(channel=5, end=20)),
             BatchItemSpec(Path('a.wav'), AnalysisSettings(channel=0, end=31)),
             BatchItemSpec(Path('a.wav'), AnalysisSettings(channel=0), input_hash='old-hash'),
             BatchItemSpec(Path('a.wav'), AnalysisSettings(channel=1, end=30))]
    result = batch.run_batch(items)
    assert [row.status for row in result.items] == ['ok', 'failed', 'failed', 'failed', 'ok']
    assert result.items[1].error and result.items[2].error
    assert '哈希' in result.items[3].error


def test_cancellation_keeps_completed_rows_and_marks_remaining(monkeypatch):
    setup_engine(monkeypatch)
    token = CancellationToken()
    items = [BatchItemSpec(Path('a.wav'), AnalysisSettings(channel=0)) for _ in range(4)]

    def progress(completed, total, status):
        if completed == 1:
            token.cancel()

    result = batch.run_batch(items, progress, token)
    assert [row.status for row in result.items] == ['ok', 'cancelled', 'cancelled', 'cancelled']
    assert all(row.result is None for row in result.items[1:])


def test_csv_bom_configuration_provenance_and_metric_status(monkeypatch, tmp_path):
    setup_engine(monkeypatch, failure_channel=1)
    context = MeasurementContext(specimen='座椅电机', supply='12 V', actual_movement='升高')
    items = [BatchItemSpec(Path('中文.wav'), AnalysisSettings(channel=0, end=20), context=context),
             BatchItemSpec(Path('中文.wav'), AnalysisSettings(channel=1, start=10, end=20), context=context)]
    result = batch.run_batch(items)
    destination = batch.export_csv(result, tmp_path / '批量 结果.csv')
    assert destination.read_bytes().startswith(b'\xef\xbb\xbf')
    with destination.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]['LZeq.value'] == '81.25'
    assert rows[1]['LZeq.value'] == ''
    assert rows[1]['channel'] == '1'
    assert rows[1]['start_sample'] == '10'
    assert '座椅电机' in rows[1]['context_json'] and '12 V' in rows[1]['context_json']
    assert rows[1]['status'] == 'failed'


def test_duplicate_identity_is_rejected_and_cancelled_export_preserves_file(tmp_path):
    item = BatchItemSpec(Path('a.wav'), AnalysisSettings(channel=0))
    with pytest.raises(ValidationError):
        batch.run_batch([item, item])
    target = tmp_path / 'old.csv'
    target.write_bytes(b'previous valid report')
    token = CancellationToken(); token.cancel()
    with pytest.raises(Exception, match='取消'):
        batch.export_csv(batch.run_batch([]), target, cancel=token)
    assert target.read_bytes() == b'previous valid report'

