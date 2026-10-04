"""Independent analysis items, compact summaries, and atomic UTF-8 CSV export."""
from __future__ import annotations

import csv
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import tempfile

from .importers import import_signal
from .model import (BatchItemSpec, BatchItemResult, BatchResult, CancelledError,
                    MetricResult, ValidationError, to_plain)

METRIC_ORDER = ('LZeq', 'LAeq', 'LAFmax', 'LASmax', 'Nmean', 'Nmax',
                'Nstationary', 'RMS', 'dBFS', 'duration')
INVALID_METRIC_STATUSES = {'failed', 'uncalibrated', 'not_applicable', 'cancelled',
                           'disabled', 'unavailable', 'unknown'}


def analyze(*args, **kwargs):
    # Keep standalone import/export usable while the analysis module loads.
    from .analysis.pipeline import analyze as implementation
    return implementation(*args, **kwargs)


def is_cancelled(cancel):
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    if hasattr(cancel, 'cancelled'):
        return bool(cancel.cancelled)
    return bool(cancel.is_set())


def check_cancel(cancel):
    if is_cancelled(cancel):
        raise CancelledError('任务已取消。')


def _row(item, status, result=None, error=''):
    return BatchItemResult(item.item_id, item.path, status, result, error,
                           specification=item)


def run_batch(items, on_progress=None, cancel=None) -> BatchResult:
    """Analyze each original item once; same paths are deliberately retained.

    Failed rows preserve their input specification, never a fake analysis.
    Cancellation preserves completed rows and marks every remaining row.
    """
    specifications = tuple(items)
    if any(not isinstance(item, BatchItemSpec) for item in specifications):
        raise ValidationError('批处理每一行必须为明确的 BatchItemSpec。')
    identities = [item.item_id for item in specifications]
    if len(set(identities)) != len(identities) or any(not value for value in identities):
        raise ValidationError('批处理条目 ID 必须非空且唯一；重复文件可使用不同条目 ID。')
    total = len(specifications)
    results = []
    cancel_seen = False
    if on_progress:
        on_progress(0, total, 'running')
    for completed, item in enumerate(specifications, 1):
        if cancel_seen or is_cancelled(cancel):
            row = _row(item, 'cancelled', error='尚未完成，任务已取消。')
        else:
            try:
                signal = import_signal(item.path, item.mapping)
                check_cancel(cancel)
                if item.input_hash and signal.source_hash != item.input_hash:
                    raise ValidationError('源文件哈希与批处理配置不一致；请重新核对文件。')
                item.settings.resolve(signal)
                result = analyze(signal, item.profile, item.settings, item.context,
                                 include_detail=False, cancel=cancel)
                check_cancel(cancel)
                # Defensively release details even if a backend ignored the
                # compact-output flag. No waveform arrays belong in a batch.
                result = replace(result, detail=None)
                failed=[name for name,metric in result.summary.metrics.items() if metric.status=='failed']
                row = _row(item,'partial' if failed else 'ok',result,
                    error='部分指标失败：'+', '.join(failed) if failed else '')
                del signal, result
            except (CancelledError, InterruptedError) as error:
                cancel_seen = True
                row = _row(item, 'cancelled', error=str(error) or '任务已取消。')
            except Exception as error:
                status = 'cancelled' if is_cancelled(cancel) else 'failed'
                row = _row(item, status, error=f'{type(error).__name__}：{error}')
            finally:
                signal = result = None
        results.append(row)
        if on_progress:
            on_progress(completed, total, row.status)
    return BatchResult(tuple(results))


def ordered_metrics(keys):
    keys = set(keys)
    return [key for key in METRIC_ORDER if key in keys] + sorted(keys - set(METRIC_ORDER))


def valid_metric_value(metric: MetricResult):
    """Preserve physical silence (-inf), never export invalid metrics as zero."""
    if metric.value is None or metric.status in INVALID_METRIC_STATUSES:
        return None
    value = float(metric.value)
    if math.isnan(value) or value == math.inf:
        return None
    return value


def _json(value):
    return json.dumps(to_plain(value), ensure_ascii=False, sort_keys=True)


def export_csv(batch: BatchResult, destination: Path, *, cancel=None) -> Path:
    """Write one row per analysis item, including failed/cancelled configs.

    Stable English headers support scripts; units, methods, contextual Chinese
    values, and errors remain explicit. Invalid metric values are empty fields.
    """
    if not isinstance(batch, BatchResult):
        raise ValidationError('CSV 导出仅接受批处理结果。')
    check_cancel(cancel)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metric_names = ordered_metrics(key for row in batch.items if row.result
                                   for key in row.result.summary.metrics)
    headers = ['item_id', 'status', 'error', 'path', 'input_hash', 'result_id',
               'channel', 'start_sample', 'end_sample_exclusive',
               'event_label', 'source_unit', 'calibration_source', 'settings_json',
               'context_json', 'calibration_json', 'provenance_json']
    for name in metric_names:
        headers += [f'{name}.{field}' for field in ('value', 'unit', 'status', 'method', 'message')]
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{destination.stem}.',
                                                  suffix='.tmp.csv', dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, headers)
            writer.writeheader()
            for row in batch.items:
                check_cancel(cancel)
                result = row.result
                specification = row.specification
                settings = result.settings if result else specification.settings if specification else None
                context = result.context if result else specification.context if specification else None
                profile = result.calibration if result else specification.profile if specification else None
                provenance = result.provenance if result else {}
                end = settings.end if settings else None
                if end is None and result:
                    end = provenance.get('event_end_sample', provenance.get('end_sample',
                                         provenance.get('source_frames', provenance.get('frames'))))
                fields = {
                    'item_id': row.item_id, 'status': row.status, 'error': row.error,
                    'path': str(row.path), 'input_hash': result.input_hash if result else specification.input_hash if specification else '',
                    'result_id': result.result_id if result else '',
                    'channel': settings.channel if settings else '',
                    'start_sample': settings.start if settings else '',
                    'end_sample_exclusive': end if end is not None else '',
                    'event_label': settings.event_label if settings else '',
                    'source_unit': provenance.get('source_unit',provenance.get('input_unit',
                                    specification.mapping.unit if specification and specification.mapping else 'unknown')),
                    'calibration_source': profile.source if profile else provenance.get('calibration_source',
                                           provenance.get('calibration',{}).get('source','')),
                    'settings_json': _json(settings) if settings else '',
                    'context_json': _json(context) if context else '',
                    'calibration_json': _json(profile) if profile else '',
                    'provenance_json': _json(provenance),
                }
                if result:
                    for name, metric in result.summary.metrics.items():
                        value = valid_metric_value(metric)
                        fields.update({
                            f'{name}.value': format(value, '.17g') if value is not None else '',
                            f'{name}.unit': metric.unit, f'{name}.status': metric.status,
                            f'{name}.method': metric.method, f'{name}.message': metric.message,
                        })
                writer.writerow(fields)
            stream.flush()
            os.fsync(stream.fileno())
        check_cancel(cancel)
        temporary.replace(destination)
        return destination
    finally:
        if temporary.exists():
            temporary.unlink()
