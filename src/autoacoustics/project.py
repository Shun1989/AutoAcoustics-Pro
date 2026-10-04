"""Versioned manifests with immutable summaries and verified, lazy NPZ details."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np

from .model import (AnalysisDetail, AnalysisResult, AnalysisSettings, AnalysisSummary,
    BatchItemSpec, CalibrationProfile, ImportMapping, MeasurementContext,
    MeasurementProject, MetricResult, ResultSnapshot, ValidationError, to_plain)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_numbers(value):
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            raise ValidationError('项目不能保存 NaN 指标，请保存失败状态。')
        return {'$float': '-inf' if value < 0 else 'inf'}
    if isinstance(value, dict):
        return {k: _safe_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_numbers(v) for v in value]
    return value


def _restore_numbers(value):
    if isinstance(value, dict):
        if set(value) == {'$float'}:
            if value['$float'] not in {'inf', '-inf'}:
                raise ValidationError('项目中的非有限值标记无效。')
            return float(value['$float'])
        return {k: _restore_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_numbers(v) for v in value]
    return value


def write_json_atomic(data, destination: Path):
    destination = Path(destination)
    if destination.suffix.lower() != '.json':
        raise ValidationError('项目和校准配置请保存为 JSON 文件。')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp-' + uuid4().hex)
    try:
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(_safe_numbers(data), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _profile(data):
    return CalibrationProfile(**data) if data else None


def _result(data):
    data = dict(data)
    data['path'] = Path(data['path']) if data['path'] else None
    data['settings'] = AnalysisSettings(**data['settings'])
    data['context'] = MeasurementContext(**data['context'])
    data['calibration'] = _profile(data['calibration'])
    summary = data['summary']
    data['summary'] = AnalysisSummary({k: MetricResult(**v) for k, v in summary['metrics'].items()},
                                      tuple(summary['warnings']))
    data['detail'] = None
    return AnalysisResult(**data)


def save_project(project: MeasurementProject, destination: Path) -> Path:
    destination = Path(destination)
    snapshots = []
    for snapshot in project.results:
        result = snapshot.result
        cache = snapshot.detail_cache
        cache_hash = snapshot.detail_hash
        if result.detail is not None:
            cache_dir = destination.parent / (destination.stem + '.cache')
            cache_dir.mkdir(parents=True, exist_ok=True)
            # A result ID is a content/config fingerprint; never use an input filename as identity.
            cache_path = cache_dir / (hashlib.sha256(result.result_id.encode()).hexdigest() + '.npz')
            temporary = cache_path.with_name(cache_path.name + '.tmp-' + uuid4().hex)
            try:
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, **dict(result.detail.arrays),
                        _units_json=np.asarray(json.dumps(dict(result.detail.units), ensure_ascii=False)))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, cache_path)
            finally:
                temporary.unlink(missing_ok=True)
            cache = str(cache_path.relative_to(destination.parent))
            cache_hash = file_hash(cache_path)
        snapshots.append(ResultSnapshot(replace(result, detail=None), cache, cache_hash))
    data = to_plain(replace(project, results=tuple(snapshots)))
    return write_json_atomic(data, destination)


def load_project(path: Path) -> MeasurementProject:
    path = Path(path)
    try:
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValidationError('项目清单过大；波形应存入详情缓存。')
        data = _restore_numbers(json.loads(path.read_text(encoding='utf-8-sig')))
        if data.get('schema_version') != 1:
            raise ValidationError('不支持此项目版本；不能猜测字段含义。')
        items = []
        for entry in data['items']:
            entry = dict(entry)
            source = Path(entry['path'])
            entry['path'] = source if source.is_absolute() else path.parent / source
            entry['settings'] = AnalysisSettings(**entry['settings'])
            entry['context'] = MeasurementContext(**entry['context'])
            entry['profile'] = _profile(entry['profile'])
            entry['mapping'] = ImportMapping(**entry['mapping']) if entry['mapping'] else None
            items.append(BatchItemSpec(**entry))
        data['items'] = tuple(items)
        data['results'] = tuple(ResultSnapshot(_result(s['result']), s['detail_cache'],
                                              s.get('detail_hash', '')) for s in data['results'])
        project = MeasurementProject(**data)
        states = source_states(project)
        states_by_hash = {item.input_hash: states[item.item_id] for item in project.items}
        snapshots = tuple(replace(s, result=replace(s.result, provenance={**s.result.provenance,
            'source_state': states_by_hash.get(s.result.input_hash, 'unverified')})) for s in project.results)
        return replace(project, results=snapshots)
    except ValidationError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValidationError(f'项目读取失败：{exc}') from exc


def source_states(project: MeasurementProject):
    states = {}
    cache = {}
    for item in project.items:
        states[item.item_id] = _source_state(item.path, item.input_hash, cache)
    return states


def _cached_file_hash(path, cache):
    key = ('sha256', Path(path).resolve())
    if key not in cache:
        cache[key] = file_hash(path)
    return cache[key]


def _session_data_state(path, cache):
    """Validate a DAQ container without loading its recorded arrays."""
    if path.suffix.lower() != '.json':
        return 'ok'
    key = ('session', path.resolve())
    if key in cache:
        return cache[key]
    try:
        # Only declared DAQ documents receive nested-file interpretation.
        # Oversized arbitrary JSON is not guessed to be a recording manifest.
        if path.stat().st_size > 8 * 1024 * 1024:
            return 'ok'
        data = json.loads(path.read_text(encoding='utf-8-sig'))
    except (ValueError, UnicodeError):
        return 'ok'
    except OSError:
        return 'missing' if not path.is_file() else 'changed'
    if not isinstance(data, dict) or not {'schema_version', 'session_id', 'backend', 'status'}.issubset(data):
        cache[key] = 'ok'
        return 'ok'
    from .acquisition.device import AcquisitionError
    from .acquisition.storage import read_manifest, safe_session_file
    data_path = None
    try:
        manifest_path, data = read_manifest(path)
        if data['status'] != 'complete':
            state = 'changed'
        else:
            data_path = safe_session_file(manifest_path, data.get('data_file'))
            if not data_path.is_file():
                state = 'missing'
            else:
                expected = data.get('data_sha256')
                state = 'ok' if isinstance(expected, str) and _cached_file_hash(data_path, cache) == expected.lower() else 'changed'
    except (AcquisitionError, ValueError, TypeError, KeyError):
        state = 'changed'
    except OSError:
        state = 'missing' if data_path is not None and not data_path.is_file() else 'changed'
    cache[key] = state
    return state


def _source_state(path, expected_hash, cache):
    path = Path(path)
    if not path.is_file():
        return 'missing'
    try:
        if expected_hash and _cached_file_hash(path, cache) != expected_hash:
            return 'changed'
        nested = _session_data_state(path, cache)
        if nested != 'ok':
            return nested
        return 'ok' if expected_hash else 'unverified'
    except OSError:
        return 'missing' if not path.is_file() else 'changed'


def relink_source(project: MeasurementProject, item_id: str, path: Path) -> MeasurementProject:
    path = Path(path).resolve()
    original = next((item for item in project.items if item.item_id == item_id), None)
    if original is None:
        raise ValidationError('项目条目不存在。')
    state = _source_state(path, original.input_hash, {})
    if state != 'ok':
        raise ValidationError('重新定位的源文件缺失、哈希不同或采集会话数据完整性校验失败，旧结果不能用于新录音。')
    items = tuple(replace(item, path=path) if item.item_id == item_id else item for item in project.items)
    results = tuple(replace(s, result=replace(s.result, path=path))
        if s.result.input_hash == original.input_hash else s for s in project.results)
    return replace(project, items=items, results=results)


def load_result_detail(snapshot: ResultSnapshot, project_path: Path) -> AnalysisDetail:
    if not snapshot.detail_cache:
        raise ValidationError('项目没有此结果的详情缓存；摘要仍可查看，重新计算须明确操作。')
    folder = Path(project_path).resolve().parent
    cache = (folder / snapshot.detail_cache).resolve()
    if not cache.is_relative_to(folder):
        raise ValidationError('详情缓存路径超出项目目录。')
    try:
        if not snapshot.detail_hash or file_hash(cache) != snapshot.detail_hash:
            raise ValidationError('详情缓存缺失或内容校验失败；不自动重算。')
        with np.load(cache, allow_pickle=False) as data:
            units = json.loads(str(data['_units_json']))
            return AnalysisDetail._from_owned_arrays({k: data[k] for k in data.files if k != '_units_json'}, units)
    except ValidationError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise ValidationError(f'详情缓存读取失败：{exc}') from exc
