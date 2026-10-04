"""Spawn isolated file-based compute jobs; Qt polls small status messages only."""
from __future__ import annotations
import copy
import multiprocessing as mp
import os
import pickle
import queue
import shutil
import tempfile
import time
from dataclasses import dataclass,replace,is_dataclass,fields
from pathlib import Path
from uuid import uuid4

from .model import BatchItemResult, BatchResult, CancelledError, FrozenMap, MappingRequired, ValidationError, to_plain

def _snapshot_request(value):
    """Copy mutable controls, reuse immutable model snapshots and their buffers."""
    if isinstance(value,FrozenMap):
        return value
    if is_dataclass(value) and type(value).__module__ == 'autoacoustics.model':
        return value
    if isinstance(value,dict):
        return {key:_snapshot_request(item) for key,item in value.items()}
    if isinstance(value,list):
        return [_snapshot_request(item) for item in value]
    if isinstance(value,tuple):
        return tuple(_snapshot_request(item) for item in value)
    if is_dataclass(value):
        return replace(value,**{f.name:_snapshot_request(getattr(value,f.name)) for f in fields(value) if f.init})
    return copy.deepcopy(value)

@dataclass(frozen=True)
class JobEvent:
    job_id: str
    kind: str
    state: str
    message: str = ''
    completed: int = 0
    total: int = 0
    result: object = None

class ProcessCancellation:
    def __init__(self, event):
        self.event = event
    @property
    def cancelled(self):
        return self.event.is_set()
    def check(self):
        if self.cancelled:
            raise CancelledError('任务已取消。')

def _write_result(result, path):
    target = Path(path)
    temporary = target.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)

def _read_result(path):
    # The process-owned file is trusted IPC. Streaming avoids allocating an
    # additional bytes object as large as every detail array combined.
    with Path(path).open('rb') as stream:
        return pickle.load(stream)

def _worker(job_id, kind, payload, events, cancelled, directory):
    token = ProcessCancellation(cancelled)
    directory = Path(directory)
    try:
        token.check()
        events.put(('running', '处理中', 0, 0))
        if kind == 'callable':
            function, arguments, keywords = payload
            result = function(*arguments, **keywords)
        elif kind == 'import':
            from .importers import import_signal
            result = import_signal(payload['path'], payload.get('mapping'))
        elif kind == 'analysis':
            from .importers import import_signal
            from .analysis.pipeline import analyze
            signal = import_signal(payload['path'], payload.get('mapping'))
            token.check()
            if payload.get('input_hash') and signal.source_hash != payload['input_hash']:
                raise ValidationError('源文件在确认后发生变化，请重新导入并检查通道、事件和单位。')
            result = analyze(signal, payload.get('profile'), payload['settings'],
                             payload.get('context'), include_detail=True, cancel=token)
        elif kind == 'calibration':
            from .importers import import_signal
            from .calibration import make_calibrator_profile
            signal=import_signal(payload['path'],payload.get('mapping'))
            token.check()
            if payload.get('input_hash') and signal.source_hash!=payload['input_hash']:
                raise ValidationError('校准录音在确认后发生变化，请重新导入。')
            result=make_calibrator_profile(signal,**payload['parameters'])
        elif kind == 'batch':
            from .batch import run_batch
            items = payload['items']
            results = []
            for index, item in enumerate(items):
                token.check()
                current = run_batch((item,), lambda *args: None, token)
                results.extend(current.items)
                _write_result(BatchResult(tuple(results)), directory / 'checkpoint.pkl')
                events.put(('progress', str(item.path.name), index + 1, len(items)))
            result = BatchResult(tuple(results))
        elif kind == 'report':
            from .report import write_docx,ReportSpec
            provided=payload['result']
            spec=replace(provided,cancel=token) if isinstance(provided,ReportSpec) else ReportSpec(provided,
                project_name=payload.get('project_name',''),project_version=payload.get('project_version'),cancel=token)
            result = write_docx(spec, payload['destination'])
        elif kind == 'export_csv':
            from .batch import export_csv
            result = export_csv(payload['result'], payload['destination'],cancel=token)
        else:
            raise ValueError('不支持此计算任务。')
        token.check()
        _write_result(result, directory / 'result.pkl')
        events.put(('complete', '已完成', 1, 1))
    except CancelledError:
        events.put(('cancelled', '已取消，保留已完成条目', 0, 0))
    except Exception as error:
        if isinstance(error, MappingRequired):
            _write_result({'mapping_options':to_plain(error.options)},directory/'error.pkl')
        events.put(('failed', f'{type(error).__name__}: {error}', 0, 0))

class JobExecutor:
    """Owns compute processes only. DAQ writers are never submitted here."""
    def __init__(self, directory=None):
        self._owns_root = directory is None
        self.root = Path(directory or tempfile.mkdtemp(prefix='autoacoustics-jobs-')).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.context = mp.get_context('spawn')
        self.jobs = {}
        self.active_id = None

    @property
    def running(self):
        return any(not record['finished'] for record in self.jobs.values())

    def submit(self, kind, payload):
        if kind not in {'import', 'analysis', 'calibration', 'batch', 'report', 'export_csv', 'callable'}:
            raise ValueError('未知任务类型。')
        if self.active_id and not self.jobs[self.active_id]['finished']:
            self.cancel()
            self._terminate(self.jobs[self.active_id])
        job_id = uuid4().hex
        directory = self.root / job_id
        directory.mkdir()
        # Serialize now: a later widget/config mutation cannot change the request.
        snapshot = _snapshot_request(payload)
        events = self.context.Queue()
        cancelled = self.context.Event()
        process = self.context.Process(target=_worker,
            args=(job_id, kind, snapshot, events, cancelled, str(directory)), daemon=False)
        record = dict(kind=kind, payload=snapshot, process=process, queue=events,
                      cancelled=cancelled, directory=directory, deadline=None, finished=False)
        self.jobs[job_id] = record
        self.active_id = job_id
        process.start()
        return job_id

    def submit_callable(self, function, *arguments, **keywords):
        return self.submit('callable', (function, arguments, keywords))

    def cancel(self):
        if not self.active_id:
            return
        record = self.jobs[self.active_id]
        if record['finished']:
            return
        record['cancelled'].set()
        record['deadline'] = time.monotonic() + .35

    def _terminate(self, record):
        process = record['process']
        if process.is_alive():
            # These children belong solely to this compute worker (e.g. ffmpeg).
            import psutil
            try:
                children = psutil.Process(process.pid).children(recursive=True)
            except psutil.Error:
                children = []
            for child in reversed(children):
                try:
                    child.kill()
                except psutil.Error:
                    pass
            process.terminate()
            process.join(timeout=.25)
            if process.is_alive():
                process.kill()
                process.join(timeout=.25)

    def _partial(self, record):
        checkpoint = record['directory'] / 'checkpoint.pkl'
        result = _read_result(checkpoint) if checkpoint.exists() else BatchResult(())
        if record['kind'] != 'batch':
            return None
        seen = {item.item_id for item in result.items}
        pending = tuple(BatchItemResult(item.item_id, item.path, 'cancelled', error='任务已取消', specification=item)
            for item in record['payload']['items'] if item.item_id not in seen)
        return BatchResult(result.items + pending)

    def poll(self):
        output = []
        for job_id, record in self.jobs.items():
            if record['finished']:
                continue
            if record['deadline'] and time.monotonic() >= record['deadline']:
                self._terminate(record)
            terminal = None
            while True:
                try:
                    state, message, done, total = record['queue'].get_nowait()
                except (queue.Empty, OSError, EOFError):
                    break
                if state in {'complete', 'failed', 'cancelled'}:
                    terminal = (state, message, done, total)
                elif job_id == self.active_id:
                    output.append(JobEvent(job_id, record['kind'], state, message, done, total))
            if record['cancelled'].is_set() and not record['process'].is_alive():
                terminal = ('cancelled', '已取消，保留已完成条目', 0, 0)
            elif not record['process'].is_alive() and terminal is None:
                # Queue feeder may finish slightly after process exit; defer one poll.
                if record.get('exited_at') is None:
                    record['exited_at'] = time.monotonic()
                elif time.monotonic() - record['exited_at'] > .3:
                    terminal = ('failed', f'工作进程异常退出 ({record["process"].exitcode})', 0, 0)
            if terminal:
                state, message, done, total = terminal
                if record['cancelled'].is_set():
                    state,message='cancelled','已取消，保留已完成条目'
                record['finished'] = True
                record['process'].join(timeout=.1)
                result = None
                if state == 'complete' and job_id == self.active_id and not record['cancelled'].is_set():
                    result = _read_result(record['directory'] / 'result.pkl')
                elif state == 'complete':
                    state, message = 'stale', '已丢弃过期任务结果'
                elif state == 'cancelled':
                    result = self._partial(record)
                elif state == 'failed' and (record['directory']/'error.pkl').exists():
                    result=_read_result(record['directory']/'error.pkl')
                record['terminal_state']=state
                record['payload']=None
                # Results/checkpoints have now been consumed. The private IPC
                # directory is not a durable project cache.
                directory=record['directory'].resolve()
                if directory.is_relative_to(self.root) and directory!=self.root:
                    shutil.rmtree(directory,ignore_errors=True)
                record['queue'].close()
                output.append(JobEvent(job_id, record['kind'], state, message, done, total, result))
        return output

    def close(self):
        for record in self.jobs.values():
            record['cancelled'].set()
            self._terminate(record)
            record['queue'].close()
            directory = record['directory'].resolve()
            if directory.is_relative_to(self.root) and directory != self.root:
                shutil.rmtree(directory, ignore_errors=True)
        self.jobs.clear()
        if self._owns_root and self.root.exists() and self.root.name.startswith('autoacoustics-jobs-'):
            self.root.rmdir()
