import importlib
import time
from pathlib import Path

def echo_job(value):
    return {'echo': value}

def test_request_snapshot_copies_mutable_controls_but_shares_readonly_results():
    import numpy as np
    from autoacoustics.model import AnalysisDetail
    from autoacoustics.jobs import _snapshot_request
    detail=AnalysisDetail({'waveform':np.arange(20.)},{'waveform':'Pa'})
    controls={'names':['first'],'result':detail}
    snapshot=_snapshot_request(controls)
    controls['names'].append('later')
    assert snapshot['names']==['first']
    assert snapshot['result'] is detail
    assert not snapshot['result'].arrays['waveform'].flags.writeable

def slow_job(seconds):
    time.sleep(seconds)
    return 'late'

def mapping_error_job():
    from autoacoustics.model import MappingRequired
    raise MappingRequired('请选择数组', {'datasets': ['pressure', 'voltage']})

def child_process_job(marker):
    import subprocess,sys
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
    Path(marker).write_text(str(child.pid))
    time.sleep(30)

def test_cancel_terminates_owned_decoder_child(tmp_path):
    import psutil
    from autoacoustics.jobs import JobExecutor
    executor=JobExecutor(tmp_path/'jobs');marker=tmp_path/'child.pid'
    try:
        job=executor.submit_callable(child_process_job,str(marker))
        deadline=time.monotonic()+5
        while not marker.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert marker.exists()
        pid=int(marker.read_text());assert psutil.pid_exists(pid)
        executor.cancel();events=wait_done(executor,job,timeout=5)
        assert any(event.state=='cancelled' for event in events)
        deadline=time.monotonic()+2
        while psutil.pid_exists(pid) and time.monotonic()<deadline:time.sleep(.01)
        assert not psutil.pid_exists(pid)
    finally:executor.close()

def wait_done(executor, job_id, timeout=15):
    deadline = time.monotonic() + timeout
    events = []
    while time.monotonic() < deadline:
        events.extend(executor.poll())
        if any(event.job_id == job_id and event.state in {'complete', 'failed', 'cancelled'} for event in events):
            return events
        time.sleep(.02)
    raise AssertionError('worker did not finish')

def test_spawn_result_and_immutable_snapshot(tmp_path):
    module = importlib.import_module('autoacoustics.jobs')
    executor = module.JobExecutor(tmp_path)
    try:
        payload = {'x': [1]}
        job = executor.submit_callable(echo_job, payload)
        payload['x'].append(2)
        events = wait_done(executor, job)
        result = next(e.result for e in events if e.state == 'complete')
        assert result == {'echo': {'x': [1]}}
        assert not list(executor.jobs[job]['directory'].glob('*.pkl'))
        assert executor.jobs[job]['payload'] is None
    finally:
        executor.close()

def test_cancel_is_prompt_and_discards_stale_output(tmp_path):
    module = importlib.import_module('autoacoustics.jobs')
    executor = module.JobExecutor(tmp_path)
    try:
        first = executor.submit_callable(slow_job, 30)
        start = time.monotonic()
        executor.cancel()
        assert time.monotonic() - start < 1
        events = wait_done(executor, first, timeout=5)
        assert any(e.state == 'cancelled' for e in events)
        second = executor.submit_callable(echo_job, 'fresh')
        events = wait_done(executor, second)
        assert [e.result for e in events if e.state == 'complete'] == [{'echo': 'fresh'}]
        assert not executor.running
    finally:
        executor.close()

def test_import_mapping_choices_survive_worker_error(tmp_path):
    module = importlib.import_module('autoacoustics.jobs')
    executor = module.JobExecutor(tmp_path)
    try:
        job = executor.submit_callable(mapping_error_job)
        events = wait_done(executor, job)
        failure = next(e for e in events if e.state == 'failed')
        assert failure.result['mapping_options']['datasets'] == ['pressure', 'voltage']
    finally:
        executor.close()

def test_cancelled_batch_keeps_specification_snapshot(tmp_path):
    from autoacoustics.jobs import JobExecutor
    from autoacoustics.model import BatchItemSpec, AnalysisSettings
    executor = JobExecutor(tmp_path)
    try:
        item = BatchItemSpec(Path('missing.wav'), AnalysisSettings(channel=1, start=12, end=34))
        record = {'kind':'batch', 'directory':tmp_path, 'payload':{'items':(item,)}}
        pending = executor._partial(record).items[0]
        assert pending.status == 'cancelled'
        assert pending.specification == item
    finally:
        executor.close()

def test_cancelled_running_batch_keeps_completed_checkpoint_and_pending_config(tmp_path):
    import numpy as np,soundfile as sf
    from autoacoustics.jobs import JobExecutor
    from autoacoustics.model import BatchItemSpec,AnalysisSettings,CalibrationProfile
    path=tmp_path/'batch.wav';rate=48000
    sf.write(path,.02*np.sin(2*np.pi*1000*np.arange(rate*20)/rate),rate,subtype='FLOAT')
    profile=CalibrationProfile('Known test coefficient','Generated signal',2,'FS')
    first=BatchItemSpec(path,AnalysisSettings(channel=0,end=rate,compute_loudness=False),profile)
    second=BatchItemSpec(path,AnalysisSettings(channel=0,compute_loudness=True),profile)
    executor=JobExecutor(tmp_path/'jobs')
    try:
        job=executor.submit('batch',{'items':(first,second)})
        deadline=time.monotonic()+15;progress=[]
        while time.monotonic()<deadline:
            progress.extend(executor.poll())
            if any(event.state=='progress' and event.completed==1 for event in progress):break
            time.sleep(.01)
        assert any(event.state=='progress' and event.completed==1 for event in progress)
        executor.cancel();events=wait_done(executor,job,timeout=5)
        result=next(event.result for event in events if event.state=='cancelled')
        assert result.items[0].status=='ok' and result.items[0].specification==first
        assert result.items[1].status=='cancelled' and result.items[1].specification==second
        assert result.items[0].result.detail is None
    finally:executor.close()
