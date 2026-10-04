"""Opt-in full-pipeline resource measurement, separate from engine-only probes."""
import json
import time
import threading
import hashlib
from pathlib import Path
import numpy as np
import psutil
from .model import SignalData,ChannelInfo,AnalysisSettings
from .analysis.pipeline import analyze

def run_probe(destination, duration=600, channels=1, include_detail=True):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    peak=[0];stop=threading.Event();process=psutil.Process()
    def monitor():
        while not stop.is_set():
            peak[0]=max(peak[0],process.memory_info().rss);stop.wait(.01)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    started=time.perf_counter();fs=48000
    samples=np.empty((channels,round(duration*fs)))
    block=.02*np.sin(2*np.pi*1000*np.arange(fs)/fs)+.003*np.sin(2*np.pi*125*np.arange(fs)/fs)
    for channel in range(channels):
        for start in range(0,samples.shape[1],fs):
            end=min(start+fs,samples.shape[1]);samples[channel,start:end]=block[:end-start]*(channel+1)
    digest=hashlib.sha256(memoryview(samples)).hexdigest()
    signal=SignalData(samples,fs,tuple(ChannelInfo(f'channel{i}','Pa') for i in range(channels)),source_hash=digest)
    del samples
    report={'scope':'complete pipeline including FFT/STFT/detail/immutable model; synthetic input, not instrument accuracy',
        'duration_seconds':duration,'channels':channels,'selected_channel':0,'include_detail':include_detail,
        'budget_wall_seconds':120 if channels==1 else 240,'budget_peak_rss_gib':3 if channels==1 else 6}
    try:
        result=analyze(signal,None,AnalysisSettings(channel=0),include_detail=include_detail)
        report['Nmax']=result.summary.metrics['Nmax'].value
        report['detail_bytes']=sum(v.nbytes for v in result.detail.arrays.values()) if result.detail else 0
        report['finite_nonnegative_loudness']=bool(np.isfinite(report['Nmax']) and report['Nmax']>=0)
    finally:
        stop.set();thread.join()
        report['wall_seconds']=time.perf_counter()-started
        report['peak_rss_gib']=peak[0]/1024**3
        report['passed']=report['wall_seconds']<=report['budget_wall_seconds'] and report['peak_rss_gib']<=report['budget_peak_rss_gib'] and report.get('finite_nonnegative_loudness',False)
        destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['passed']:raise RuntimeError('Complete-pipeline resource budget exceeded; see probe JSON')
    return report
