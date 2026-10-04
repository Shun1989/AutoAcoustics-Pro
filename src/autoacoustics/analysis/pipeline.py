"""Single numerical entry point shared by GUI, batch and reports."""
import hashlib
from importlib.metadata import version
import numpy as np
from ..model import (AnalysisResult,AnalysisSummary,AnalysisDetail,MetricResult,
                     MeasurementContext,ValidationError,CancelledError,fingerprint,to_plain)
from ..calibration import apply_calibration
from .spectrum import compute_spectrum,compute_stft
from .octave import compute_octave
from .spl import compute_spl
from .loudness import compute_loudness

METHOD_REVISION='autoacoustics-numerics-1'

def analyze(signal,profile,settings,context=None,*,include_detail=True,cancel=None):
    def check():
        if cancel is not None: cancel.check()
    check()
    if 'incomplete_recording' in signal.quality:
        raise ValidationError('此录制未完成，只能恢复查看原始数据，不能生成完整测量结果。')
    channel,start,end=settings.resolve(signal)
    origin=signal.time_origin
    context=context or MeasurementContext()
    info=signal.channels[channel]
    calibrated=info.unit=='Pa' or (profile is not None and info.unit in ('FS','V'))
    effective_profile=profile if info.unit!='Pa' else None
    if calibrated:
        physical=apply_calibration(signal,profile,channel=channel)
        x=physical.samples[0]
        unit='Pa'
        calibration_provenance=physical.metadata['calibration']
    else:
        x=signal.samples[channel]
        unit=info.unit
        calibration_provenance={'source':'未校准' if unit in ('FS','V') else '非声压通道',
                                'input_unit':unit,'coefficient':None}
    removed_dc=float(np.mean(x)) if settings.remove_dc else 0.
    if settings.remove_dc: x=x-removed_dc
    selected=x[start:end]
    silent=bool(np.all(selected==0))
    warnings=list(signal.quality)
    if signal.metadata.get('time_origin_seconds',0.) is None:
        warnings.append('源文件未给出时间起点；图表以第一个音频样本为 0 s。')
    if effective_profile and effective_profile.mode=='reference':
        warnings.append('校准来自同源 HEAD Pa 导出的一致性换算；未验证独立绝对声校准。')
    if info.unit=='Pa' and profile:
        warnings.append('源通道已声明 Pa，选定换算配置被忽略；声压仅缩放一次。')
    if not calibrated:
        warnings.append('缺少声压换算来源，仅提供源单位波形和频谱。')
    metrics={
        'duration':MetricResult((end-start)/signal.sample_rate,'s','ok','original samples [start,end)'),
        'RMS':MetricResult(float(np.sqrt(np.mean(selected**2))),unit,'silent' if silent else 'ok','event RMS')}
    full_scale=1. if info.unit=='FS' else info.full_scale
    if full_scale is not None:
        fs_samples=signal.samples[channel,start:end]
        if settings.remove_dc: fs_samples=fs_samples-np.mean(signal.samples[channel])
        rms=float(np.sqrt(np.mean(fs_samples**2)))
        with np.errstate(divide='ignore'):
            metrics['RMS_dBFS']=MetricResult(float(20*np.log10(rms/full_scale)),'dBFS',
                'silent' if rms==0 else 'ok','RMS / declared ADC full scale')
    arrays={};units={};methods={}
    def calculate(name,function,*args,**kwargs):
        check()
        try:return function(*args,**kwargs)
        except CancelledError:raise
        except Exception as error:
            warnings.append(f'{name} 计算失败：{type(error).__name__}: {error}')
            methods[name]={'status':'failed','error':f'{type(error).__name__}: {error}'}
            return None
    reference=20e-6 if calibrated else 1.
    spectrum_unit='dB re 20 µPa (peak)' if calibrated else ('dBFS (peak)' if unit=='FS' else f'dB re 1 {unit} (peak)')
    if include_detail:
        check()
        arrays.update(wave_time=origin+np.arange(start,end)/signal.sample_rate,waveform=selected)
        units.update(wave_time='s',waveform=unit)
        if len(selected)>=2:
            spectrum=calculate('spectrum',compute_spectrum,selected,signal.sample_rate,settings.fft_size,settings.window,reference)
            if spectrum is not None:
                arrays.update(frequency=spectrum['frequency'],amplitude=spectrum['amplitude'],spectrum_db=spectrum['db'],
                    psd_frequency=spectrum['psd_frequency'],psd=spectrum['psd'])
                units.update(frequency='Hz',amplitude=unit,spectrum_db=spectrum_unit,psd_frequency='Hz',psd=f'{unit}²/Hz')
                methods['spectrum']={k:v for k,v in spectrum.items() if not isinstance(v,np.ndarray)}
            del spectrum
        else:
            warnings.append('事件不足两个样本，未生成 FFT/PSD；其他有效指标保留。')
            methods['spectrum']={'status':'unavailable','reason':'fewer than two samples'}
        check()
        if len(selected)>=settings.stft_size:
            stft=calculate('stft',compute_stft,selected,signal.sample_rate,settings.stft_size,settings.stft_hop,
                              settings.window,origin+start/signal.sample_rate,reference)
            if stft is not None:
                arrays.update(stft_time=stft['time'],stft_frequency=stft['frequency'],stft_db=stft['db'])
                units.update(stft_time='s',stft_frequency='Hz',stft_db=spectrum_unit)
                methods['stft']={k:v for k,v in stft.items() if not isinstance(v,np.ndarray)}
            del stft  # Retain only dB, not another full linear spectrogram.
        else:
            warnings.append('事件不足一个 STFT 窗，未生成语谱图；可选择更短窗口。')
    check()
    spl_names=('LZeq','LAeq','LAFmax','LASmax')
    loudness_names=('Nmean','Nmax','Nstationary')
    if calibrated:
        spl=calculate('spl',compute_spl,x,signal.sample_rate,start,end,settings.history_step_s)
        if spl is not None:methods['spl']=spl['metadata']
        for name in spl_names:
            if spl is None:
                metrics[name]=MetricResult(None,'dB re 20 µPa','failed','SPL',methods['spl']['error'])
                continue
            value=spl[name]
            metrics[name]=MetricResult(value,'dB re 20 µPa','silent' if value==float('-inf') else 'ok',
                'event energy' if name.endswith('eq') else 'maximum continuous exponential A power')
        if include_detail:
            if spl is not None:
                arrays.update(spl_time=origin+spl['time'],laf=spl['laf'],las=spl['las'])
                units.update(spl_time='s',laf='dB re 20 µPa',las='dB re 20 µPa')
            octave=calculate('octave',compute_octave,x,signal.sample_rate,start,end,cancel)
            if octave is not None:
                arrays.update(octave_centers=octave['exact'],octave_levels=octave['levels'])
                units.update(octave_centers='Hz',octave_levels='dB re 20 µPa')
                methods['octave']={k:v for k,v in octave.items() if not isinstance(v,np.ndarray)}
                startup=octave['startup_reference']
                if startup['affected_nominal_hz']:
                    warnings.append('倍频程选段靠近录音起点，低频滤波前史不足；'
                        f"最低频带NI 5/BW参考约{startup['maximum_reference_seconds']:.3f}s。"
                        '此为工程对照提示，非当前滤波器的认证稳定时间；未裁样本或重置滤波。')
        check()
        if settings.compute_loudness:
            loudness=calculate('loudness',compute_loudness,x,signal.sample_rate,start,end,settings.field_type,
                                     settings.stationary_loudness,cancel,include_specific=include_detail)
            if loudness is None:
                for name in loudness_names:
                    metrics[name]=MetricResult(None,'sone','failed' if name!='Nstationary' or settings.stationary_loudness else 'disabled',
                        'ISO 532-1',methods['loudness']['error'])
        if settings.compute_loudness and loudness is not None:
            values=loudness['total']
            status=loudness['status']
            metrics['Nmean']=MetricResult(float(np.mean(values)) if values.size else None,'sone',
                'silent' if silent else status if values.size else 'unavailable','mean 2 ms output in selected event')
            metrics['Nmax']=MetricResult(float(np.max(values)) if values.size else None,'sone',
                'silent' if silent else status if values.size else 'unavailable','maximum 2 ms output in selected event')
            steady=loudness['stationary_total']
            metrics['Nstationary']=MetricResult(steady,'sone',status if steady is not None else 'disabled',
                                               'ISO 532-1 stationary selected event')
            methods['loudness']=loudness['metadata']
            warnings.extend(loudness['metadata']['warnings'])
            if not values.size: warnings.append('事件不包含响度输出时刻；未给出虚构的 0 sone。')
            if include_detail:
                arrays.update(loudness_time=origin+loudness['time'],loudness=values,bark=loudness['bark'])
                units.update(loudness_time='s',loudness='sone',bark='Bark',specific_loudness='sone/Bark')
                if loudness['specific'] is not None: arrays['specific_loudness']=loudness['specific']
        elif not settings.compute_loudness:
            for name in loudness_names: metrics[name]=MetricResult(None,'sone','disabled','ISO 532-1','未启用响度计算。')
    else:
        status='uncalibrated' if unit in ('FS','V') else 'not_applicable'
        for name in spl_names: metrics[name]=MetricResult(None,'dB re 20 µPa',status,'','无可用声压换算。')
        for name in loudness_names: metrics[name]=MetricResult(None,'sone',status,'ISO 532-1','无可用声压换算。')
    check()
    versions={name:version(name) for name in ('numpy','scipy','mosqito','soundfile')}
    if signal.source_hash:
        input_hash=signal.source_hash
    else:
        digest=hashlib.sha256(memoryview(signal.samples))
        digest.update(str(signal.sample_rate).encode())
        input_hash=digest.hexdigest()
    selected_digest=hashlib.sha256()
    for left in range(0,signal.frames,131072):
        check()
        block=np.ascontiguousarray(signal.samples[channel,left:left+131072])
        selected_digest.update(memoryview(block))
    source_layout={'sample_rate':signal.sample_rate,'frames':signal.frames,'channel':info,
        'time_origin_seconds':origin,'source_metadata':signal.metadata,
        'selected_samples_sha256':selected_digest.hexdigest()}
    identity={'input_hash':input_hash,'source_layout':source_layout,'settings':settings,'calibration':effective_profile,
              'context':context,'method_revision':METHOD_REVISION,'versions':versions,
              'metric_statuses':{name:metric.status for name,metric in metrics.items()}}
    provenance={'sample_rate':signal.sample_rate,'time_origin_seconds':origin,'source_frames':signal.frames,'channel':channel,
        'source_time_origin_known':signal.metadata.get('time_origin_seconds',0.) is not None,
        'channel_name':info.name,'source_unit':info.unit,'unit':unit,'source_metadata':signal.metadata,
        'quality':signal.quality,'calibration':calibration_provenance,'versions':versions,
        'method_revision':METHOD_REVISION,'preprocessing':{'scope':'full_recording',
            'remove_dc':settings.remove_dc,'removed_dc':removed_dc},'methods':methods,
        'event_start_sample':start,'event_end_sample':end,'event_interval':'[start,end)',
        'event_start_time_seconds':origin+start/signal.sample_rate,
        'event_end_time_seconds':origin+end/signal.sample_rate,
        'include_detail':include_detail}
    provenance['selected_samples_sha256']=selected_digest.hexdigest()
    return AnalysisResult(fingerprint(identity),input_hash,signal.path,settings,context,effective_profile,
        AnalysisSummary(metrics,warnings),AnalysisDetail._from_owned_arrays(arrays,units) if include_detail else None,provenance)
