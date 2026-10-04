"""Opt-in release verification, executed inside the same frozen executable."""
import csv
import hashlib
import json
import math
import time
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import replace
import numpy as np
from .model import ImportMapping,BatchItemSpec,AnalysisSettings,MeasurementContext,CalibrationProfile,to_plain
from .importers import import_signal
from .calibration import load_profile
from .project import write_json_atomic


def _verify_ni_software_contract():
    """Run the bundled guards with explicit test doubles, never claim hardware."""
    from types import SimpleNamespace
    import nidaqmx.constants as constants
    from .acquisition.device import AcquisitionError, ChannelConfig
    from .acquisition.ni_daqmx import NiDaqmxBackend, _microphone_readback
    request=ChannelConfig('test/ai0','test','Pa',measurement_type='microphone',
                          sensitivity_mv_pa=50,max_sound_pressure_db=110)
    facts=dict(ai_microphone_sensitivity=50.,ai_sound_pressure_db_ref=20e-6,
               ai_sound_pressure_max_sound_pressure_lvl=110.,ai_sound_pressure_units=constants.SoundPressureUnits.PA)
    actual=_microphone_readback(SimpleNamespace(**facts),request,constants)
    assert actual['microphone_sensitivity_mv_pa_actual']==50. and actual['sound_pressure_reference_pa_actual']==20e-6
    rejected=0
    for name,value in [('ai_microphone_sensitivity',45.),('ai_sound_pressure_db_ref',1.),
                       ('ai_sound_pressure_units',constants.SoundPressureUnits.FROM_CUSTOM_SCALE),
                       ('ai_sound_pressure_max_sound_pressure_lvl',float('nan'))]:
        try:_microphone_readback(SimpleNamespace(**{**facts,name:value}),request,constants)
        except AcquisitionError as error:
            assert error.flag=='microphone_scaling_unconfirmed';rejected+=1
        else:raise AssertionError('Bundled NI guard accepted an unconfirmed Pa scale')
    order=[]
    class LatchedStream:
        @property
        def overloaded_chans_exist(self):order.append('exists');return True
        @property
        def overloaded_chans(self):
            assert order[-1]=='exists';order.append('channels');return ['test/ai0']
    backend=NiDaqmxBackend(is_simulated=True)
    backend.task=SimpleNamespace(in_stream=LatchedStream())
    overloaded=backend._overload_error()
    assert overloaded.flag=='hardware_overload' and 'test/ai0' in str(overloaded)
    assert order==['exists','channels']
    backend.task=SimpleNamespace(in_stream=SimpleNamespace())
    assert backend._overload_error().flag=='hardware_overload_detection_unavailable'
    return dict(passed=True,is_simulated=True,real_hardware_tested=False,
                Pa_scale_readback=True,invalid_scales_rejected=rejected,
                overload_latch_read_order=True,unknown_overload_rejected=True)

def _verify_head_c2_workflow(window, wait, output, manifest_path):
    """Exercise one real, complete HEAD file session through product jobs.

    Only the manifest-relative data file is read. The historical source_file
    field is provenance, so moving the completed session needs no old folder.
    """
    from .acquisition.storage import read_manifest, safe_session_file
    from .project import file_hash
    from .report import METRIC_NAMES, STATUS_NAMES

    manifest_path, manifest = read_manifest(Path(manifest_path))
    assert manifest['backend']=='HEAD_FILE_HANDOFF' and manifest['status']=='complete'
    assert manifest['is_simulated'] is False and manifest['controlled_recording'] is False
    data_path=safe_session_file(manifest_path,manifest['data_file'])
    manifest_hash=file_hash(manifest_path);data_hash=file_hash(data_path)
    assert data_hash==manifest['data_sha256'].lower()
    window.open_path(manifest_path);wait()
    signal=window.signal
    assert signal is not None and signal.path==manifest_path and signal.source_hash==manifest_hash
    assert signal.channel_count==1 and signal.channels[0].unit=='Pa'
    assert signal.metadata['acquisition_data_sha256']==data_hash
    assert Path(signal.metadata['acquisition_data_path'])==data_path
    assert to_plain(signal.metadata['acquisition_session'])==manifest
    window.profile_combo.setCurrentIndex(0);window.select_full()
    window.loudness_check.setChecked(True);window.stationary_check.setChecked(False)
    window.start_analysis();wait()
    single=window.current_result
    assert single is not None and single.detail is not None and single.input_hash==manifest_hash
    assert single.provenance['source_unit']=='Pa' and single.provenance['calibration']['coefficient']==1.
    for name in ('LZeq','LAeq','LAFmax','LASmax','Nmean','Nmax'):
        metric=single.summary.metrics[name]
        assert metric.status=='ok' and metric.value is not None and math.isfinite(metric.value)
        if name.startswith('N'):assert metric.value>=0
    assert all(metric.status in ('ok','disabled','silent') for metric in single.summary.metrics.values())
    item=BatchItemSpec(manifest_path,single.settings,single.calibration,single.context,
                       window.mapping,manifest_hash)
    window._start_job('batch',{'items':(item,)});wait()
    batch=window.batch_panel.result
    assert batch is not None and len(batch.items)==1
    row=batch.items[0];result=row.result
    assert row.status=='ok' and not row.error and result is not None and result.detail is None
    assert result.result_id==single.result_id and result.input_hash==manifest_hash
    assert result.path==manifest_path and result.provenance['calibration']['coefficient']==1.
    assert to_plain(result.summary.metrics)==to_plain(single.summary.metrics)
    # Summary jobs do not compute octave details. Require all common quality
    # warnings unchanged, and permit only the explicitly scoped startup note.
    startup_prefix='倍频程选段靠近录音起点，低频滤波前史不足；'
    detail_warnings=tuple(w for w in single.summary.warnings if w.startswith(startup_prefix))
    assert tuple(w for w in single.summary.warnings if not w.startswith(startup_prefix))==tuple(result.summary.warnings)
    assert result.provenance['include_detail'] is False and 'octave' not in result.provenance['methods']
    if detail_warnings:
        assert len(detail_warnings)==1
        assert single.provenance['methods']['octave']['startup_reference']['affected_nominal_hz']
    assert to_plain(result.provenance['source_metadata'])==to_plain(single.provenance['source_metadata'])
    assert result.provenance['source_metadata']['acquisition_data_sha256']==data_hash
    output=Path(output).resolve()
    csv_path=output/'head_c2_batch.csv';docx_path=output/'head_c2_batch.docx'
    for kind,target in [('export_csv',csv_path),('report',docx_path)]:
        window._start_job(kind,{'result':batch,'destination':target});wait()
        assert target.is_file() and target.stat().st_size>0
    with csv_path.open(encoding='utf-8-sig',newline='') as stream:
        exported=list(csv.DictReader(stream))
    assert len(exported)==1
    exported=exported[0]
    assert exported['status']=='ok' and not exported['error']
    assert exported['input_hash']==manifest_hash and exported['result_id']==result.result_id
    assert Path(exported['path'])==manifest_path
    assert json.loads(exported['provenance_json'])==to_plain(result.provenance)
    assert json.loads(exported['settings_json'])==to_plain(result.settings)
    assert json.loads(exported['context_json'])==to_plain(result.context)
    for name,metric in result.summary.metrics.items():
        assert all(exported[name+'.'+field]==getattr(metric,field) for field in ('unit','status','method','message'))
        if metric.value is None:assert exported[name+'.value']==''
        else:assert float(exported[name+'.value'])==metric.value
    source_metadata=to_plain(result.provenance['source_metadata'])
    source_digest=hashlib.sha256(json.dumps(source_metadata,ensure_ascii=False,
        sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()
    namespace='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    with zipfile.ZipFile(docx_path) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            if name.endswith(('.xml','.rels')):ET.fromstring(archive.read(name))
        document=ET.fromstring(archive.read('word/document.xml'))
    text=''.join(element.text or '' for element in document.iter(namespace+'t'))
    assert result.result_id in text and manifest_hash in text and source_digest in text
    metric_rows=[]
    for table in document.iter(namespace+'tbl'):
        rows=[[''.join(part.text or '' for part in cell.iter(namespace+'t'))
               for cell in row.findall(namespace+'tc')] for row in table.findall(namespace+'tr')]
        if rows and rows[0]==['指标','数值','单位','状态','方法及说明']:metric_rows.extend(rows[1:])
    assert len(metric_rows)==len(result.summary.metrics)
    by_name={row[0]:row[1:] for row in metric_rows}
    for name,metric in result.summary.metrics.items():
        value='—' if metric.value is None else '−∞' if metric.value==-math.inf else format(metric.value,'.10g')
        actual=by_name[METRIC_NAMES.get(name,name)]
        assert actual[:3]==[value,metric.unit,STATUS_NAMES.get(metric.status,metric.status)]
        assert (not metric.method or 'M1 / '+name in actual[3]) and metric.message in actual[3]
    project_path=output/'HEAD_C2分析项目.json'
    window.save_project_path(project_path);window.open_project_path(project_path);wait()
    assert window.signal.path==manifest_path and window.signal.source_hash==manifest_hash
    assert window.signal.metadata['acquisition_data_sha256']==data_hash
    assert window.current_result.result_id==single.result_id
    assert to_plain(window.current_result.summary)==to_plain(single.summary)
    assert to_plain(window.current_result.provenance['source_metadata'])==source_metadata
    assert file_hash(manifest_path)==manifest_hash and file_hash(data_path)==data_hash
    return {'historical_real_file':manifest['source_file'],
        'selected_manifest':str(manifest_path),'manifest_sha256':manifest_hash,
        'nested_data_path':str(data_path),'nested_data_sha256':data_hash,
        'manifest_import_analysis_project_reopen':True,'batch_csv_docx':True,
        'batch_count':1,'batch_result_id':result.result_id,'Pa_scaled_once':True,
        'summary_preserved':True,'summary_metrics':to_plain(result.summary.metrics),
        'single_and_batch_metrics_identical':True,'common_quality_warnings_preserved':True,
        'detail_only_octave_startup_warnings':len(detail_warnings),
        'source_metadata_sha256':source_digest,'controlled_recording':False,
        'real_device_recording_tested':False,
        'artifacts':{name:{'path':str(path),'sha256':file_hash(path)} for name,path in
                     [('csv',csv_path),('docx',docx_path),('project',project_path)]}}

def verify_bundle(app, request_path):
    from .ui.main_window import MainWindow
    from PyQt6.QtWidgets import QPushButton
    request=json.loads(Path(request_path).read_text(encoding='utf-8-sig'))
    output=Path(request['output_directory']);output.mkdir(parents=True,exist_ok=True)
    fixture_root=Path(request['fixtures_directory'])
    manifest=json.loads((fixture_root/'fixture_manifest.json').read_text(encoding='utf-8'))
    report={'scope':'real codec fixtures and real corpus through product desktop workflow',
            'frozen':bool(getattr(sys,'frozen',False)),
            'started':time.time(),'formats':[],'passed':False}
    report['NI_software_contract']=_verify_ni_software_contract()
    for case in manifest['cases']:
        mapping=ImportMapping(**case['mapping']) if case['mapping'] else None
        signal=import_signal(fixture_root/case['path'],mapping)
        assert math.isclose(signal.sample_rate,case['sample_rate'],rel_tol=1e-10,abs_tol=1e-6),case['path']
        assert signal.channel_count==case['channels'],case['path']
        assert abs(signal.frames-case['source_frames'])<=case['frame_tolerance'],case['path']
        assert [c.unit for c in signal.channels]==case['units'],case['path']
        report['formats'].append({'path':case['path'],'frames':signal.frames,'sample_rate':signal.sample_rate,
            'channels':signal.channel_count,'units':[c.unit for c in signal.channels],'passed':True})
        del signal
    window=MainWindow();window.show();app.processEvents()
    available=window.screen().availableGeometry()
    assert available.contains(window.frameGeometry()),'Main window exceeds the actual available desktop'
    report['desktop_geometry']={'available_width':available.width(),'available_height':available.height(),
        'main_frame_width':window.frameGeometry().width(),'main_frame_height':window.frameGeometry().height(),
        'main_frame_fits':True}
    gaps=[]
    def wait(timeout=90,allowed_states=('complete',)):
        target=window._expected_job[0] if window._expected_job else None
        deadline=time.monotonic()+timeout;last=time.monotonic()
        while time.monotonic()<deadline:
            app.processEvents();window.poll_jobs()
            now=time.monotonic();gaps.append(now-last);last=now
            if window._expected_job is None:
                if target is not None:
                    state=window.jobs.jobs[target].get('terminal_state')
                    assert state in allowed_states,f'{state}: {window.status_label.text()}'
                return
            time.sleep(.01)
        raise TimeoutError(window.status_label.text())
    try:
        corpus_root=Path(request['corpus_directory'])
        paths=sorted(corpus_root.rglob('*.wav'))
        assert len(paths)==64
        window.open_path(paths[0]);wait()
        assert window.signal is not None,window.status_label.text()
        profile=load_profile(Path(request['profile']))
        window.add_profile(profile)
        window.context=MeasurementContext(specimen='真实座椅电机语料',test_level='unknown',
                                         motor_rotation='unknown',actual_movement='unknown',
                                         notes='验证输入仅有文件标签；未确认实际方向或观察基准。')
        window.start_spin.setValue(.5);window.end_spin.setValue(min(2.5,window.signal.duration))
        window.start_analysis();wait()
        first=window.current_result
        assert first is not None and first.detail is not None,window.status_label.text()
        assert first.summary.metrics['Nmax'].value>0
        window.start_spin.setValue(1.);window.end_spin.setValue(min(3.,window.signal.duration))
        window.start_analysis();wait()
        assert len(window.results)==2,window.status_label.text()
        second=window.current_result
        # Two real event snapshots share recorded conditions but have unknown
        # physical directions; the UI must not call them confirmed repeats.
        window.repeat_filter.setCurrentIndex(window.repeat_filter.findData('matching'))
        app.processEvents()
        assert window.history_list.count()==2
        assert '仅记录匹配' in window.repeat_filter_status.text()
        assert '重复试验未确认' in window.repeat_filter_status.text()
        assert window.left_combo.count()==window.right_combo.count()==2
        window.history_list.setCurrentRow(0);app.processEvents()
        assert window.current_result.result_id==first.result_id
        assert window.grab().save(str(output/'desktop_recorded_conditions.png'))
        window.repeat_filter.setCurrentIndex(window.repeat_filter.findData('all'))
        window.display_result(second)
        report['recorded_condition_filter']={'matched_rows':2,'unknown_conditions_visible':True,
            'independent_repeated_trials_confirmed':False,'original_selection_identity_preserved':True,
            'AB_history_preserved':True}
        window.left_combo.setCurrentIndex(0);window.right_combo.setCurrentIndex(1)
        window.compare_selected()
        assert window.comparison is not None
        window.plots.show_dashboard()
        window.plots.select_dashboard(time_key='spl',frequency_key='octave')
        app.processEvents()
        assert window.workspace.currentIndex()==0 and window.diff_table.isVisibleTo(window)
        assert all(window.plots.plots[key].isVisibleTo(window) for key in ('spl','octave','stft'))
        assert window.analysis_tabs.isVisibleTo(window)
        assert 'B−A' in window.metric_cards['LAeq'].title_label.text()
        assert window.grab().save(str(output/'desktop_four_quadrants.png'))
        dialog=window.plots.pop_out('stft');app.processEvents()
        assert window.plots.plots['stft'].isVisibleTo(dialog)
        assert dialog.grab().save(str(output/'desktop_stft_popout.png'))
        dialog.close();app.processEvents()
        assert window.plots.plots['stft'].isVisibleTo(window)
        report['workbench_experience']={'four_quadrants_simultaneously_visible':True,
            'comparison_and_difference_visible':True,'same_plot_popout_and_restore':True,
            'core_metrics_from_result_snapshot':True,'human_acceptance':False}
        report['first_result']={'id':first.result_id,'metrics':to_plain(first.summary.metrics),
            'methods':to_plain(first.provenance['methods'])}
        report['comparison_direction']=window.comparison.direction
        for name,index in [('waveform',0),('spectrum',1),('stft',3),('loudness',6)]:
            window.plots.tabs.setCurrentIndex(index);app.processEvents()
            assert window.grab().save(str(output/f'desktop_{name}.png'))
        for name,result in [('single',first),('comparison',window.comparison)]:
            window._start_job('report',{'result':result,'destination':output/f'{name}.docx'});wait()
            assert (output/f'{name}.docx').is_file(),window.status_label.text()
        if request.get('run_real_batch',True):
            items=tuple(BatchItemSpec(path,AnalysisSettings(channel=0),profile,
                MeasurementContext(specimen=path.stem,
                    notes='样本名称来自文件标签；电机方向、观察基准及实际移动方向尚未确认。')) for path in paths)
            window.batch_panel.specifications=list(items);window.batch_panel.refresh()
            started=time.monotonic();window.batch_panel.findChild(QPushButton,'batch_run').click();wait(600)
            batch=window.batch_panel.result
            assert batch is not None and len(batch.items)==64,window.status_label.text()
            window.workspace.setCurrentWidget(window.batch_panel);app.processEvents()
            assert window.batch_panel.table.rowCount()==64
            window.batch_panel.table.setCurrentCell(0,0);app.processEvents()
            assert 'LAeq' in window.batch_panel.result_readout.toPlainText()
            assert all(window.batch_panel.table.item(row,7).text()=='已完成' for row in range(64))
            assert window.grab().save(str(output/'desktop_batch_workspace.png'))
            report['workbench_experience']['populated_batch_rows_and_readout']=64
            for row in batch.items:
                assert row.status=='ok',f'{row.path}: {row.error}'
                assert row.result.detail is None
                for name in ('LZeq','LAeq','LAFmax','LASmax','Nmean','Nmax'):
                    value=row.result.summary.metrics[name].value
                    assert value is not None and math.isfinite(value)
                    if name.startswith('N'):assert value>=0
            report['batch']={'count':64,'wall_seconds':time.monotonic()-started,'all_finite':True,'details_retained':False}
            for kind,extension in [('export_csv','csv'),('report','docx')]:
                window._start_job(kind,{'result':batch,'destination':output/f'batch.{extension}'});wait()
                assert (output/f'batch.{extension}').is_file(),window.status_label.text()
        # Reopen two detail snapshots from cache, without numerical recomputation.
        project_path=output/'真实音频项目.json';window.save_project_path(project_path)
        window.open_project_path(project_path);wait()
        assert any(r.result_id==first.result_id and r.detail is not None for r in window.results)
        report['project_cache_reopen']=True
        # Batch summaries are also saved in history; select the original event
        # explicitly instead of assuming it is the last reopened result.
        window.history_list.setCurrentRow(next(i for i,r in enumerate(window.results)
                                               if r.result_id==first.result_id))
        app.processEvents()
        assert window.current_result.result_id==first.result_id
        window.repeat_filter.setCurrentIndex(window.repeat_filter.findData('matching'))
        app.processEvents()
        assert window.history_list.count()==2
        assert '重复试验未确认' in window.repeat_filter_status.text()
        report['recorded_condition_filter']['project_reopen']=True
        window.repeat_filter.setCurrentIndex(window.repeat_filter.findData('all'))
        if request.get('run_head',False):
            head_paths=sorted(corpus_root.rglob('*.hdf'))
            assert len(head_paths)==66
            for path in head_paths:
                head=import_signal(path)
                assert head.channels[0].unit=='Pa' and head.channel_count==1
                assert head.metadata['calibration_applied_by_importer'] is False
            window.open_path(head_paths[0]);wait()
            assert window.signal is not None and window.signal.channels[0].unit=='Pa'
            window.context=MeasurementContext(specimen=head_paths[0].stem,
                notes='样本名称来自当前HEAD文件标签；方向、观察基准及实际移动尚未确认。')
            window.context_label.setText('验证工况：'+head_paths[0].stem+'；方向及观察基准未知')
            window.profile_combo.setCurrentIndex(0)
            window.start_spin.setValue(window.signal.time_origin+.5)
            window.end_spin.setValue(window.signal.time_origin+min(2.5,window.signal.duration))
            window.start_analysis();wait()
            head_result=window.current_result
            assert head_result is not None and head_result.detail is not None
            assert head_result.provenance['calibration']['coefficient']==1.
            window.plots.tabs.setCurrentIndex(6);app.processEvents()
            assert window.grab().save(str(output/'desktop_head.png'))
            window._start_job('report',{'result':head_result,'destination':output/'head_single.docx'});wait()
            items=tuple(BatchItemSpec(path,AnalysisSettings(channel=0),
                context=MeasurementContext(specimen=path.stem,notes='文件标签不是经确认的运动方向。')) for path in head_paths)
            window.batch_panel.specifications=list(items);window.batch_panel.refresh()
            started=time.monotonic();window.batch_panel.findChild(QPushButton,'batch_run').click();wait(600)
            batch=window.batch_panel.result
            assert batch is not None and len(batch.items)==66
            assert all(row.status=='ok' and row.result.detail is None for row in batch.items)
            assert all(row.result.provenance['source_unit']=='Pa' and
                       row.result.provenance['calibration']['coefficient']==1. for row in batch.items)
            for row in batch.items:
                for name in ('LZeq','LAeq','LAFmax','LASmax','Nmean','Nmax'):
                    metric=row.result.summary.metrics[name]
                    assert metric.status=='ok' and metric.value is not None and math.isfinite(metric.value)
                    if name.startswith('N'):assert metric.value>=0
            for kind,extension in [('export_csv','csv'),('report','docx')]:
                window._start_job(kind,{'result':batch,'destination':output/f'head_batch.{extension}'});wait(180)
                assert (output/f'head_batch.{extension}').is_file()
            target=output/'HEAD真实音频项目.json';window.save_project_path(target)
            window.open_project_path(target);wait()
            assert any(row.result_id==head_result.result_id for row in window.results)
            report['head']={'count':66,'wall_seconds':time.monotonic()-started,
                'Pa_scaled_once':True,'project_reopen':True,'extra_count':2}
        # Independent extension dialogs, local synthetic evidence only. This
        # verifies frozen dependencies/UI; it is not hardware or real API proof.
        from .ui.knowledge_dialog import KnowledgeDialog
        from .knowledge.models import KnowledgeEntry
        from .knowledge.provider import ProviderClient
        knowledge_path=output/'software-validation-knowledge.sqlite'
        knowledge=KnowledgeDialog(knowledge_path,analysis_provider=lambda:first,parent=window)
        try:
            knowledge.show();app.processEvents()
            assert knowledge.screen().availableGeometry().contains(knowledge.frameGeometry())
            report['desktop_geometry']['knowledge_frame_fits']=True
            saved=knowledge.store.add_source(KnowledgeEntry('软件验证样例：校准记录',
                '合成检索夹具：校准记录须核对传感器与供电。本条不是用户实测案例。'))
            knowledge.refresh_sources();knowledge.tabs.setCurrentIndex(1);knowledge.include_analysis.setChecked(True)
            knowledge.question.setPlainText('LAeq 与校准记录');knowledge.preview_button.click()
            assert knowledge.evidence_list.count()>=1 and str(first.summary.metrics['LAeq'].value) in knowledge.facts.toPlainText()
            assert knowledge.grab().save(str(output/'desktop_knowledge.png'))
            knowledge.store.purge_source(saved.source_id)
            report['knowledge_software']={'local_retrieval_and_factcard':True,'real_API_tested':False,
                'scope':'isolated synthetic local entry plus actual WAV summary; no HTTP request'}
        finally:
            knowledge.close();app.processEvents()
            for suffix in ('','-wal','-shm'):
                path=Path(str(knowledge_path)+suffix)
                if path.exists():path.unlink()
        window.open_acquisition();acquisition=window._extension_dialogs['acquisition']
        deadline=time.monotonic()+15
        while acquisition._busy and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
        assert not acquisition._busy
        assert acquisition.screen().availableGeometry().contains(acquisition.frameGeometry())
        report['desktop_geometry']['acquisition_frame_fits']=True
        import nidaqmx
        report['daq_discovery']={'python_package':nidaqmx.__version__,
            'devices':acquisition.device_combo.count(),'reason':acquisition.device_status.text(),
            'real_acquisition_tested':False}
        assert acquisition.grab().save(str(output/'desktop_acquisition.png'))
        if request.get('run_head'):
            from uuid import uuid4
            handoff_directory=output/('HEAD-C2-'+uuid4().hex[:8])
            handoff_directory.mkdir()
            delivered=[]
            acquisition.recordingReady.connect(lambda path:delivered.append(Path(path)))
            acquisition.tabs.setCurrentIndex(1)
            acquisition.head_source.setText(str(head_paths[0]))
            acquisition.head_directory.setText(str(handoff_directory))
            acquisition.head_condition.setText('历史真实HEAD文件的软件交接验证；当前设备型号与工况未知')
            acquisition.head_stopped.setChecked(True)
            acquisition.head_import_button.click()
            deadline=time.monotonic()+30
            while acquisition._busy and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
            assert not acquisition._busy and delivered,acquisition.head_file_status.text()
            handoff_directory=delivered[-1].parent
            wait()
            assert window.signal.path==handoff_directory/'session.json'
            assert window.signal.metadata['acquisition_data_sha256']
            assert '历史真实HEAD' in window.context.notes
            report['HEAD_C2_software']=_verify_head_c2_workflow(window,wait,output,delivered[-1])
            assert acquisition.grab().save(str(output/'desktop_head_handoff.png'))
        if request.get('head_c2_manifest'):
            report['HEAD_C2_software']=_verify_head_c2_workflow(window,wait,output,request['head_c2_manifest'])
        acquisition.close();app.processEvents()
        if request.get('long_fixture'):
            import psutil
            import threading
            peak=[0.];stop=threading.Event();process=psutil.Process()
            def monitor():
                while not stop.is_set():
                    processes=[process]+process.children(recursive=True)
                    rss=0
                    for candidate in processes:
                        try:rss+=candidate.memory_info().rss
                        except psutil.Error:pass
                    peak[0]=max(peak[0],rss/1024**3);stop.wait(.02)
            thread=threading.Thread(target=monitor,daemon=True);thread.start()
            try:
                long_path=Path(request['long_fixture'])
                window.open_path(long_path);wait(90)
                assert window.signal is not None and window.signal.channel_count==2
                assert window.signal.frames==600*48000
                profile=CalibrationProfile('资源验证合成系数','合成信号已知数值；非仪器校准',1.,'FS')
                window.add_profile(profile)
                window.channel_combo.setCurrentIndex(window.channel_combo.findData(0))
                window.loudness_check.setChecked(True)
                window.start_analysis()
                until=time.monotonic()+.5
                while time.monotonic()<until:app.processEvents();time.sleep(.01)
                cancelled_job=window._expected_job[0]
                started=time.monotonic();window.cancel_job();wait(5,allowed_states=('cancelled',))
                cancel_seconds=time.monotonic()-started
                assert not window.jobs.jobs[cancelled_job]['process'].is_alive()
                assert cancel_seconds<=5
                items=tuple(BatchItemSpec(long_path,AnalysisSettings(channel=channel),profile,
                    MeasurementContext(notes='600秒双通道合成资源输入，不是实测校准。')) for channel in (0,1))
                started=time.monotonic();window._start_job('batch',{'items':items});wait(240)
                batch=window.batch_panel.result
                assert batch is not None and len(batch.items)==2
                assert all(row.status=='ok' and row.result.detail is None for row in batch.items)
                for row in batch.items:
                    for name in ('LZeq','LAeq','LAFmax','LASmax','Nmean','Nmax'):
                        metric=row.result.summary.metrics[name]
                        assert metric.status=='ok' and metric.value is not None and math.isfinite(metric.value)
                        if name.startswith('N'):assert metric.value>=0
                elapsed=time.monotonic()-started
                assert elapsed<=240 and peak[0]<=6
                report['long_dual']={'duration_seconds':600,'channels_analyzed':[0,1],
                    'batch_wall_seconds':elapsed,'parent_and_children_peak_rss_gib':peak[0],
                    'cancel_seconds':cancel_seconds,'cancelled_worker_stopped':True,
                    'budget_wall_seconds':240,'budget_peak_rss_gib':6,'passed':True}
            finally:
                stop.set();thread.join()
        report['maximum_event_loop_gap_seconds']=max(gaps)
        report['passed']=True
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'
        write_json_atomic(report,output/'bundle_validation.json')
        raise
    finally:
        window.close();app.processEvents()
    report['elapsed_seconds']=time.time()-report['started']
    write_json_atomic(report,output/'bundle_validation.json')
    return 0
