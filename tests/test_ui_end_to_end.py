"""Exercise real spawned workers and shared mathematics through desktop state."""
import os,time
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
import numpy as np
import soundfile as sf
from PyQt6.QtWidgets import QApplication
from autoacoustics.ui.main_window import MainWindow
from autoacoustics.model import CalibrationProfile,MeasurementContext

def finish(window,timeout=40):
    app=QApplication.instance()
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        app.processEvents();window.poll_jobs()
        if window._expected_job is None:return
        time.sleep(.01)
    raise AssertionError(window.status_label.text())

def test_real_import_analysis_comparison_batch_report_and_project(tmp_path):
    app=QApplication.instance() or QApplication([])
    window=MainWindow()
    try:
        path=tmp_path/'桌面实际验证.wav'
        rate=48000;t=np.arange(rate)/rate
        sf.write(path,.03*np.sin(2*np.pi*1000*t),rate,subtype='FLOAT')
        window.open_path(path);finish(window)
        assert window.signal is not None,window.status_label.text()
        window.add_profile(CalibrationProfile('本测试已知换算','合成输入已知幅值',2.,'FS'))
        window.context=MeasurementContext(specimen='Seat motor test',test_level='motor',actual_movement='forward')
        window.start_spin.setValue(.2);window.end_spin.setValue(.8)
        window.loudness_check.setChecked(True)
        window.start_analysis();finish(window)
        first=window.current_result
        assert first is not None,window.status_label.text()
        assert first.settings.start==9600 and first.settings.end==38400
        assert first.summary.metrics['LAeq'].status=='ok'
        assert first.summary.metrics['Nmean'].value>0
        assert first.detail.arrays['stft_db'].shape[0]==len(first.detail.arrays['stft_frequency'])
        assert window.plots.plots['psd'].listDataItems()
        window.add_to_batch()
        window.start_spin.setValue(.4);window.end_spin.setValue(.9)
        window.start_analysis();finish(window)
        assert len(window.results)==2
        window.compare_selected()
        assert window.comparison is not None and window.comparison.direction=='right_minus_left'
        assert len(window.plots.plots['waveform'].listDataItems())==2
        window.add_to_batch();window.start_batch();finish(window)
        batch=window.batch_panel.result
        assert batch is not None and len(batch.items)==2,window.status_label.text()
        assert all(row.status=='ok' and row.specification is not None for row in batch.items)
        assert all(row.result.detail is None for row in batch.items)
        destination=tmp_path/'中文报告.docx'
        window._start_job('report',{'result':first,'destination':destination});finish(window)
        assert destination.is_file() and destination.stat().st_size>10000,window.status_label.text()
        csv=tmp_path/'中文汇总.csv'
        window._start_job('export_csv',{'result':batch,'destination':csv});finish(window)
        assert csv.is_file() and 'Seat motor test' in csv.read_text(encoding='utf-8-sig')
        project_path=tmp_path/'项目.json';window.save_project_path(project_path)
        restored=MainWindow()
        try:
            restored.open_project_path(project_path);finish(restored)
            assert len(restored.results)>=2
            assert any(r.result_id==first.result_id and r.detail is not None for r in restored.results)
            assert str(destination) in restored.reports
        finally:restored.close()
    finally:window.close();app.processEvents()

def test_calibrator_worker_builds_real_profile_without_gui_computation(tmp_path):
    app=QApplication.instance() or QApplication([])
    window=MainWindow()
    try:
        path=tmp_path/'校准音.wav';rate=48000;t=np.arange(rate)/rate
        sf.write(path,.1*np.sin(2*np.pi*1000*t),rate,subtype='FLOAT')
        window.open_path(path);finish(window)
        window._start_job('calibration',{'path':path,'input_hash':window.signal.source_hash,
            'parameters':{'known_level_db':94,'channel':0,'start':0,'end':48000,'frequency':1000,'name':'独立校准'}})
        finish(window)
        profile=window.selected_profile()
        assert profile is not None and profile.mode=='calibrator',window.status_label.text()
        assert profile.details['reference_input_hash']==window.signal.source_hash
        assert 14<profile.coefficient<14.3
    finally:window.close();app.processEvents()

def test_historical_head_file_handoff_through_main_window_reopens_session(tmp_path):
    import zipfile
    app=QApplication.instance() or QApplication([]);window=MainWindow()
    with zipfile.ZipFile(Path(__file__).resolve().parents[1]/'压缩.zip') as archive:
        source=tmp_path/'historical.hdf';source.write_bytes(archive.read('HDF/01-01CW.hdf'))
    try:
        window.open_acquisition();dialog=window._extension_dialogs['acquisition']
        deadline=time.monotonic()+15
        while dialog._busy and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
        assert not dialog._busy
        delivered=[];dialog.recordingReady.connect(lambda path:delivered.append(Path(path)))
        dialog.tabs.setCurrentIndex(1);dialog.head_source.setText(str(source))
        dialog.head_directory.setText(str(tmp_path));dialog.head_stopped.setChecked(True)
        dialog.head_condition.setText('historical file software handoff; real current device unknown')
        dialog.head_import_button.click()
        deadline=time.monotonic()+15
        while dialog._busy and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
        assert delivered,dialog.head_file_status.text()
        finish(window)
        assert window.signal.path==delivered[0]
        assert window.signal.channels[0].unit=='Pa'
        assert window.signal.metadata['acquisition_session']['controlled_recording'] is False
        assert 'software handoff' in window.context.notes
        window.profile_combo.setCurrentIndex(0);window.loudness_check.setChecked(False)
        window.start_analysis();finish(window)
        result=window.current_result
        assert result.summary.metrics['LAeq'].status=='ok'
        assert result.provenance['calibration']['coefficient']==1.
        project=tmp_path/'handoff-project.json';window.save_project_path(project)
        window.open_project_path(project);finish(window)
        assert window.signal.path==delivered[0] and window.current_result.result_id==result.result_id
    finally:window.close();app.processEvents()
