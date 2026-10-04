"""Explicit native desktop/capture acceptance; never runs during normal startup."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np


def verify_live_nvh(app,destination,capture_seconds=0.):
    from .ui.main_window import MainWindow
    from .acquisition.soundcard import discover_soundcards
    from .acquisition.storage import open_acquisition_session
    from .importers.registry import import_signal
    from .analysis.pipeline import analyze
    from .analysis.rotation import export_rotation_json,export_rotation_report
    from .model import AnalysisSettings
    import soundfile as sf
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    proof={'schema_version':1,'frozen':bool(getattr(sys,'frozen',False)),
           'capture_requested_seconds':capture_seconds,'checks':{},'errors':[]}
    window=MainWindow();window.show();window.open_soundcard();app.processEvents()
    panel=window.soundcard_panel
    def wait_until(predicate,timeout=20):
        started=time.monotonic()
        while not predicate():
            app.processEvents()
            if time.monotonic()-started>timeout:raise RuntimeError('Desktop operation timed out')
            time.sleep(.01)
        app.processEvents()
    try:
        discovery=discover_soundcards()
        proof['devices']=[{'name':d.name,'id':d.device_id,'metadata':dict(d.metadata)} for d in discovery.devices]
        if not discovery.available:raise RuntimeError(discovery.reason)
        if not window.grab().save(str(destination/'desktop_capture_ready.png')):raise RuntimeError('Screenshot failed')
        proof['checks']['real_device_list']=panel.device_combo.count()==len(discovery.devices)
        if capture_seconds:
            if not 1<=capture_seconds<=30:raise ValueError('Capture verification must be 1–30 seconds')
            panel.save_parent.setText(str(destination/'recordings'))
            panel.target_duration.setValue(capture_seconds)
            panel.start_recording()
            wait_until(lambda:panel.preview_frames>0 or (not panel.is_recording and panel._engine is None))
            if not panel.preview_frames:raise RuntimeError(panel.status_label.text())
            preview_rate=float(panel.rate_combo.currentData() or panel.rate_combo.currentText())
            wait_until(lambda:panel.preview_frames>=min(capture_seconds,1.)*preview_rate or panel.completed_wav is not None)
            window.grab().save(str(destination/'desktop_live_microphone.png'))
            wait_until(lambda:panel.completed_wav is not None or (not panel.is_recording and panel._engine is not None and panel._engine.snapshot().status.value=='failed'))
            if panel.completed_wav is None:raise RuntimeError(panel.status_label.text())
            wait_until(lambda:not panel.is_recording)
            signal=open_acquisition_session(panel.completed_manifest)
            samples,rate=sf.read(panel.completed_wav,dtype='float64',always_2d=True)
            np.testing.assert_array_equal(samples.T,signal.samples)
            session=panel._engine.snapshot()
            proof['capture']={'manifest':str(panel.completed_manifest),'wav':str(panel.completed_wav),
                'source_hash':signal.source_hash,'frames_read':session.frames_read,'frames_written':session.frames_written,
                'sample_rate':signal.sample_rate,'channels':signal.channel_count,'unit':signal.channels[0].unit,
                'status':session.status.value,'flags':list(session.flags),'errors':list(session.errors)}
            proof['checks']['wav_equals_session']=True
            panel.send_to_analysis()
            wait_until(lambda:window.current_result_valid,40)
            proof['checks']['handoff_analyzed_source']=window.signal.source_hash==signal.source_hash
            proof['checks']['uncalibrated_mic_not_absolute_spl']=window.current_result.summary.metrics['LAeq'].status=='uncalibrated'
            window.grab().save(str(destination/'desktop_recorded_analysis.png'))
        audio=Path.cwd()/'data'/'corpus'/'HDF'/'01-01CCW.hdf'
        if not audio.is_file():raise RuntimeError('Acceptance HEAD recording not found in working directory')
        signal=import_signal(audio);window.set_signal(signal)
        result=analyze(signal,None,AnalysisSettings(channel=0,compute_loudness=False));window.accept_result(result)
        window.open_rotation();rotation=window.rotation_panel
        rotation.rpm_spin.setValue(3000);rotation.gear_teeth_spin.setValue(20);rotation.gear_ratio_spin.setValue(5)
        rotation.start_analysis();wait_until(lambda:not rotation.is_analyzing,40)
        if rotation.result is None or rotation.result_stale:raise RuntimeError(rotation.status_label.text())
        path=rotation.export_result(destination/'HEAD真实录音_NVH.docx')
        proof['nvh']={'source':str(audio),'source_hash':signal.source_hash,'rpm_mode':rotation.result.rpm_mode,
            'rpm':rotation.result.config.rpm,'rpm_is_test_assumption':True,'report':str(path),
            'peaks':len(rotation.result.peaks),'findings':len(rotation.result.findings)}
        proof['checks']['selected_source_nvh']=rotation._result_snapshot['source_hash']==signal.source_hash
        proof['checks']['physical_references_visible']=bool(window.plots._rotation_items)
        window.grab().save(str(destination/'desktop_rotation_analysis.png'))
        available=window.screen().availableGeometry()
        frame=window.frameGeometry()
        proof['geometry']={'available':[available.x(),available.y(),available.width(),available.height()],
                           'frame':[frame.x(),frame.y(),frame.width(),frame.height()]}
        proof['checks']['rotation_window_fits_screen']=frame.width()<=available.width() and frame.height()<=available.height()
        window.workspace.setCurrentIndex(0);app.processEvents()
        window.grab().save(str(destination/'desktop_frequency_references.png'))
        window.resize(1024,700);window.open_soundcard();app.processEvents()
        window.grab().save(str(destination/'desktop_capture_small.png'))
        from PyQt6.QtWidgets import QPushButton
        proof['checks']['primary_actions_fit_small_screen']=all(window.rect().contains(b.mapTo(window,b.rect().center()))
            for name in ('soundcard_entry','rotation_entry','open_file','analyze')
            if (b:=window.findChild(QPushButton,name)) is not None)
        proof['checks']['capture_actions_visible']=all(button.isVisibleTo(window) and
            window.rect().contains(button.mapTo(window,button.rect().center()))
            for button in (panel.start_button,panel.stop_button))
        proof['passed']=all(proof['checks'].values())
    except Exception as error:
        import traceback
        proof['errors'].append({'message':str(error),'traceback':traceback.format_exc()});proof['passed']=False
    finally:
        window.close()
        wait_until(lambda:not window.isVisible(),20)
        (destination/'live_nvh_validation.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0 if proof['passed'] else 2
