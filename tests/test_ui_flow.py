import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import importlib
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication
from autoacoustics.model import (SignalData, ChannelInfo, AnalysisSettings,
    AnalysisResult, AnalysisSummary, AnalysisDetail, MetricResult, MeasurementContext)

@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    windows = []
    def make():
        module = importlib.import_module('autoacoustics.ui.main_window')
        win = module.MainWindow()
        windows.append(win)
        return win
    yield make
    for win in windows:
        win.close()
    app.processEvents()

def signal(tmp_path, channels=1):
    import soundfile as sf
    path = tmp_path / '真实音频.wav'
    samples = np.zeros((channels, 4800))
    sf.write(path, samples.T, 48000)
    from autoacoustics.project import file_hash
    return SignalData(samples, 48000, tuple(ChannelInfo(str(i), 'FS', 'digital') for i in range(channels)),
                      path, file_hash(path))

def result(sig):
    return AnalysisResult('result-1', sig.source_hash, sig.path,
        AnalysisSettings(channel=0, start=100, end=2000), MeasurementContext(), None,
        AnalysisSummary({'LAeq': MetricResult(None, 'dB', 'uncalibrated')}),
        AnalysisDetail({'wave_time': np.arange(1900)/48000+100/48000,
            'waveform': np.zeros(1900)}, {'waveform':'FS'}))

def test_six_step_channel_and_event_flow(window, tmp_path):
    window = window()
    sig = signal(tmp_path, 2)
    window.set_signal(sig)
    assert window.channel_combo.currentData() is None
    assert not window.analyze_button.isEnabled()
    window.channel_combo.setCurrentIndex(1)
    assert window.analyze_button.isEnabled()
    window.start_spin.setValue(.01)
    window.end_spin.setValue(.05)
    settings = window.analysis_settings()
    assert (settings.channel, settings.start, settings.end) == (0, 480, 2400)
    assert 'FS' in window.unit_label.text()
    window.add_current_event('启动')
    assert window.event_list.count() == 1
    assert window.project_items[0].settings.event_label == 'startup'

def test_result_changes_become_stale_and_project_reopens(window, tmp_path):
    window = window()
    sig = signal(tmp_path)
    window.set_signal(sig)
    res = result(sig)
    window.accept_result(res)
    assert '未校准' in window.metric_table.item(0, 1).text()
    project_path = tmp_path / '测量项目.json'
    window.save_project_path(project_path)
    window.start_spin.setValue(.01)
    assert not window.current_result_valid
    window.open_project_path(project_path)
    assert len(window.results) == 1
    assert window.results[0].detail is not None
    assert window.results[0].result_id == 'result-1'

def test_batch_multiple_events_have_independent_configuration(window, tmp_path):
    window = window()
    sig = signal(tmp_path)
    window.set_signal(sig)
    window.start_spin.setValue(.01)
    window.end_spin.setValue(.03)
    window.add_to_batch()
    window.start_spin.setValue(.04)
    window.end_spin.setValue(.08)
    window.add_to_batch()
    items = window.batch_panel.items()
    assert len(items) == 2 and items[0].item_id != items[1].item_id
    assert (items[0].settings.start, items[1].settings.start) == (480, 1920)

def test_pending_failed_import_is_the_mapping_target(window,tmp_path):
    window=window()
    window.set_signal(signal(tmp_path))
    new=tmp_path/'新数据.mat'
    window._pending_import=(new,None)
    assert window.mapping_target()==new

def test_mapping_dialog_preserves_explicit_time_unit(window):
    window=window()
    from autoacoustics.model import ImportMapping
    from autoacoustics.ui.import_dialog import ImportDialog
    dialog=ImportDialog(window,mapping=ImportMapping(time_column='time',time_unit='ms',channels=('Group/Pressure',)))
    assert dialog.mapping().time_unit=='ms'
    dialog.time_unit.setCurrentIndex(dialog.time_unit.findData('us'))
    assert dialog.mapping().time_unit=='us'

def test_batch_editor_preserves_embedded_profile_and_untouched_settings(window,tmp_path):
    window=window()
    from autoacoustics.ui.batch_panel import BatchItemDialog
    from autoacoustics.model import CalibrationProfile,BatchItemSpec
    profile=CalibrationProfile('Project-only profile','Record 42',2,'FS')
    settings=AnalysisSettings(channel=0,start=10,end=100,fft_size=5555,stft_size=500,stft_hop=73,
        history_step_s=.02,window='hamming',spl_time_weighting='Slow')
    item=BatchItemSpec(tmp_path/'sample.wav',settings,profile)
    dialog=BatchItemDialog(item,[],window)
    assert dialog.specification().profile==profile
    assert dialog.specification().settings==settings

def test_history_selection_restores_waveform_selection_after_overlay(window,tmp_path):
    from autoacoustics.analysis.pipeline import analyze
    window=window();sig=signal(tmp_path);window.set_signal(sig)
    res=analyze(sig,None,AnalysisSettings(channel=0,start=100,end=2000,compute_loudness=False))
    window.accept_result(res)
    second=replace(res,result_id='result-2')
    window.accept_result(second);window.plots.overlay(res,second)
    window.show_history_result(0)
    assert window.plots.region in window.plots.plots['waveform'].items()
    assert window.analysis_settings()==res.settings

def test_acquisition_session_restores_context_and_explicit_point_markers(window,tmp_path):
    sig=signal(tmp_path)
    sig=replace(sig,metadata={'acquisition_session':{
        'measurement_context':{'condition':'12 V / 负载未知','operator':'operator-1',
            'motion_direction':'座椅向前','cw_reference':'从输出轴端观察'},
        'events':[{'kind':'startup','sample_index':480,'time_seconds':.01,
            'direction':'CW','note':'人工启动标记'}]}})
    win=window();win.set_signal(sig)
    assert win.context.actual_movement=='座椅向前'
    assert win.context.rotation_view=='从输出轴端观察'
    assert win.context.load=='unknown' and win.context.motor_rotation=='unknown'
    assert '12 V / 负载未知' in win.context.notes
    assert win.acquisition_markers.count()==1
    win.acquisition_markers.setCurrentRow(0);win.use_acquisition_marker()
    assert win.analysis_settings().start==480
    assert win.analysis_settings().end==4800
    assert win.context.motor_rotation=='CW'
    assert win.selected_profile() is None
    assert '终点' in win.status_label.text()
    win.set_signal(signal(tmp_path))
    assert win.acquisition_markers.count()==0

def test_long_source_labels_do_not_force_horizontal_controls_overflow(window,tmp_path):
    window=window();window.resize(1280,850);window.set_signal(signal(tmp_path))
    window.file_label.setText('C:\\'+'very_long_measurement_folder\\'*10+'座椅电机.wav\n1 通道 · 48000 Hz')
    window.show();QApplication.instance().processEvents()
    from PyQt6.QtWidgets import QScrollArea
    scroll=window.findChild(QScrollArea)
    assert scroll.widget().width()<=scroll.viewport().width()

def test_main_window_initial_geometry_fits_the_available_screen(window):
    win=window();available=win.screen().availableGeometry()
    assert win.height()<=available.height()-48
    assert win.width()<=max(win.minimumSizeHint().width(),available.width()-24)

def test_new_import_does_not_replace_previous_signal_mapping_until_success(window,tmp_path):
    from autoacoustics.model import ImportMapping
    window=window();window.set_signal(signal(tmp_path))
    old=ImportMapping(unit='FS');new=ImportMapping(unit='V',sample_rate=1000)
    window.mapping=old;window.open_path(tmp_path/'missing.mat',new)
    assert window.mapping==old
    assert window._pending_import[1]==new

def test_nonzero_source_origin_preserves_samples_plot_region_and_playback(window,tmp_path):
    window=window();sig=replace(signal(tmp_path),metadata={'time_origin_seconds':2.5})
    window.set_signal(sig)
    assert window.start_spin.value()==2.5 and window.end_spin.value()==2.6
    window.start_spin.setValue(2.51);window.end_spin.setValue(2.55)
    assert (window.analysis_settings().start,window.analysis_settings().end)==(480,2400)
    x,y=window.plots.plots['waveform'].listDataItems()[0].getData()
    assert x[0]==2.5 and tuple(window.plots.region.getRegion())==(2.51,2.55)
    window._restore_item_controls(window.current_item())
    assert window.start_spin.value()==2.51
    window.player.set_segment(sig.samples[0],sig.sample_rate,480,2400,time_origin=sig.time_origin)
    assert window.player.origin==2.51
def test_full_detail_replaces_same_identity_summary_in_history_and_project(tmp_path):
    import numpy as np
    from autoacoustics.model import SignalData,ChannelInfo,AnalysisSettings
    from autoacoustics.analysis.pipeline import analyze
    from autoacoustics.ui.main_window import MainWindow
    from PyQt6.QtWidgets import QApplication
    app=QApplication.instance() or QApplication([])
    signal=SignalData(np.ones((1,2000))*.01,48000,(ChannelInfo('mic','Pa'),),path=tmp_path/'source.wav',source_hash='source')
    settings=AnalysisSettings(channel=0,compute_loudness=False)
    summary=analyze(signal,None,settings,include_detail=False)
    complete=analyze(signal,None,settings)
    window=MainWindow()
    try:
        window.results=[summary]
        window.accept_result(complete)
        assert len(window.results)==1 and window.results[0].detail is not None
        assert window.project().results[0].result.detail is not None
    finally:window.close();app.processEvents()

def test_history_same_file_other_dataset_shows_cached_waveform_not_current_source(window,tmp_path):
    from autoacoustics.analysis.pipeline import analyze
    from autoacoustics.model import ImportMapping
    first=SignalData(np.ones((1,2000))*.01,48000,(ChannelInfo('mic','Pa'),),
        path=tmp_path/'same.hdf5',source_hash='same-file',metadata={'selected_dataset':'a','import_mapping':{'dataset':'a'}})
    second=replace(first,samples=first.samples*2,metadata={'selected_dataset':'b','import_mapping':{'dataset':'b'}})
    saved=analyze(second,None,AnalysisSettings(channel=0,compute_loudness=False))
    win=window();win.mapping=ImportMapping(dataset='a');win.set_signal(first)
    win.results=[saved];win.show_history_result(0)
    _,shown=win.plots.plots['waveform'].listDataItems()[0].getData()
    np.testing.assert_array_equal(shown,second.samples[0])
    assert win.mapping.dataset=='a'

