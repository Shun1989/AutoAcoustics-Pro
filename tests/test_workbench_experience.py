"""User-visible workflow regressions from the original product video."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from dataclasses import replace
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QPushButton
from autoacoustics.model import SignalData, ChannelInfo, AnalysisSettings
from autoacoustics.analysis.pipeline import analyze
from autoacoustics.ui.main_window import MainWindow


@pytest.fixture
def win():
    app=QApplication.instance() or QApplication([])
    window=MainWindow();window.resize(1380,790);window.show();app.processEvents()
    yield window
    window.close();app.processEvents()


def recording(tmp_path, unit='Pa'):
    return SignalData(.01*np.sin(2*np.pi*1000*np.arange(4800)/48000)[None,:],48000,
        (ChannelInfo('测点1',unit),),path=tmp_path/'source.wav',source_hash='input-1')


def test_core_actions_are_visible_without_scrolling(win):
    for name in ('open_file','open_project','save_project','analyze','export_report','play_segment','stop_audio'):
        button=win.findChild(QPushButton,name)
        assert button.isVisibleTo(win),name
        assert win.rect().contains(button.mapTo(win,button.rect().center())),name
    assert win.analysis_tabs.count()>=5
    assert win.plots.dashboard.isVisibleTo(win)


def test_metric_cards_preserve_uncalibrated_and_stale_states(win,tmp_path):
    signal=recording(tmp_path,'FS');win.set_signal(signal)
    result=analyze(signal,None,AnalysisSettings(channel=0,compute_loudness=False))
    win.accept_result(result)
    assert '未校准' in win.metric_cards['LAeq'].status_label.text()
    assert win.metric_cards['LAeq'].value_label.text()=='—'
    win.start_spin.setValue(.01)
    assert '过期' in win.metric_cards['LAeq'].status_label.text()
    assert not win.current_result_valid


def test_comparison_opens_visible_overlay_and_difference_panel(win,tmp_path):
    signal=recording(tmp_path);win.set_signal(signal)
    settings=AnalysisSettings(channel=0,compute_loudness=False)
    left=analyze(signal,None,settings)
    right=replace(left,result_id='other-event')
    win.accept_result(left);win.accept_result(right)
    win.workspace.setCurrentIndex(1)
    win.compare_selected();QApplication.instance().processEvents()
    assert win.workspace.currentIndex()==0
    assert win.diff_table.isVisibleTo(win)
    assert win.plots.plots['spectrum'].isVisibleTo(win)
    assert len(win.plots.plots['spectrum'].listDataItems())>=2
    assert 'B−A' in win.metric_cards['LAeq'].title_label.text()
    win.show_history_result(0)
    assert 'B−A' not in win.metric_cards['LAeq'].title_label.text()


def test_small_workbench_keeps_dashboard_and_navigation_accessible(win):
    win.resize(1024,700);QApplication.instance().processEvents()
    assert win.width()<=1024
    assert win.plots.dashboard.isVisibleTo(win)
    assert win.plots.plots['waveform'].height()>=120
    assert win.plots.plots['spectrum'].height()>=120
    assert win.analysis_tabs.isVisibleTo(win)


def test_new_import_clears_previous_physical_metric_cards(win,tmp_path):
    signal=recording(tmp_path);win.set_signal(signal)
    win.accept_result(analyze(signal,None,AnalysisSettings(channel=0,compute_loudness=False)))
    assert win.metric_cards['LAeq'].value_label.text()!='—'
    win.set_signal(recording(tmp_path,'FS'))
    assert win.metric_cards['LAeq'].value_label.text()=='—'
    assert win.metric_cards['LAeq'].status_label.text()=='等待分析'


def test_opening_empty_project_clears_previous_metric_cards(win,tmp_path):
    from autoacoustics.model import MeasurementProject
    from autoacoustics.project import save_project
    signal=recording(tmp_path);win.set_signal(signal)
    win.accept_result(analyze(signal,None,AnalysisSettings(channel=0,compute_loudness=False)))
    path=tmp_path/'empty.json';save_project(MeasurementProject('空项目'),path)
    win.open_project_path(path)
    assert win.metric_cards['LAeq'].value_label.text()=='—'


def test_toolbar_exports_visible_comparison_instead_of_another_result(win,tmp_path,monkeypatch):
    signal=recording(tmp_path);win.set_signal(signal)
    left=analyze(signal,None,AnalysisSettings(channel=0,compute_loudness=False))
    for result in (left,replace(left,result_id='B'),replace(left,result_id='C')):
        win.accept_result(result)
    win.left_combo.setCurrentIndex(0);win.right_combo.setCurrentIndex(1);win.compare_selected()
    exported=[];monkeypatch.setattr(win,'_export',lambda result,*args:exported.append(result))
    win.findChild(QPushButton,'export_report').click()
    assert exported==[win.comparison]
    played=[];monkeypatch.setattr(win,'play_result',played.append)
    win.findChild(QPushButton,'play_segment').click()
    assert played==[win.comparison.right]


def test_history_return_hides_old_comparison_details(win,tmp_path):
    signal=recording(tmp_path);win.set_signal(signal)
    left=analyze(signal,None,AnalysisSettings(channel=0,compute_loudness=False))
    win.accept_result(left);win.accept_result(replace(left,result_id='B'));win.compare_selected()
    win.show_history_result(0)
    assert win.analysis_tabs.currentIndex()==0
    assert not win.diff_table.isVisibleTo(win)


def test_visible_comparison_playback_uses_displayed_snapshot(win,tmp_path,monkeypatch):
    signal=recording(tmp_path);win.set_signal(signal)
    left=analyze(signal,None,AnalysisSettings(channel=0,compute_loudness=False))
    for result in (left,replace(left,result_id='B'),replace(left,result_id='C')):
        win.accept_result(result)
    win.left_combo.setCurrentIndex(0);win.right_combo.setCurrentIndex(1);win.compare_selected()
    win.left_combo.setCurrentIndex(2)
    played=[];monkeypatch.setattr(win,'play_result',played.append)
    win.findChild(QPushButton,'play_a').click()
    assert played==[win.comparison.left]


def test_editing_pending_parameters_keeps_comparison_snapshot_but_channel_switch_exits(win,tmp_path):
    source=recording(tmp_path)
    signal=replace(source,samples=np.vstack((source.samples,source.samples)),
        channels=(ChannelInfo('A','Pa'),ChannelInfo('B','Pa')))
    win.set_signal(signal);win.channel_combo.setCurrentIndex(win.channel_combo.findData(0))
    left=analyze(signal,None,AnalysisSettings(channel=0,compute_loudness=False))
    win.accept_result(left);win.accept_result(replace(left,result_id='B'));win.compare_selected()
    comparison=win.comparison
    win.fft_combo.setCurrentIndex(2)
    assert win._view_comparison and win.comparison is comparison
    assert 'B−A' in win.metric_cards['LAeq'].title_label.text()
    assert not win.current_result_valid
    win.channel_combo.setCurrentIndex(win.channel_combo.findData(1))
    assert not win._view_comparison
    assert 'B−A' not in win.metric_cards['LAeq'].title_label.text()
    assert '过期' in win.metric_cards['LAeq'].status_label.text()


def test_many_unknown_comparison_conditions_leave_differences_readable(win,tmp_path):
    signal=recording(tmp_path);win.set_signal(signal)
    left=analyze(signal,None,AnalysisSettings(channel=0,compute_loudness=False))
    win.accept_result(left);win.accept_result(replace(left,result_id='B'));win.compare_selected()
    QApplication.instance().processEvents()
    assert len(win.comparison.condition_differences)>10
    assert win.diff_table.height()>=100
