"""The actual desktop routes from live capture to selected-source NVH."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QPushButton
from autoacoustics.model import ChannelInfo, SignalData
from autoacoustics.ui.main_window import MainWindow
from autoacoustics.ui.plots import PlotPanel


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    win = MainWindow(); win.resize(1024,700); win.show(); app.processEvents()
    yield win
    win.close(); app.processEvents()


def test_core_live_and_nvh_entries_are_visible(window):
    for name in ('soundcard_entry','rotation_entry'):
        button = window.findChild(QPushButton,name)
        assert button is not None, f'Missing core action: {name}'
        assert button.isVisibleTo(window)
        assert window.rect().contains(button.mapTo(window,button.rect().center()))


def test_selected_channel_and_event_are_sent_to_nvh(window,tmp_path):
    class Receiver:
        def set_signal(self,*args,**kwargs): self.value=(args,kwargs)
    receiver=Receiver(); window.rotation_panel=receiver
    signal=SignalData(np.zeros((2,4800)),48000,(ChannelInfo('L','FS'),ChannelInfo('R','FS')),
                      path=tmp_path/'soundcard.wav',source_hash='recorded')
    window.set_signal(signal)
    window.channel_combo.setCurrentIndex(2)
    window.start_spin.setValue(.02);window.end_spin.setValue(.08)
    args,kwargs=receiver.value
    assert args[0] is signal
    assert kwargs['channel']==1
    assert kwargs['start_seconds']==pytest.approx(.02)
    assert kwargs['end_seconds']==pytest.approx(.08)
    window.rotation_panel=None


def test_rotational_references_clear_on_a_new_signal(window,tmp_path):
    signal=SignalData(np.ones((1,1000)),1000,(ChannelInfo('mic','FS'),),path=tmp_path/'mic.wav')
    window.plots.set_signal(signal,0)
    window.plots.set_rotation_reference({'markers':[{'label':'1X','frequency':50.,'order':1.}],
                                        'rpm_mode':'manual','stale':False})
    assert len(window.plots._rotation_items)==2
    window.plots.set_signal(signal,0)
    assert not window.plots._rotation_items


def test_variable_speed_reference_follows_measured_rpm(window):
    window.plots.set_rotation_reference({'markers':[{'label':'1X','frequency':50.,'order':1.}],
        'rpm_mode':'tracked','rpm_times':[1.,2.,3.],'rpm_values':[1200.,2400.,3600.],'stale':False})
    curves=window.plots.plots['stft'].listDataItems()
    assert len(curves)==1
    x,y=curves[0].getData()
    np.testing.assert_allclose(x,[1.,2.,3.]);np.testing.assert_allclose(y,[20.,40.,60.])
    window.plots.set_rotation_reference({'stale':True})
    assert not window.plots.plots['stft'].listDataItems()


def test_recorded_direction_reference_and_chinese_markers_survive_handoff(window,tmp_path):
    signal=SignalData(np.zeros((1,4800)),48000,(ChannelInfo('mic','FS'),),path=tmp_path/'mic.wav',
        metadata={'acquisition_session':{'measurement_context':{'rotation_reference':'从输出轴端观察','motor_rotation':'CW'},
            'events':[{'kind':'稳定','sample_index':480,'time_seconds':.01,'direction':'CW'}]}})
    window.set_signal(signal)
    assert window.context.rotation_view=='从输出轴端观察'
    window.acquisition_markers.setCurrentRow(0);window.use_acquisition_marker()
    assert window.event_combo.currentText()=='运行'
