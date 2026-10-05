"""Playback state must reflect one active output and the user's current selection."""
import numpy as np
import pytest
from PyQt6.QtMultimedia import QAudio


class Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, value):
        for callback in list(self.callbacks):
            callback(value)


class Output:
    def isNull(self):
        return False

    def isFormatSupported(self, format):
        return True


@pytest.fixture
def playback_backend(monkeypatch):
    from autoacoustics.ui import playback

    class Sink:
        instances = []
        start_error = QAudio.Error.NoError

        def __init__(self, *args):
            self.stateChanged = Signal()
            self.current_state = QAudio.State.StoppedState
            self.current_error = self.start_error
            self.deleted = False
            self.instances.append(self)

        def start(self, buffer):
            self.current_state = (QAudio.State.ActiveState if self.current_error == QAudio.Error.NoError
                                  else QAudio.State.StoppedState)
            self.stateChanged.emit(self.current_state)

        def state(self):
            return self.current_state

        def error(self):
            return self.current_error

        def processedUSecs(self):
            return 100000

        def stop(self):
            self.current_state = QAudio.State.StoppedState
            self.stateChanged.emit(self.current_state)

        def suspend(self):
            self.current_state = QAudio.State.SuspendedState
            self.stateChanged.emit(self.current_state)

        def resume(self):
            self.current_state = QAudio.State.ActiveState
            self.stateChanged.emit(self.current_state)

        def deleteLater(self):
            self.deleted = True

    monkeypatch.setattr(playback, 'QAudioSink', Sink)
    monkeypatch.setattr(playback.QMediaDevices, 'defaultAudioOutput', lambda: Output())
    return playback, Sink


def test_repeated_play_owns_one_sink_and_replay_releases_finished_output(playback_backend):
    module, backend = playback_backend
    player = module.SegmentPlayer()
    try:
        player.set_segment(np.zeros(4800), 48000)
        player.play()
        first = player.sink
        player.play()
        assert player.sink is first and len(backend.instances) == 1
        first.current_state = QAudio.State.IdleState
        first.stateChanged.emit(first.current_state)
        assert player.state == 'stopped'
        player.play()
        assert len(backend.instances) == 2 and first.deleted
        assert player.state == 'playing'
        first.stateChanged.emit(QAudio.State.IdleState)
        assert player.state == 'playing', 'An obsolete output must not stop the new playback.'
    finally:
        player.stop()


@pytest.mark.parametrize('operation', ['start', 'run', 'pause', 'resume'])
def test_output_failure_stops_timer_releases_output_and_reports_error(playback_backend, operation):
    module, backend = playback_backend
    player = module.SegmentPlayer()
    messages = []
    player.message.connect(messages.append)
    try:
        player.set_segment(np.zeros(4800), 48000)
        if operation == 'start':
            backend.start_error = QAudio.Error.OpenError
        player.play()
        failed = backend.instances[-1]
        if operation != 'start':
            assert player.state == 'playing'
            def disconnect():
                failed.current_error = QAudio.Error.IOError
                failed.current_state = QAudio.State.StoppedState
                failed.stateChanged.emit(failed.current_state)
            if operation == 'pause':
                failed.suspend = disconnect
                player.pause()
            elif operation == 'resume':
                player.pause()
                failed.resume = disconnect
                player.play()
            else:
                disconnect()
        assert player.state == 'stopped' and not player.timer.isActive()
        assert player.sink is None and player.buffer is None and failed.deleted
        assert messages and any('播放' in message and '失败' in message for message in messages)
    finally:
        player.stop()


def test_workbench_play_resumes_same_selection_and_switches_changed_channel():
    from autoacoustics.model import ChannelInfo, SignalData
    from autoacoustics.ui.main_window import MainWindow

    class Player:
        state = 'stopped'
        preparations = 0
        resumes = 0

        def stop(self):
            self.state = 'stopped'

        def set_segment(self, samples, rate, start=0, end=None, gain=1., time_origin=0.):
            self.stop()
            self.preparations += 1
            self.samples = np.array(samples[start:end], copy=True)

        def play(self):
            if self.state == 'paused':
                self.resumes += 1
            self.state = 'playing'

    window = MainWindow()
    window.player.stop()
    window.player = Player()
    try:
        window.set_signal(SignalData(np.vstack([np.full(4800, .01), np.full(4800, .02)]),
                                     48000, (ChannelInfo('left', 'FS'), ChannelInfo('right', 'FS'))))
        window.profile_combo.setCurrentIndex(0)
        window.channel_combo.setCurrentIndex(1)
        window.play_visible()
        assert window.player.preparations == 1
        window.player.state = 'paused'
        window.play_visible()
        assert window.player.resumes == 1 and window.player.preparations == 1
        window.channel_combo.setCurrentIndex(2)
        assert window.player.state == 'stopped'
        window.play_visible()
        assert window.player.preparations == 2
        np.testing.assert_array_equal(window.player.samples, np.full(4800, .02))
    finally:
        window.close()


def test_saved_result_playback_uses_recorded_rate_with_large_time_origin():
    from types import SimpleNamespace
    from autoacoustics.ui.main_window import MainWindow

    class Player:
        state = 'stopped'

        def stop(self):
            self.state = 'stopped'

        def set_segment(self, samples, rate, **kwargs):
            self.rate = rate
            self.state = 'stopped'

        def play(self):
            self.state = 'playing'

    window = MainWindow()
    window.player.stop()
    window.player = Player()
    try:
        result = SimpleNamespace(result_id='saved-result', provenance={'sample_rate': 48000},
                                 detail=SimpleNamespace(arrays={'wave_time': 1e9 + np.arange(4800)/48000,
                                                              'waveform': np.zeros(4800)}))
        window.play_result(result)
        assert window.player.rate == 48000
    finally:
        window.close()


def test_history_play_uses_displayed_snapshot_instead_of_other_loaded_audio():
    from dataclasses import replace
    from pathlib import Path
    from autoacoustics.analysis.pipeline import analyze
    from autoacoustics.model import AnalysisSettings, ChannelInfo, SignalData
    from autoacoustics.ui.main_window import MainWindow

    class Player:
        state = 'stopped'

        def stop(self):
            self.state = 'stopped'

        def set_segment(self, samples, rate, start=0, end=None, gain=1., **kwargs):
            self.samples = np.array(samples[start:end], copy=True)

        def play(self):
            self.state = 'playing'

    old = SignalData(np.full((1, 4800), .01), 48000, (ChannelInfo('old', 'FS'),),
                     Path('old.wav'), 'old-hash')
    new = SignalData(np.full((1, 4800), .02), 48000, (ChannelInfo('new', 'FS'),),
                     Path('new.wav'), 'new-hash')
    result = analyze(old, None, AnalysisSettings(channel=0, compute_loudness=False))
    window = MainWindow()
    window.player.stop()
    window.player = Player()
    try:
        window.set_signal(new)
        window.profile_combo.setCurrentIndex(0)
        window.results = [result]
        window._refresh_history()
        window.show_history_result(0)
        window.play_visible()
        np.testing.assert_array_equal(window.player.samples, old.samples[0])
        window.results = [replace(result, detail=None)]
        window._refresh_history()
        window.show_history_result(0)
        window.play_visible()
        assert window.player.state == 'stopped'
        assert '没有试听详情' in window.status_label.text()
    finally:
        window.close()
