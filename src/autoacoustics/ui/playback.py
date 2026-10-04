"""Explicit-gain segment playback; sound output does not affect measurements."""
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, QBuffer, QByteArray, QIODevice, QTimer
from PyQt6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices, QAudio

def prepare_playback(samples, gain=1.):
    values = np.asarray(samples, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or not np.isfinite(gain) or gain <= 0:
        raise ValueError('试听需要有限的一维样本和正增益。')
    scaled = values * gain
    clipped = float(np.mean(np.abs(scaled) > 1))
    return np.clip(scaled, -1, 1).astype('<f4'), clipped

class SegmentPlayer(QObject):
    positionChanged = pyqtSignal(float)
    stateChanged = pyqtSignal(str)
    message = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sink = None
        self.buffer = None
        self.samples = None
        self.sample_rate = 48000
        self.origin = 0.
        self.state = 'stopped'
        self.gain = 1.
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._update_cursor)

    def set_segment(self, samples, sample_rate, start=0, end=None, gain=1.,time_origin=0.):
        self.stop()
        values = np.asarray(samples)
        end = len(values) if end is None else end
        if not 0 <= start < end <= len(values) or not np.isfinite(sample_rate) or sample_rate <= 0 or not np.isfinite(time_origin):
            raise ValueError('试听区间无效。')
        self.samples = np.array(values[start:end], copy=True)
        self.sample_rate = sample_rate
        self.origin = time_origin+start / sample_rate
        self.gain = gain

    def play(self):
        if self.state == 'paused' and self.sink:
            self.sink.resume()
            self.state = 'playing'
            self.timer.start()
            self.stateChanged.emit(self.state)
            return
        if self.samples is None:
            self.message.emit('请先选择录音和试听区间。')
            return
        device = QMediaDevices.defaultAudioOutput()
        if device.isNull():
            self.message.emit('没有可用播放设备；分析和报告仍可使用。')
            return
        format = QAudioFormat()
        format.setSampleRate(round(self.sample_rate))
        format.setChannelCount(1)
        format.setSampleFormat(QAudioFormat.SampleFormat.Float)
        if not device.isFormatSupported(format):
            self.message.emit('当前播放设备不支持此采样率；请更换系统播放设备。原音频未重采样。')
            return
        audio, clipped = prepare_playback(self.samples, self.gain)
        if clipped:
            self.message.emit(f'试听增益超范围，{clipped:.1%} 样本被限制；请降低增益。分析数据未改变。')
        self.buffer = QBuffer(self)
        self.buffer.setData(QByteArray(audio.tobytes()))
        self.buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        self.sink = QAudioSink(device, format, self)
        self.sink.stateChanged.connect(self._sink_state)
        self.sink.start(self.buffer)
        self.state = 'playing'
        self.timer.start()
        self.stateChanged.emit(self.state)

    def pause(self):
        if self.sink and self.state == 'playing':
            self.sink.suspend()
            self.state = 'paused'
            self.timer.stop()
            self.stateChanged.emit(self.state)

    def stop(self):
        self.timer.stop()
        if self.sink:
            self.sink.stop()
            self.sink.deleteLater()
            self.sink = None
        if self.buffer:
            self.buffer.close()
            self.buffer.deleteLater()
            self.buffer = None
        self.state = 'stopped'
        self.stateChanged.emit(self.state)

    def _update_cursor(self):
        if self.sink:
            self.positionChanged.emit(self.origin + self.sink.processedUSecs() / 1e6)

    def _sink_state(self, state):
        if state == QAudio.State.IdleState:
            self._update_cursor()
            self.timer.stop()
            self.state = 'stopped'
            self.stateChanged.emit(self.state)
