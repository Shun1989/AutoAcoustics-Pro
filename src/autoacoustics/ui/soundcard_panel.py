"""Computer/USB audio capture with bounded preview and durable recording handoff."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import math
from uuid import uuid4

import numpy as np
import pyqtgraph as pg
from scipy.signal import stft
from PyQt6.QtCore import QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QComboBox, QLineEdit, QPushButton, QFormLayout,
    QDoubleSpinBox, QFileDialog, QDialog, QScrollArea,QSizePolicy)

from autoacoustics.acquisition.device import AcquisitionConfig, AcquisitionStatus, ChannelConfig,AcquisitionError
from autoacoustics.acquisition.engine import RecordingEngine
from autoacoustics.acquisition.soundcard import (SoundCardBackend, discover_soundcards,
                                                export_soundcard_wav)
from .workbench import WORKBENCH_STYLE


class _AudioTask(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function

    def run(self):
        try:
            self.succeeded.emit(self.function())
        except Exception as error:
            self.failed.emit(str(error))


class SoundCardPanel(QWidget):
    recordingReady = pyqtSignal(object)

    def __init__(self, parent=None, *, discovery_fn=None, backend_factory=None,
                 wav_exporter=None):
        super().__init__(parent)
        self.setObjectName('soundCardPanel')
        self.setStyleSheet(WORKBENCH_STYLE)
        self._discovery_fn = discovery_fn or discover_soundcards
        self._backend_factory = backend_factory or SoundCardBackend
        self._wav_exporter = wav_exporter or export_soundcard_wav
        self._engine = self._starter = self._exporter = None
        self._stop_requested = self._closing = False
        self._wav_started = False
        self._preferred_device_identity = None
        self._preferred_channels = self._preferred_rate = None
        self._emitted = set()
        self.completed_manifest = self.completed_wav = None
        self.preview_frames = 0
        self.preview_meters = {}
        self._last_plot = {}
        self._dialogs = []
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.poll_recording)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(.3)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self._playback_position)
        self.player.errorOccurred.connect(self._playback_error)
        self.refresh_devices()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        heading = QLabel('电脑麦克风 / USB 声卡 · 录制与实时预览')
        heading.setProperty('role', 'sectionTitle')
        layout.addWidget(heading)
        notice = QLabel('源数据为 FS（数字满量程），未校准；不会显示绝对声压或响度。完整录音独立写盘，实时预览只取最近 3 秒；不实时回放到扬声器。')
        notice.setWordWrap(True)
        notice.setMaximumHeight(44)
        notice.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred)
        layout.addWidget(notice)
        grid = QGridLayout()
        layout.addLayout(grid, 1)
        self.wave_plot = self._plot_card(grid, 0, 0, '原始波形 / FS', 'wave')
        self.spectrum_plot = self._plot_card(grid, 0, 1, '峰值幅度谱 / dBFS', 'spectrum')
        self.spectrogram_plot = self._plot_card(grid, 1, 0, '滚动语谱图 / dBFS', 'spectrogram')
        self.wave_plot.setLabel('bottom', '录制原时间', units='s')
        self.wave_plot.setLabel('left', '幅度', units='FS')
        self.spectrum_plot.setLabel('bottom', '频率', units='Hz')
        self.spectrum_plot.setLabel('left', '峰值幅度', units='dBFS')
        self.spectrogram_plot.setLabel('bottom', '录制原时间', units='s')
        self.spectrogram_plot.setLabel('left', '频率', units='Hz')
        self.wave_curve = self.wave_plot.plot(pen=pg.mkPen('#64d4e3', width=1))
        self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen('#7fdfac', width=1))
        self.image = pg.ImageItem(axisOrder='row-major')
        self.spectrogram_plot.addItem(self.image)
        self.image.setLookupTable(pg.colormap.get('viridis').getLookupTable())
        control = QGroupBox('采集 / 保存 / 工况')
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(control)
        grid.addWidget(scroll, 1, 1)
        form = QFormLayout(control)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.device_combo, self.channel_combo = QComboBox(), QComboBox()
        self.device_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.device_combo.setMinimumContentsLength(12)
        self.device_combo.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Fixed)
        self.device_combo.setObjectName('soundcardDevice')
        self.device_combo.currentIndexChanged.connect(self._device_changed)
        self.rate_combo = QComboBox()
        self.rate_combo.setEditable(True)
        self.rate_combo.addItems(['48000', '44100', '96000', '32000', '16000'])
        self.rate_combo.setToolTip('这是请求值；开始时实际检查设备支持并回读采样率。')
        refresh = QPushButton('刷新真实输入设备')
        refresh.clicked.connect(self.refresh_devices)
        self.refresh_button = refresh
        form.addRow('设备 / Host API', self.device_combo)
        form.addRow('', refresh)
        form.addRow('输入通道（明确选择）', self.channel_combo)
        form.addRow('请求采样率 / Hz', self.rate_combo)
        self.target_duration = QDoubleSpinBox()
        self.target_duration.setRange(0, 86400)
        self.target_duration.setDecimals(3)
        self.target_duration.setSpecialValueText('连续，按停止结束')
        self.target_duration.setSuffix(' s')
        form.addRow('录制时长（0 为连续）', self.target_duration)
        self.save_parent = QLineEdit(str(Path.home()/'Documents'/'AutoAcoustics Pro'/'Recordings'))
        choose = QPushButton('选择目录')
        choose.clicked.connect(self._choose_directory)
        row = QHBoxLayout()
        row.addWidget(self.save_parent, 1)
        row.addWidget(choose)
        self.choose_button = choose
        form.addRow('保存父目录', row)
        self.condition_edit = QLineEdit()
        self.condition_edit.setPlaceholderText('负载、姿态、动作等已知工况')
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(['未知', 'CW', 'CCW'])
        self.direction_reference = QLineEdit()
        self.direction_reference.setPlaceholderText('观察端 / 旋向基准；不要从文件名推断')
        form.addRow('工况', self.condition_edit)
        form.addRow('电机旋向', self.direction_combo)
        form.addRow('旋向观察基准', self.direction_reference)
        self.start_button, self.stop_button = QPushButton('开始录制'), QPushButton('停止并保存')
        self.start_button.setProperty('role', 'primary')
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        row = QHBoxLayout()
        row.addWidget(self.start_button)
        row.addWidget(self.stop_button)
        layout.insertLayout(2,row)
        self.event_kind = QComboBox()
        self.event_kind.addItems(['启动', '稳定', '停止', '异常'])
        self.event_note = QLineEdit()
        self.event_note.setPlaceholderText('事件备注')
        self.event_button = QPushButton('标记当前帧')
        self.event_button.clicked.connect(self.mark_event)
        row = QHBoxLayout()
        row.addWidget(self.event_kind)
        row.addWidget(self.event_note, 1)
        row.addWidget(self.event_button)
        form.addRow('原样本事件', row)
        self.retry_wav_button = QPushButton('重试生成标准 WAV')
        self.retry_wav_button.clicked.connect(self._start_wav_export)
        form.addRow(self.retry_wav_button)
        self.send_button = QPushButton('送入分析（会话 + 原始 FS）')
        self.send_button.clicked.connect(self.send_to_analysis)
        form.addRow(self.send_button)
        self.listen_start, self.listen_end = QDoubleSpinBox(), QDoubleSpinBox()
        for field in (self.listen_start, self.listen_end):
            field.setRange(0, 86400)
            field.setDecimals(3)
            field.setSuffix(' s')
        row = QHBoxLayout()
        row.addWidget(self.listen_start)
        row.addWidget(self.listen_end)
        form.addRow('保存后试听区间', row)
        self.listen_button = QPushButton('试听保存的 WAV')
        self.listen_button.clicked.connect(self.listen_recording)
        stop_listen = QPushButton('停止试听')
        stop_listen.clicked.connect(lambda:self.player.stop())
        row = QHBoxLayout()
        row.addWidget(self.listen_button)
        row.addWidget(stop_listen)
        form.addRow(row)
        self.metrics_label = QLabel('Peak / RMS dBFS：—；FS 未校准')
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setMaximumHeight(44)
        self.metrics_label.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred)
        layout.addWidget(self.metrics_label)
        self.status_label = QLabel('尚未开始录制。')
        self.status_label.setMaximumHeight(44)
        self.status_label.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred)
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(self.status_label.textInteractionFlags())
        layout.addWidget(self.status_label)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        self._config_controls = (self.device_combo, self.channel_combo, self.rate_combo,
            self.target_duration, self.save_parent, self.choose_button, self.refresh_button,
            self.condition_edit, self.direction_combo, self.direction_reference)

    def _plot_card(self, grid, row, column, title, key):
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        plot = pg.PlotWidget(background='#101018')
        plot.setMinimumSize(180, 105)
        layout.addWidget(plot, 1)
        button = QPushButton('放大图表')
        button.clicked.connect(lambda:self._expand(key, title))
        layout.addWidget(button)
        grid.addWidget(group, row, column)
        return plot

    def _expand(self, key, title):
        dialog = QDialog(self)
        dialog.setWindowTitle(title+' · 预览快照')
        dialog.resize(960, 620)
        dialog.setStyleSheet(WORKBENCH_STYLE)
        layout = QVBoxLayout(dialog)
        plot = pg.PlotWidget(background='#101018')
        layout.addWidget(plot)
        value = self._last_plot.get(key)
        if value is not None:
            if key == 'spectrogram':
                data, rect = value
                item = pg.ImageItem(data, axisOrder='row-major')
                item.setLookupTable(pg.colormap.get('viridis').getLookupTable())
                item.setLevels([-100, 0])
                item.setRect(rect)
                plot.addItem(item)
            else:
                plot.plot(value[0], value[1], pen='#64d4e3')
        self._dialogs.append(dialog)
        dialog.finished.connect(lambda:self._dialogs.remove(dialog) if dialog in self._dialogs else None)
        dialog.show()

    def refresh_devices(self):
        if self.is_recording:
            return
        previous = self.device_combo.currentData()
        identity = self._device_identity(previous) if previous else self._preferred_device_identity
        if previous is not None:
            self._preferred_channels = self.channel_combo.currentData()
            self._preferred_rate = self.rate_combo.currentText()
        selected_channels, requested_rate = self._preferred_channels, self._preferred_rate
        try:
            result = self._discovery_fn()
        except Exception as error:
            self.status_label.setText(f'无法枚举输入：{error}；请检查麦克风权限、USB 连接和 Windows 音频服务。')
            result = None
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        devices = result.devices if result and result.available else ()
        for device in devices:
            meta = device.metadata
            default = ' · 默认' if meta.get('is_default_input') else ''
            self.device_combo.addItem(f"{meta.get('host_api', 'Host API 未知')} · {device.name}{default}", device)
        if devices:
            if identity is not None:
                matches = [i for i, device in enumerate(devices) if self._device_identity(device) == identity]
                selected = matches[0] if len(matches) == 1 else -1
            else:
                selected = next((i for i,d in enumerate(devices) if 'WASAPI' in str(d.metadata.get('host_api', '')).upper()),
                    next((i for i,d in enumerate(devices) if d.metadata.get('is_default_input')), 0))
            self.device_combo.setCurrentIndex(selected)
            self.status_label.setText('已列出真实输入。采样率须在开始时通过设备检查；声卡 FS 不等于 Pa。'
                if selected >= 0 else '此前选择的输入已不可用或出现同名设备，无法确认测量来源；请重新选择输入设备。')
        elif result:
            self.status_label.setText(result.reason or '没有可用音频输入；请连接麦克风并开启 Windows 麦克风权限。')
        self.device_combo.blockSignals(False)
        self._device_changed()
        if identity is not None and self.device_combo.currentData() is not None:
            channel_index = next((i for i in range(self.channel_combo.count())
                if self.channel_combo.itemData(i) == selected_channels), -1)
            if channel_index >= 0:
                self.channel_combo.setCurrentIndex(channel_index)
            if requested_rate is not None:
                self.rate_combo.setCurrentText(requested_rate)
        self._enable_controls()

    @staticmethod
    def _device_identity(device):
        # PortAudio indices may change after reconnect. It exposes no portable
        # serial number, so identical names/API/channel counts are ambiguous.
        return (device.name, str(device.metadata.get('host_api', '')),
                int(device.metadata.get('max_input_channels', len(device.physical_channels))))

    def _device_changed(self, *unused):
        self.channel_combo.clear()
        device = self.device_combo.currentData()
        if device is None:
            self._enable_controls()
            return
        self._preferred_device_identity = self._device_identity(device)
        count = int(device.metadata.get('max_input_channels', len(device.physical_channels)))
        for channel in range(count):
            self.channel_combo.addItem(f'输入 {channel+1}（index {channel}）', (channel,))
        if count >= 2:
            self.channel_combo.addItem('双通道：输入 1 + 2', (0, 1))
        self.rate_combo.setCurrentText(str(int(float(device.metadata.get('default_samplerate', 48000)))))
        self._enable_controls()

    def _choose_directory(self):
        directory = QFileDialog.getExistingDirectory(self, '选择录制保存父目录', self.save_parent.text())
        if directory:
            self.save_parent.setText(directory)

    @property
    def is_recording(self):
        # Keep busy until queued success/finalization callbacks are consumed.
        return bool(self._starter is not None or self._exporter is not None or
                    (self._engine and self._engine.snapshot().status in
                     {AcquisitionStatus.RECORDING, AcquisitionStatus.FINALIZING}) or
                    (self._engine and self._engine.snapshot().status == AcquisitionStatus.COMPLETE
                     and not self._wav_started))

    def _enable_controls(self):
        active = self.is_recording
        for control in self._config_controls:
            control.setEnabled(not active)
        self.start_button.setEnabled(not active and self.device_combo.currentData() is not None and not self._closing)
        self.stop_button.setEnabled(active and not self._stop_requested)
        self.event_button.setEnabled(bool(self._engine and self._engine.snapshot().status == AcquisitionStatus.RECORDING))
        self.send_button.setEnabled(bool(not active and self.completed_manifest and self.completed_wav
                                         and self.completed_manifest not in self._emitted))
        self.listen_button.setEnabled(bool(not active and self.completed_wav))
        self.retry_wav_button.setEnabled(bool(not active and self.completed_manifest and not self.completed_wav))

    def start_recording(self):
        if self.is_recording or self._closing:
            return
        device, indices = self.device_combo.currentData(), self.channel_combo.currentData()
        if device is None or not indices:
            self.status_label.setText('没有可用输入或通道；请刷新设备并选择输入。')
            return
        try:
            rate = float(self.rate_combo.currentText())
            if not math.isfinite(rate) or rate <= 0 or rate != round(rate):
                raise ValueError('采样率须为有限正整数 Hz。')
            parent = Path(self.save_parent.text().strip()).expanduser()
            if not self.save_parent.text().strip():
                raise ValueError('请选择保存父目录。')
            channels = tuple(ChannelConfig(device.physical_channels[i], f'Input {i+1}',
                'FS', 'physical', -1., 1.) for i in indices)
            target = max(1,round(self.target_duration.value()*rate)) if self.target_duration.value() else None
            config = AcquisitionConfig(channels, rate, block_frames=2048, preview_frames=round(3*rate),
                target_frames=target, measurement_context={'motor_rotation':self.direction_combo.currentText(),
                    'rotation_view':self.direction_reference.text().strip() or None,
                    'notes':self.condition_edit.text().strip(), 'actual_movement':'unknown'})
            directory = parent/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8])
        except (ValueError, IndexError, TypeError,AcquisitionError) as error:
            self.status_label.setText(f'配置无效：{error}')
            return
        self.player.stop()
        self.wave_curve.clear()
        self.spectrum_curve.clear()
        self.image.clear()
        self._last_plot.clear()
        self.preview_frames = 0
        self.preview_meters = {}
        self.metrics_label.setText('等待当前录制数据 · Peak / RMS dBFS：—；FS 未校准')
        self.completed_manifest = self.completed_wav = None
        for field in (self.listen_start, self.listen_end):
            field.setRange(0, 86400)
            field.setValue(0)
        self._engine = None
        self._stop_requested = self._wav_started = False
        self.status_label.setText('正在检查设备 / 权限并建立原始录制会话…')
        def start():
            backend = self._backend_factory(int(device.metadata['device_index']), tuple(indices))
            engine = RecordingEngine(backend, config, directory)
            try:
                configured = engine.configure()
                if target is not None and not math.isclose(configured.actual_sample_rate, rate, rel_tol=0, abs_tol=1e-9):
                    raise AcquisitionError(
                        f'固定时长录制请求 {rate:g} Hz，但驱动实际回读 {configured.actual_sample_rate:g} Hz；'
                        f'为确保录制秒数正确，尚未开始采集。请将请求采样率改为 {configured.actual_sample_rate:g} Hz 后重试。',
                        flag='fixed_duration_clock_mismatch')
                engine.start()
                return engine
            except Exception:
                backend.close()
                raise
        self._starter = _AudioTask(start, self)
        self._starter.succeeded.connect(self._recording_started)
        self._starter.failed.connect(self._recording_failed)
        self._starter.finished.connect(self._starter_finished)
        self._starter.start()
        self.timer.start()
        self._enable_controls()

    def _recording_started(self, engine):
        self._engine = engine
        if self._stop_requested or self._closing:
            engine.request_stop()
        self.poll_recording()

    def _recording_failed(self, message):
        self.status_label.setText(f'录制未完成：{message}；请检查设备占用、Windows 麦克风权限、USB 连接及采样率。')
        self.timer.stop()

    def _starter_finished(self):
        if self._starter:
            self._starter.deleteLater()
            self._starter = None
        self._enable_controls()

    def stop_recording(self):
        self._stop_requested = True
        if self._engine:
            self._engine.request_stop()
        self.status_label.setText('停止请求已发送，正在排空原数据并核对样本计数；此操作不控制电机电源。')
        self._enable_controls()

    def poll_recording(self):
        if self._engine is None:
            self._enable_controls()
            return
        session = self._engine.snapshot()
        preview = self._engine.preview()
        if preview.samples.size:
            self._show_preview(preview)
        rate = session.actual_sample_rate or 0
        duration = session.frames_read/rate if rate else 0
        if session.status in {AcquisitionStatus.RECORDING, AcquisitionStatus.FINALIZING}:
            state = '排空 / 终结' if self._stop_requested or session.status == AcquisitionStatus.FINALIZING else '录制中'
            self.status_label.setText(f'{state} · 实际 {rate:g} Hz · {duration:.2f} s · 已读 {session.frames_read} / 已写 {session.frames_written} 帧')
        elif session.status == AcquisitionStatus.COMPLETE and not self._wav_started:
            self.completed_manifest = session.manifest_path
            self.listen_start.setRange(0, duration)
            self.listen_end.setRange(0, duration)
            self.listen_start.setValue(0)
            self.listen_end.setValue(duration)
            self._start_wav_export()
        elif session.status == AcquisitionStatus.FAILED:
            self.status_label.setText(f'录制不完整，不能送入完整分析：{"；".join(session.errors)}；质量标记 {", ".join(session.flags)}。保留数据：{session.manifest_path}')
            self.timer.stop()
        self._enable_controls()

    def _start_wav_export(self):
        if not self.completed_manifest or self._exporter:
            return
        self._wav_started = True
        path = self.completed_manifest
        self.status_label.setText(f'原始 HDF 会话已完成，正在生成标准 FLOAT WAV：{path}')
        self._exporter = _AudioTask(lambda:self._wav_exporter(path), self)
        self._exporter.succeeded.connect(self._wav_completed)
        self._exporter.failed.connect(self._wav_failed)
        self._exporter.finished.connect(self._exporter_finished)
        self._exporter.start()
        self._enable_controls()

    def _wav_completed(self, path):
        self.completed_wav = Path(path)
        self.status_label.setText(f'录制和计数核对完成。会话：{self.completed_manifest}\n标准 FLOAT WAV：{self.completed_wav}；可试听或明确送入分析。')
        self.timer.stop()

    def _wav_failed(self, message):
        self.status_label.setText(f'HDF 会话已保存，但 WAV 导出失败：{message}；请检查磁盘空间 / 权限后重试。尚未送入分析。')
        self.timer.stop()

    def _exporter_finished(self):
        if self._exporter:
            self._exporter.deleteLater()
            self._exporter = None
        self._enable_controls()

    def send_to_analysis(self):
        if self.is_recording or not self.completed_manifest or not self.completed_wav:
            return
        if self.completed_manifest not in self._emitted:
            self._emitted.add(self.completed_manifest)
            self.recordingReady.emit(self.completed_manifest)
        self._enable_controls()

    def mark_event(self):
        if self._engine is None:
            return
        try:
            event = self._engine.mark_event(self.event_kind.currentText(),
                direction=None if self.direction_combo.currentText() == '未知' else self.direction_combo.currentText(),
                note=self.event_note.text().strip())
            self.status_label.setText(f'已标记 {event.kind}，原样本 {event.sample_index} / {event.time_seconds:.3f} s')
        except Exception as error:
            self.status_label.setText(f'事件未保存：{error}')

    def _show_preview(self, preview):
        from PyQt6.QtCore import QRectF
        rate = preview.sample_rate
        if not rate or not preview.samples.size:
            return
        limit = max(1, round(3*rate))
        samples = preview.samples[:, -limit:]
        self.preview_frames = samples.shape[1]
        first = preview.first_sample_index+preview.samples.shape[1]-self.preview_frames
        x = samples[0]
        peak = float(np.max(np.abs(x)))
        rms = float(np.sqrt(np.mean(x**2)))
        peak_db = 20*np.log10(peak) if peak else -np.inf
        rms_db = 20*np.log10(rms) if rms else -np.inf
        self.preview_meters = {'peak_dbfs':float(peak_db), 'rms_dbfs':float(rms_db),
                               'clipped':bool(np.any(np.abs(samples) >= 1.))}
        self.metrics_label.setText(f'最近 {self.preview_frames/rate:.2f} s · 预览第 1 通道 / 共 {samples.shape[0]} 通道 · Peak {peak_db:.1f} dBFS · RMS {rms_db:.1f} dBFS · '+
                                  ('触轨 / 削波风险，请降低输入增益' if self.preview_meters['clipped'] else '未见数字触轨')+' · FS 未校准')
        # Display decimation keeps interval extrema; full raw recording is unaffected.
        from .plots import envelope_plot_data
        times, values = envelope_plot_data(x, rate, max_points=3000, origin=first/rate)
        self.wave_curve.setData(times, values)
        self._last_plot['wave'] = (times, values)
        count = min(len(x), 8192)
        if count < 2:
            return
        window = np.hanning(count+1)[:-1]
        transform = np.fft.rfft(x[-count:]*window)
        factors = np.full(len(transform), 2.)
        factors[0] = 1.
        if count % 2 == 0:
            factors[-1] = 1.
        amplitude = np.abs(transform)*factors/window.sum()
        db = 20*np.log10(np.maximum(amplitude, 1e-8))
        frequencies = np.fft.rfftfreq(count, 1/rate)
        self.spectrum_curve.setData(frequencies, db)
        self._last_plot['spectrum'] = (frequencies, db)
        segment = min(1024, len(x))
        if segment < 16:
            return
        frequencies, times, z = stft(x, fs=rate, nperseg=segment, noverlap=segment//2,
                                     boundary=None, padded=False, detrend=False)
        factors = np.full(len(frequencies), 2.)
        factors[0] = 1.
        if segment % 2 == 0:
            factors[-1] = 1.
        image = 20*np.log10(np.maximum(np.abs(z)*factors[:, None], 1e-8))
        times = times+first/rate
        step = segment/2/rate
        rect = QRectF(float(times[0]-step/2), 0, float(max(step, len(times)*step)), rate/2)
        self.image.setImage(image, autoLevels=False)
        self.image.setLevels([-100, 0])
        self.image.setRect(rect)
        self.spectrogram_plot.setXRange(rect.left(), rect.right(), padding=0)
        self.spectrogram_plot.setYRange(0, rate/2, padding=0)
        self._last_plot['spectrogram'] = (image, rect)

    def listen_recording(self):
        if not self.completed_wav or self.is_recording:
            return
        start, end = self.listen_start.value(), self.listen_end.value()
        if end <= start:
            self.status_label.setText('试听区间须满足结束 > 开始；单位为保存录音的秒数。')
            return
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(self.completed_wav)))
        self.player.setPosition(round(start*1000))
        self.player.play()

    def _playback_position(self, milliseconds):
        if self.listen_end.value() > 0 and milliseconds >= round(self.listen_end.value()*1000):
            self.player.stop()

    def _playback_error(self, error, message):
        if error != QMediaPlayer.Error.NoError:
            self.status_label.setText(f'试听失败：{message}；原录音和分析数据未改变。')

    def shutdown(self):
        self._closing = True
        if self.is_recording:
            self.stop_recording()
            return False
        self.timer.stop()
        self.player.stop()
        for dialog in tuple(self._dialogs):
            dialog.close()
        return True
