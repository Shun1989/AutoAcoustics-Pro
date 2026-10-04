"""NI recording and HEAD stopped-file handoff, with explicit device availability."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import threading
from uuid import uuid4

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QTabWidget, QTableWidget, QVBoxLayout, QWidget)
from .window_geometry import fit_to_available_screen,scroll_page,flexible_wrapped_label

from ..acquisition.device import AcquisitionConfig, AcquisitionError, AcquisitionStatus, ChannelConfig
from ..acquisition.engine import RecordingEngine
from ..acquisition.head_recorder import HeadManagedRecorder, copy_completed_recording
from ..acquisition.ni_daqmx import NiDaqmxBackend, discover_ni
from ..acquisition.storage import open_acquisition_session, recover_session
from ..analysis.spl import level_from_power


STATE_TEXT = {AcquisitionStatus.IDLE: "未配置", AcquisitionStatus.CONFIGURED: "已配置",
    AcquisitionStatus.RECORDING: "录制中", AcquisitionStatus.FINALIZING: "停止排空／终结中",
    AcquisitionStatus.COMPLETE: "完整录制已保存", AcquisitionStatus.FAILED: "录制失败／不完整"}


class AcquisitionDialog(QDialog):
    recordingReady = pyqtSignal(object)
    _operationDone = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DAQ · NI 录制与 HEAD 设备文件交接")
        self.resize(1140, 810)
        self._engine = None
        self._busy = False
        self._closing_requested = False
        self._closed = False
        self._notified_manifest = None
        self._last_preview_index = None
        self._recovered_view = False
        self._curves = []
        self._geometry_fitted = False
        self._build()
        fit_to_available_screen(self,(1140,810))
        self._operationDone.connect(self._operation_done)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll_recording)
        self.timer.start()
        self.refresh_devices()

    def _build(self):
        outer = QVBoxLayout(self)
        self.stop_notice = QLabel("停止录制只停止记录，不切断电机电源。10 分钟验证指记录稳定性；座椅电机按允许工作制间歇动作。")
        self.stop_notice.setWordWrap(True)
        outer.addWidget(self.stop_notice)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        ni, head, ni_body = QWidget(), QWidget(), QWidget()
        self.head_page = head
        self.tabs.addTab(ni, "NI · 实时录制")
        self.head_scroll=scroll_page(head,'daqHeadScroll')
        self.tabs.addTab(self.head_scroll, "HEAD · 停止后设备文件交接")
        ni_outer=QVBoxLayout(ni)
        self.ni_scroll=scroll_page(ni_body,'daqNiScroll')
        ni_outer.addWidget(self.ni_scroll,1)
        layout = QVBoxLayout(ni_body)
        self.ni_config_group = QGroupBox("1 · 确认设备、逐通道参数与保存目录")
        config = QVBoxLayout(self.ni_config_group)
        layout.addWidget(self.ni_config_group)
        self.device_status = QLabel("正在只读检查 NI-DAQmx 驱动与设备…")
        self.device_status.setWordWrap(True)
        config.addWidget(self.device_status)
        row = QHBoxLayout()
        config.addLayout(row)
        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self._device_changed)
        self.refresh_button = QPushButton("刷新真实 NI 设备")
        self.refresh_button.clicked.connect(self.refresh_devices)
        row.addWidget(self.device_combo, 1)
        row.addWidget(self.refresh_button)
        self.channel_table = QTableWidget(0, 11)
        self.channel_table.setHorizontalHeaderLabels(["物理通道", "通道名称", "测量／源单位", "最小 V", "最大 V",
            "耦合", "激励来源", "电流 mA", "灵敏度 mV/Pa", "最大 SPL dB", "传感器／校准记录"])
        self.channel_table.setMinimumHeight(115)
        config.addWidget(self.channel_table)
        row = QHBoxLayout()
        config.addLayout(row)
        self.add_channel_button = QPushButton("添加通道配置")
        self.add_channel_button.clicked.connect(self.add_channel_row)
        self.remove_channel_button = QPushButton("移除选中行")
        self.remove_channel_button.clicked.connect(self.remove_channel_row)
        row.addWidget(self.add_channel_button)
        row.addWidget(self.remove_channel_button)
        unit_notice=QLabel("电压保留 V；麦克风由驱动给 Pa，须明确灵敏度与激励。")
        flexible_wrapped_label(unit_notice)
        row.addWidget(unit_notice,1)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        config.addLayout(form)
        self.requested_rate, self.target_frames = QLineEdit("48000"), QLineEdit()
        self.target_frames.setPlaceholderText("留空：连续录制至按停止；固定帧数请输入整数")
        self.output_directory = QLineEdit()
        row = QHBoxLayout()
        row.addWidget(self.output_directory, 1)
        choose = QPushButton("选择保存父目录")
        choose.clicked.connect(lambda: self._choose_directory(self.output_directory))
        row.addWidget(choose)
        form.addRow("请求采样率 / Hz（实际值由驱动回读）", self.requested_rate)
        form.addRow("有限目标帧数（可选）", self.target_frames)
        form.addRow("保存目录（自动建新会话，不覆盖旧数据）", row)
        self.condition, self.motion_direction, self.cw_reference, self.operator = (QLineEdit() for _ in range(4))
        form.addRow("工况 / 操作者", self._pair(self.condition, self.operator))
        form.addRow("实际运动方向 / CW-CCW 观察基准", self._pair(self.motion_direction, self.cw_reference))
        self.configure_button = QPushButton("核对并配置 NI")
        self.configure_button.clicked.connect(self.configure_recording)
        config.addWidget(self.configure_button)
        self.rate_status = QLabel("尚未提交设备配置；不会把请求 48 kHz 当成实际采样率。")
        self.rate_status.setWordWrap(True)
        layout.addWidget(self.rate_status)
        self.capability_status = QLabel("范围、耦合、激励与测量类型以设备属性及提交后的回读为准；未知字段会保留未知。")
        self.capability_status.setWordWrap(True)
        layout.addWidget(self.capability_status)
        row = QHBoxLayout()
        ni_outer.addLayout(row)
        self.start_button, self.stop_button = QPushButton("开始录制"), QPushButton("停止录制并排空")
        self.start_button.setObjectName('daqStart');self.stop_button.setObjectName('daqStop')
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.recover_button = QPushButton("恢复已提交数据（仅查看，不完整）")
        self.recover_button.clicked.connect(self.recover_partial)
        row.addWidget(self.start_button)
        row.addWidget(self.stop_button)
        row.addWidget(self.recover_button)
        self.event_group = QGroupBox("2 · 标注原样本事件")
        row = QHBoxLayout(self.event_group)
        self.event_kind, self.event_direction = QComboBox(), QComboBox()
        self.event_kind.addItems(["启动", "稳定", "停止", "异常"])
        self.event_direction.addItems(["未标记", "CW", "CCW"])
        self.event_note = QLineEdit()
        self.event_note.setPlaceholderText("事件备注；方向与观察基准分开保存")
        event = QPushButton("添加事件")
        event.clicked.connect(self.add_event)
        for widget in (self.event_kind, self.event_direction, self.event_note, event):
            row.addWidget(widget)
        layout.addWidget(self.event_group)
        self.recording_status = QLabel("未开始录制。")
        self.recording_status.setWordWrap(True)
        layout.addWidget(self.recording_status)
        self.preview_plot = pg.PlotWidget(title="原单位限长预览；完整数据独立落盘")
        self.preview_plot.setMinimumHeight(140)
        self.preview_plot.setLabel("bottom", "原录制时间", units="s")
        self.preview_plot.addLegend()
        layout.addWidget(self.preview_plot, 1)
        self.preview_metrics = QLabel("V 通道未校准，不显示绝对 SPL／响度；Pa 预览使用同一声级函数。")
        self.preview_metrics.setWordWrap(True)
        layout.addWidget(self.preview_metrics)
        head_layout = QVBoxLayout(head)
        self.head_control_status = QLabel(HeadManagedRecorder().discover().reason)
        self.head_control_status.setWordWrap(True)
        head_layout.addWidget(self.head_control_status)
        hint = QLabel("此入口只接收已停止的设备／官方 Recorder 文件。源件只读，经稳定性、哈希与完整载荷检查后保存到新会话；不代表程序直接控制 HEAD。")
        hint.setWordWrap(True)
        head_layout.addWidget(hint)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        head_layout.addLayout(form)
        self.head_source, self.head_directory = QLineEdit(), QLineEdit()
        row = QHBoxLayout()
        row.addWidget(self.head_source, 1)
        choose = QPushButton("选择单个设备录制文件")
        choose.clicked.connect(self._choose_head_source)
        row.addWidget(choose)
        form.addRow("用户指定文件（不扫描整盘）", row)
        row = QHBoxLayout()
        row.addWidget(self.head_directory, 1)
        choose = QPushButton("选择保存父目录")
        choose.clicked.connect(lambda: self._choose_directory(self.head_directory))
        row.addWidget(choose)
        form.addRow("保存副本与会话", row)
        self.head_model, self.head_firmware, self.head_operator, self.head_condition = (QLineEdit() for _ in range(4))
        form.addRow("设备型号 / 固件（未知可留空）", self._pair(self.head_model, self.head_firmware))
        form.addRow("操作者 / 工况", self._pair(self.head_operator, self.head_condition))
        self.head_stopped = QCheckBox("我已在设备或官方 Recorder 中停止录制，并确认该文件已完成")
        head_layout.addWidget(self.head_stopped)
        self.head_import_button = QPushButton("校验并交接完整设备文件")
        self.head_import_button.clicked.connect(self.import_head_recording)
        head_layout.addWidget(self.head_import_button)
        self.head_file_status = QLabel("尚未交接设备文件。")
        self.head_file_status.setWordWrap(True)
        head_layout.addWidget(self.head_file_status)
        head_layout.addStretch()
        self.message = QLabel()
        self.message.setWordWrap(True)
        message_scroll=scroll_page(self.message,'daqMessageScroll');message_scroll.setMinimumHeight(24);message_scroll.setMaximumHeight(72)
        outer.addWidget(message_scroll)
        close = QPushButton("关闭（活动录制先停止排空）")
        close.setObjectName('daqClose')
        close.clicked.connect(self.close)
        outer.addWidget(close)
        for label in self.findChildren(QLabel):
            if label.wordWrap():flexible_wrapped_label(label)
        for field in (self.requested_rate, self.target_frames, self.output_directory, self.condition,
                      self.motion_direction, self.cw_reference, self.operator):
            field.textChanged.connect(self._configuration_changed)
        self._enable_controls()

    def showEvent(self,event):
        super().showEvent(event)
        if not self._geometry_fitted:
            fit_to_available_screen(self,(1140,810));self._geometry_fitted=True

    @staticmethod
    def _pair(first, second):
        row = QHBoxLayout()
        row.addWidget(first)
        row.addWidget(second)
        return row

    def _choose_directory(self, field):
        path = QFileDialog.getExistingDirectory(self, "选择录制保存父目录")
        if path:
            field.setText(path)

    def _choose_head_source(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择已停止的设备录制文件", "",
            "已支持录制文件 (*.hdf *.h5 *.hdf5 *.wav *.flac *.tdms);;全部文件 (*)")
        if path:
            self.head_source.setText(path)

    def _active(self):
        return bool(self._engine and self._engine.snapshot().status in
                    {AcquisitionStatus.RECORDING, AcquisitionStatus.FINALIZING})

    def _enable_controls(self):
        active = self._active()
        editable = not self._busy and not active
        self.ni_config_group.setEnabled(editable)
        self.channel_table.setEnabled(editable)
        self.requested_rate.setEnabled(editable)
        self.configure_button.setEnabled(editable and self.device_combo.currentData() is not None)
        self.add_channel_button.setEnabled(editable and self.device_combo.currentData() is not None)
        configured = bool(self._engine and self._engine.snapshot().status == AcquisitionStatus.CONFIGURED)
        self.start_button.setEnabled(not self._busy and configured)
        self.stop_button.setEnabled(active)
        self.event_group.setEnabled(active and self._engine.snapshot().status == AcquisitionStatus.RECORDING)
        failed = bool(self._engine and self._engine.snapshot().status == AcquisitionStatus.FAILED
                      and self._engine.snapshot().frames_written > 0)
        self.recover_button.setEnabled(editable and failed)
        self.head_import_button.setEnabled(editable)
        self.head_page.setEnabled(editable)
        self.tabs.setTabEnabled(1, not active)

    def _run_operation(self, kind, function):
        if self._busy:
            return
        self._busy = True
        self._enable_controls()
        def run():
            try:
                value = function()
                payload = {"kind": kind, "value": value}
            except Exception as error:
                payload = {"kind": kind, "error": str(error)}
            if not self._closed:
                self._operationDone.emit(payload)
        self._operation_thread = threading.Thread(target=run, name=f"DAQ-UI-{kind}", daemon=True)
        self._operation_thread.start()

    def refresh_devices(self):
        if self._active() or self._busy:
            return
        self.device_status.setText("正在只读检查真实 NI 驱动与设备…")
        self._run_operation("discovery", discover_ni)

    def _device_changed(self):
        self._configuration_changed()
        self.channel_table.setRowCount(0)
        self._enable_controls()

    def _configuration_changed(self):
        if self._busy or self._active() or not self._engine:
            return
        if self._engine.snapshot().status == AcquisitionStatus.CONFIGURED:
            try:
                self._engine.backend.close()
            except Exception as error:
                self.message.setText(f"旧配置关闭返回错误：{error}")
            self._engine = None
            self.rate_status.setText("配置已改变，请重新核对并提交设备配置，再开始录制。")
            self._enable_controls()

    def add_channel_row(self):
        device = self.device_combo.currentData()
        if device is None or self._active():
            return
        row = self.channel_table.rowCount()
        self.channel_table.insertRow(row)
        physical, kind, coupling, excitation = (QComboBox() for _ in range(4))
        physical.addItem("明确选择通道", None)
        for channel in device.physical_channels:
            physical.addItem(channel, channel)
        kind.addItem("选择测量类型", None)
        kind.addItem("电压 / V", "voltage")
        kind.addItem("麦克风 / Pa（驱动缩放）", "microphone")
        for title, value in [("设备原设置", None), ("AC", "AC"), ("DC", "DC")]:
            coupling.addItem(title, value)
        for title, value in [("无内部激励", "none"), ("内部 IEPE", "internal"), ("外部激励", "external")]:
            excitation.addItem(title, value)
        widgets = [physical, QLineEdit(), kind, QLineEdit(), QLineEdit(), coupling, excitation,
                   QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()]
        for col, widget in enumerate(widgets):
            self.channel_table.setCellWidget(row, col, widget)
            (widget.currentIndexChanged if isinstance(widget, QComboBox) else widget.textChanged).connect(
                self._configuration_changed)
        self.channel_table.resizeColumnsToContents()
        self._configuration_changed()

    def remove_channel_row(self):
        if not self._active() and self.channel_table.currentRow() >= 0:
            self.channel_table.removeRow(self.channel_table.currentRow())
            self._configuration_changed()

    def _collect_configuration(self):
        channels = []
        for row in range(self.channel_table.rowCount()):
            fields = [self.channel_table.cellWidget(row, col) for col in range(11)]
            physical, name, kind = fields[0].currentData(), fields[1].text().strip(), fields[2].currentData()
            if physical is None or not name or kind is None:
                raise AcquisitionError("每行须明确物理通道、名称和测量类型。")
            def number(col):
                text = fields[col].text().strip()
                return float(text) if text else None
            channels.append(ChannelConfig(physical, name, "V" if kind == "voltage" else "Pa", kind,
                input_min=number(3), input_max=number(4), coupling=fields[5].currentData(),
                excitation_source=fields[6].currentData(),
                excitation_current_a=number(7)/1000 if number(7) is not None else None,
                sensitivity_mv_pa=number(8), max_sound_pressure_db=number(9),
                sensor={"operator_record": fields[10].text().strip()},
                calibration_snapshot={"source": "operator_configuration", "record": fields[10].text().strip()}))
        target = self.target_frames.text().strip()
        return AcquisitionConfig(tuple(channels), float(self.requested_rate.text()),
            target_frames=int(target) if target else None,
            measurement_context={"condition": self.condition.text().strip(), "operator": self.operator.text().strip(),
                "motion_direction": self.motion_direction.text().strip(), "cw_reference": self.cw_reference.text().strip()})

    @staticmethod
    def _new_directory(text, prefix):
        if not text.strip():
            raise AcquisitionError("请选择保存父目录。")
        parent = Path(text).expanduser().resolve()
        if not parent.is_dir():
            raise AcquisitionError("保存父目录不存在。")
        return parent / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"

    def configure_recording(self):
        if self._busy or self._active():
            return
        try:
            config = self._collect_configuration()
            directory = self._new_directory(self.output_directory.text(), "NI")
            if self._engine:
                self._engine.backend.close()
            self._engine = RecordingEngine(NiDaqmxBackend(), config, directory)
            self.message.setText("正在配置真实 NI，回读实际采样率与参数…")
            self._run_operation("configure", self._engine.configure)
        except Exception as error:
            self.message.setText(f"配置失败：{error}")
            self._enable_controls()

    def start_recording(self):
        if not self._engine or self._busy or self._active():
            return
        self._notified_manifest = None
        self._last_preview_index = None
        self._recovered_view = False
        self._run_operation("start", self._engine.start)

    def stop_recording(self):
        if self._active():
            self._engine.request_stop()
            self.recording_status.setText("已请求停止：正在停止设备并排空，完成前不会作为成功数据导入。")

    def add_event(self):
        try:
            direction = self.event_direction.currentText()
            event = self._engine.mark_event(self.event_kind.currentText(),
                direction=None if direction == "未标记" else direction, note=self.event_note.text().strip())
            self.message.setText(f"已保存事件 {event.kind}：样本 {event.sample_index}，{event.time_seconds:.6f} s。")
        except Exception as error:
            self.message.setText(f"事件未保存：{error}")

    def recover_partial(self):
        if self._busy or not self._engine:
            return
        path = self._engine.snapshot().manifest_path
        def recover():
            recover_session(path)
            return open_acquisition_session(path, allow_partial=True)
        self._run_operation("recovery", recover)

    def import_head_recording(self):
        if self._busy or self._active():
            return
        try:
            if not self.head_stopped.isChecked():
                raise AcquisitionError("请先确认已在设备或官方 Recorder 中停止录制。")
            directory = self._new_directory(self.head_directory.text(), "HEAD")
            source = Path(self.head_source.text())
            info = {"model": self.head_model.text().strip() or "unknown",
                    "firmware": self.head_firmware.text().strip() or "unknown"}
            operator, condition = self.head_operator.text().strip(), self.head_condition.text().strip()
            self.head_file_status.setText("正在确认文件稳定性、只读复制、哈希与完整载荷…")
            self._run_operation("head", lambda: copy_completed_recording(source, directory, device_stopped=True,
                device_info=info, operator=operator, measurement_context={"condition": condition}))
        except Exception as error:
            self.head_file_status.setText(f"设备文件未导入：{error}")

    def _operation_done(self, payload):
        self._busy = False
        kind = payload["kind"]
        if "error" in payload:
            text = f"操作失败：{payload['error']}"
            self.message.setText(text)
            if kind == "head":
                self.head_file_status.setText(text)
        else:
            value = payload.get("value")
            if kind == "discovery":
                self.device_combo.clear()
                for device in value.devices:
                    self.device_combo.addItem(f"{device.name} · {device.model or '型号未知'}" +
                                              ("（模拟测试）" if device.is_simulated else ""), device)
                self.device_status.setText(value.reason if not value.available or not value.devices else
                    f"已发现 {len(value.devices)} 台 NI；Python 包 {value.python_package_version or '未知'}，"
                    f"驱动 {value.driver_version or '未知'}。参数支持以此设备提交后的回读为准。")
            elif kind == "configure":
                self.rate_status.setText(f"请求 {value.requested_sample_rate:g} Hz；实际 {value.actual_sample_rate:g} Hz；"
                    f"{len(value.channels)} 通道 · {'模拟测试，非真机' if value.device.is_simulated else value.device.model or '型号未知'}。")
                settings = value.metadata.get("channel_settings", ())
                actual = []
                for channel in settings:
                    actual.append(f"{channel['name']} [{channel['unit']}]：范围 "
                        f"{channel.get('input_min_actual', '未知')}–{channel.get('input_max_actual', '未知')}；"
                        f"耦合 {channel.get('coupling_actual') or '未知'}；"
                        f"激励 {channel.get('excitation_source_actual') or '未知'}，"
                        f"{channel.get('excitation_current_a_actual')} A")
                self.capability_status.setText("实际通道回读：" + ("；".join(actual) or "此测试后端无硬件参数，不能当作真机配置。"))
                self.message.setText(f"配置已回读。将保存至 {self._engine.directory}；原单位／传感器配置写入会话。")
            elif kind == "head":
                self.head_file_status.setText(f"文件交接完成：{value.frames_written} 帧，{value.actual_sample_rate:g} Hz；"
                                              f"会话 {value.manifest_path}。此流程没有直接控制 HEAD。")
                self.recordingReady.emit(value.manifest_path)
            elif kind == "recovery":
                self._recovered_view = True
                self.recording_status.setText("已恢复已提交样本（不完整，仅原单位查看）；未发送完整测量，不生成报告。")
                self._draw_samples(value.samples[:, -48000:], value.sample_rate,
                                   max(0, value.frames-48000), value.channels)
        self._enable_controls()
        if self._closing_requested:
            self.close()

    def _draw_samples(self, samples, rate, first, channels):
        if samples.shape[1] == 0:
            return
        if len(self._curves) != len(channels):
            self.preview_plot.clear()
            self._curves = [self.preview_plot.plot(pen=pg.intColor(index), name=f"{channel.name} [{channel.unit}]")
                            for index, channel in enumerate(channels)]
        time = (first+np.arange(samples.shape[1])) / rate
        levels = []
        for index, channel in enumerate(channels):
            self._curves[index].setData(time, samples[index])
            power = float(np.mean(samples[index]**2))
            if channel.unit == "Pa":
                level = float(level_from_power(power))
                levels.append(f"{channel.name}：预览段 LZeq {level:.3g} dB SPL（源 Pa）")
            else:
                levels.append(f"{channel.name}：RMS {np.sqrt(power):.4g} {channel.unit}，未作声压换算")
        self.preview_metrics.setText("；".join(levels))

    def poll_recording(self):
        if not self._engine:
            return
        result = self._engine.snapshot()
        if result.status in {AcquisitionStatus.RECORDING, AcquisitionStatus.FINALIZING}:
            expected = "待停止确认" if result.frames_expected is None else str(result.frames_expected)
            self.recording_status.setText(f"{STATE_TEXT[result.status]} · 设备累计 {expected}；已读 {result.frames_read}；"
                f"已写 {result.frames_written} 帧；实际 {result.actual_sample_rate:g} Hz。")
            preview = self._engine.preview()
            marker = (preview.first_sample_index, preview.samples.shape[1])
            if marker != self._last_preview_index:
                self._draw_samples(preview.samples, preview.sample_rate, preview.first_sample_index,
                                   self._engine.configured.channels)
                self._last_preview_index = marker
        elif result.status == AcquisitionStatus.COMPLETE:
            self.recording_status.setText(f"完整录制已保存 · 设备/已读/已写 {result.frames_written} 帧；"
                                         f"{result.manifest_path}")
            if self._notified_manifest != result.manifest_path:
                self._notified_manifest = result.manifest_path
                self.recordingReady.emit(result.manifest_path)
        elif result.status == AcquisitionStatus.FAILED and not self._busy:
            prefix = "已恢复查看（不完整）" if self._recovered_view else "录制失败／不完整"
            self.recording_status.setText(f"{prefix} · 已读 {result.frames_read}；已写 {result.frames_written} 帧；"
                f"{'；'.join(result.flags)}；{'；'.join(result.errors)}。会话 {result.manifest_path}")
        self._enable_controls()
        if self._closing_requested and not self._busy and not self._active():
            self.close()

    def closeEvent(self, event):
        if self._busy or self._active():
            self._closing_requested = True
            if self._active():
                self._engine.request_stop()
            self.message.setText("正在结束活动操作并排空录制；数据终结后关闭。")
            event.ignore()
            return
        self._cleanup()
        event.accept()

    def _cleanup(self):
        if self._closed:
            return
        if self._engine:
            try:
                self._engine.backend.close()
            except Exception as error:
                self.message.setText(f"设备关闭返回错误：{error}")
        self._closed = True
        self.timer.stop()

    def reject(self):
        if self._busy or self._active():
            self.close()
        else:
            self._cleanup()
            super().reject()

    def accept(self):
        if self._busy or self._active():
            self.close()
        else:
            self._cleanup()
            super().accept()
