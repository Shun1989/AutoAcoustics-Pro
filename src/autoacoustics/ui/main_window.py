"""Six-step desktop measurement workflow using one shared analysis pipeline."""
from __future__ import annotations
from dataclasses import replace,fields
from collections.abc import Mapping
from pathlib import Path
import math
import hashlib
import numpy as np
from PyQt6.QtCore import Qt,QTimer,QSettings,QUrl,QStandardPaths
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,
    QGroupBox,QScrollArea,QSplitter,QLabel,QPushButton,QComboBox,QDoubleSpinBox,
    QCheckBox,QListWidget,QListWidgetItem,QTableWidget,QTableWidgetItem,QTabWidget,QFileDialog,QDialog,
    QAbstractItemView,QProgressBar,QSizePolicy,QLayout)
from ..model import (AnalysisSettings,MeasurementContext,MeasurementProject,ResultSnapshot,
    BatchItemSpec,ImportMapping,CalibrationProfile,ComparisonResult)
from ..jobs import JobExecutor
from ..project import save_project,load_project,load_result_detail,relink_source,source_states
from .plots import PlotPanel
from .playback import SegmentPlayer
from .formatting import metric_text,METRIC_LABELS,quality_text
from .context_dialog import ContextDialog,CONTEXT_LABELS
from .import_dialog import ImportDialog
from .calibration_dialog import CalibrationDialog
from .batch_panel import BatchPanel
from .workbench import install_workbench,refresh_metric_cards,restore_workbench,save_workbench,WORKBENCH_STYLE

EVENTS={'全事件':'full','启动':'startup','运行':'running','停止':'stop','异常':'anomaly'}

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AutoAcoustics Pro 0.3.1 · 现场采集与 NVH 工作台')
        self.resize(1480,940)
        self.signal=None;self.mapping=None;self.context=MeasurementContext()
        self._signal_channel_hashes={}
        self.results=[];self.project_items=[];self.reports=[];self.project_path=None
        self.current_result=None;self.current_result_valid=False;self.comparison=None
        self._view_comparison=False
        self.profiles=[];self._updating=False;self._expected_job=None;self._pending_import=None
        self._pending_project_result=None;self._pending_event=None;self._job_snapshot=None;self._project=None
        self._import_options=None;self._history_selected=False
        self._extension_dialogs={};self._close_requested=False
        self.soundcard_panel=None;self.rotation_panel=None;self._live_import_pending=False;self._live_import_path=None
        self.jobs=JobExecutor()
        self.preferences=QSettings('AutoAcoustics','AutoAcousticsPro')
        self.player=SegmentPlayer(self)
        self._playback_key=None
        self.player.message.connect(self.show_message)
        self.player.positionChanged.connect(lambda value:self.plots.cursor.setValue(value))
        self.player.stateChanged.connect(lambda state:self.playback_state.setText({'playing':'播放中','paused':'已暂停','stopped':'已停止'}.get(state,state)))
        self._build()
        self._restore_preferences()
        restore_workbench(self)
        geometry=self.preferences.value('geometry')
        if geometry:self.restoreGeometry(geometry)
        available=self.screen().availableGeometry()
        self.resize(min(self.width(),max(self.minimumSizeHint().width(),available.width()-48)),
            min(self.height(),max(self.minimumSizeHint().height(),available.height()-80)))
        self.move(max(available.left()+12,min(self.x(),available.right()-self.width()-12)),
            max(available.top()+24,min(self.y(),available.bottom()-self.height()-48)))
        self.timer=QTimer(self);self.timer.setInterval(100);self.timer.timeout.connect(self.poll_jobs);self.timer.start()
        self._load_reference_profile()
        self.setStyleSheet(WORKBENCH_STYLE)

    def _button(self,title,name,slot):
        button=QPushButton(title);button.setObjectName(name);button.clicked.connect(slot);return button
    def _group(self,title):
        group=QGroupBox(title);layout=QVBoxLayout(group);self.controls.addWidget(group);return layout
    def _row(self,layout,*widgets):
        row=QHBoxLayout();layout.addLayout(row)
        for widget in widgets:row.addWidget(widget)
    def _build(self):
        central=QWidget();outer=QVBoxLayout(central);self.setCentralWidget(central)
        top=QHBoxLayout();outer.addLayout(top)
        heading=QLabel('AutoAcoustics Pro');heading.setStyleSheet('font-size:23px;font-weight:600;color:#65d3db;');top.addWidget(heading)
        top.addWidget(QLabel('文件 → 来源/单位 → 校准 → 事件 → 分析/比较 → 保存/报告'));top.addStretch()
        top.addWidget(self._button('采集 / HEAD 文件交接','daq_entry',self.open_acquisition))
        top.addWidget(self._button('知识问答','knowledge_entry',self.open_knowledge))
        self.status_label=QLabel('打开一份录音开始；ZIP 请先解压。');self.status_label.setObjectName('workflow_status');self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)
        splitter=QSplitter();outer.addWidget(splitter,1)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setMinimumWidth(340);scroll.setMaximumWidth(460)
        side=QWidget();self.controls=QVBoxLayout(side);scroll.setWidget(side);splitter.addWidget(scroll)
        side.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred)
        self.controls.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        group=self._group('1 · 导入录音')
        self._row(group,self._button('打开录音','open_file',self.choose_file),self._button('打开项目','open_project',self.choose_project))
        self.file_label=QLabel('尚未导入');self.file_label.setObjectName('source_path');self.file_label.setWordWrap(True);group.addWidget(self.file_label)
        group.addWidget(self._button('修改导入映射','edit_mapping',self.edit_mapping))
        group=self._group('2 · 确认通道和单位')
        self.channel_combo=QComboBox();self.channel_combo.setObjectName('channel_selector');group.addWidget(self.channel_combo)
        self.unit_label=QLabel('必须明确选择通道。');self.unit_label.setObjectName('source_unit');self.unit_label.setWordWrap(True);group.addWidget(self.unit_label)
        self.channel_combo.currentIndexChanged.connect(self._channel_changed)
        group=self._group('3 · 选择校准来源')
        self.profile_combo=QComboBox();self.profile_combo.setObjectName('calibration_selector');self.profile_combo.addItem('不使用校准：Pa直通／相对量',None);group.addWidget(self.profile_combo)
        self.calibration_label=QLabel('未校准数字录音不输出 SPL 或 sone。');self.calibration_label.setWordWrap(True);group.addWidget(self.calibration_label)
        self._row(group,self._button('建立配置','create_calibration',self.create_profile),self._button('打开配置','load_calibration',self.choose_profile))
        group.addWidget(self._button('保存所选配置','save_calibration',self.save_profile_dialog))
        self.profile_combo.currentIndexChanged.connect(self._profile_changed)
        group=self._group('4 · 标记原录音事件')
        form=QFormLayout();group.addLayout(form)
        self.start_spin=QDoubleSpinBox();self.end_spin=QDoubleSpinBox()
        for widget,name in [(self.start_spin,'event_start'),(self.end_spin,'event_end')]:
            widget.setObjectName(name);widget.setDecimals(6);widget.setRange(0,1e9);widget.setSuffix(' s');widget.valueChanged.connect(self._selection_changed)
        form.addRow('起点（源时间）',self.start_spin);form.addRow('终点（源时间，不包含）',self.end_spin)
        self.event_combo=QComboBox();self.event_combo.addItems(EVENTS);self.event_combo.setObjectName('event_kind');form.addRow('事件类型',self.event_combo)
        self.event_combo.currentIndexChanged.connect(self.invalidate_result)
        self._row(group,self._button('全事件','select_full',self.select_full),self._button('保存事件','add_event',lambda:self.add_current_event()))
        self.event_list=QListWidget();self.event_list.setObjectName('event_list');self.event_list.setMaximumHeight(115);group.addWidget(self.event_list)
        self.event_list.currentRowChanged.connect(self._select_event)
        self.acquisition_markers=QListWidget();self.acquisition_markers.setObjectName('acquisition_markers')
        self.acquisition_markers.setMaximumHeight(85);self.acquisition_markers.hide();group.addWidget(self.acquisition_markers)
        self.marker_button=self._button('将采集标记设为选段起点（终点需确认）','use_acquisition_marker',self.use_acquisition_marker)
        self.marker_button.hide();group.addWidget(self.marker_button)
        self.context_label=QLabel('工况：未知项可保留，比较与报告会显示。');self.context_label.setWordWrap(True);group.addWidget(self.context_label)
        group.addWidget(self._button('填写/复用工况','measurement_context',self.edit_context))
        group=self._group('5 · 分析与试听')
        form=QFormLayout();group.addLayout(form)
        self.field_combo=QComboBox();self.field_combo.addItem('自由场','free');self.field_combo.addItem('扩散场','diffuse');form.addRow('响度声场',self.field_combo)
        self.fft_combo=QComboBox()
        for size in (1024,4096,8192,16384):self.fft_combo.addItem(str(size),size)
        self.fft_combo.setCurrentIndex(1);form.addRow('FFT 最低点数 / Welch 段长',self.fft_combo)
        self.stft_combo=QComboBox();self.stft_combo.addItem('瞬态：1024 / 512',(1024,512));self.stft_combo.addItem('窄带：8192 / 4096',(8192,4096));form.addRow('语谱图预设',self.stft_combo)
        self.resolution_label=QLabel('');self.resolution_label.setWordWrap(True);group.addWidget(self.resolution_label)
        self.loudness_check=QCheckBox('计算时变响度 Nmean / Nmax');self.loudness_check.setChecked(True);group.addWidget(self.loudness_check)
        self.stationary_check=QCheckBox('额外计算稳态响度');group.addWidget(self.stationary_check)
        self.dc_check=QCheckBox('对全录音去直流（记录到结果）');group.addWidget(self.dc_check)
        for widget in (self.field_combo,self.fft_combo,self.stft_combo):widget.currentIndexChanged.connect(self.invalidate_result)
        for widget in (self.loudness_check,self.stationary_check,self.dc_check):widget.toggled.connect(self.invalidate_result)
        self.stft_combo.currentIndexChanged.connect(self._show_resolution)
        self.fft_combo.currentIndexChanged.connect(self._show_resolution)
        self.analyze_button=self._button('开始分析','analyze',self.start_analysis);self.analyze_button.setEnabled(False)
        self.cancel_button=self._button('取消计算','cancel_job',self.cancel_job);self.cancel_button.setEnabled(False)
        self._row(group,self.analyze_button,self.cancel_button)
        self.progress=QProgressBar();self.progress.setObjectName('job_progress');self.progress.setRange(0,100);group.addWidget(self.progress)
        self._row(group,self._button('播放选段','play_segment',self.play_visible),self._button('暂停','pause_audio',self.player.pause),self._button('停止','stop_audio',self.player.stop))
        self.playback_gain=QDoubleSpinBox();self.playback_gain.setObjectName('playback_gain');self.playback_gain.setDecimals(3);self.playback_gain.setRange(.001,1000);self.playback_gain.setValue(1)
        self.playback_gain.valueChanged.connect(lambda *_:self.player.stop());form.addRow('试听共同增益（不归一化）',self.playback_gain)
        self.playback_state=QLabel('未校准播放设备，不代表真实重放声压。');self.playback_state.setWordWrap(True);group.addWidget(self.playback_state)
        playback_hint=QLabel('试听用于定位声音事件；A/B 保持同一显式增益。播放器与扬声器未作声级校准。');playback_hint.setWordWrap(True);group.addWidget(playback_hint)
        group=self._group('6 · 保存与报告')
        self._row(group,self._button('保存项目','save_project',self.save_project_dialog),self._button('重新定位源文件','relink_source',self.relink_dialog))
        self._row(group,self._button('单文件 DOCX','export_report',self.export_visible_report),self._button('当前事件加入批处理','add_batch_item',self.add_to_batch))
        self.report_combo=QComboBox();self.report_combo.setObjectName('saved_reports');group.addWidget(self.report_combo)
        group.addWidget(self._button('打开已存报告','open_saved_report',self.open_saved_report))
        self.controls.addStretch()
        for label in side.findChildren(QLabel):
            if label.wordWrap():label.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred)
        for combo in side.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon);combo.setMinimumContentsLength(10)
        for form in side.findChildren(QFormLayout):form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.workspace=QTabWidget();splitter.addWidget(self.workspace)
        analysis=QWidget();layout=QVBoxLayout(analysis)
        self.result_label=QLabel('尚无分析结果');self.result_label.setObjectName('result_provenance');self.result_label.setWordWrap(True);layout.addWidget(self.result_label)
        self.metric_table=QTableWidget(0,3);self.metric_table.setObjectName('metric_results');self.metric_table.setHorizontalHeaderLabels(['指标','结果 / 状态','方法 / 说明']);self.metric_table.setMaximumHeight(255)
        self.metric_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers);self.metric_table.horizontalHeader().setStretchLastSection(True);layout.addWidget(self.metric_table)
        self.plots=PlotPanel();layout.addWidget(self.plots,1);self.plots.selectionChanged.connect(self._plot_selection)
        self.workspace.addTab(analysis,'分析工作台')
        compare=QWidget();cl=QVBoxLayout(compare)
        cl.addWidget(QLabel('青色=A 左侧，橙色=B 右侧；差值固定为 B−A。不同工况可以查看，不能自动归因。'))
        self.left_combo=QComboBox();self.right_combo=QComboBox();self.left_combo.setObjectName('comparison_left');self.right_combo.setObjectName('comparison_right')
        self._row(cl,QLabel('A'),self.left_combo,QLabel('B'),self.right_combo,self._button('比较并叠图','compare_results',self.compare_selected))
        self._row(cl,self._button('试听 A','play_a',lambda:self.play_result(self.comparison.left if self.comparison else None)),self._button('试听 B','play_b',lambda:self.play_result(self.comparison.right if self.comparison else None)),self._button('比较 DOCX','comparison_report',self.export_comparison_report))
        self.comparison_label=QLabel('选择两个已完成的结果。');self.comparison_label.setWordWrap(True);cl.addWidget(self.comparison_label)
        self.diff_table=QTableWidget(0,3);self.diff_table.setHorizontalHeaderLabels(['指标','B−A','状态／说明']);self.diff_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers);cl.addWidget(self.diff_table)
        self.repeat_filter=QComboBox();self.repeat_filter.setObjectName('repeat_condition_filter')
        self.repeat_filter.addItem('全部结果','all');self.repeat_filter.addItem('与当前结果条件匹配','matching')
        self._row(cl,QLabel('结果列表'),self.repeat_filter)
        self.repeat_filter_status=QLabel('全部结果；可选择一个结果作为条件筛选的参考。')
        self.repeat_filter_status.setObjectName('repeat_filter_status');self.repeat_filter_status.setWordWrap(True);cl.addWidget(self.repeat_filter_status)
        self.history_list=QListWidget();self.history_list.setObjectName('result_history');self.history_list.currentRowChanged.connect(self.show_history_result);cl.addWidget(self.history_list)
        self.repeat_filter.currentIndexChanged.connect(self._refresh_repeat_history)
        self.workspace.addTab(compare,'比较与重复结果')
        self.batch_panel=BatchPanel();self.workspace.addTab(self.batch_panel,'逐条批处理')
        self.batch_panel.runRequested.connect(self.start_batch);self.batch_panel.addFilesRequested.connect(self.choose_batch_files)
        self.batch_panel.csvRequested.connect(self.export_batch_csv);self.batch_panel.reportRequested.connect(self.export_batch_report);self.batch_panel.error.connect(self.show_message)
        install_workbench(self)

    def show_message(self,message):
        self.status_label.setText(str(message));self.status_label.setToolTip(str(message));self.statusBar().showMessage(str(message))

    def open_soundcard(self):
        if self.soundcard_panel is None:
            from .soundcard_panel import SoundCardPanel
            self.soundcard_panel=SoundCardPanel(self)
            self.soundcard_page.layout().addWidget(self.soundcard_panel)
            self.soundcard_panel.recordingReady.connect(self._import_live_recording)
        self.workspace.setCurrentWidget(self.soundcard_page)

    def _import_live_recording(self,path):
        if self._close_requested:return
        self.player.stop();self._live_import_pending=True
        self._live_import_path=Path(path).resolve()
        self.workspace.setCurrentIndex(0)
        self.open_path(Path(path))

    def open_rotation(self):
        if self.rotation_panel is None:
            from .rotation_panel import RotationPanel
            self.rotation_panel=RotationPanel(self)
            self.rotation_page.layout().addWidget(self.rotation_panel)
            self.rotation_panel.overlayReady.connect(self._rotation_overlay)
        self._sync_rotation_source()
        self.workspace.setCurrentWidget(self.rotation_page)

    def _sync_rotation_source(self):
        if self.rotation_panel is None:return
        channel=self.channel_combo.currentData()
        if self.signal is None or channel is None:
            self.rotation_panel.set_signal(None)
        elif self.start_spin.value()<self.end_spin.value():
            self.rotation_panel.set_signal(self.signal,channel=channel,
                start_seconds=self.start_spin.value(),end_seconds=self.end_spin.value())
        else:
            self.rotation_panel.set_signal(None)

    def _rotation_overlay(self,payload):
        if payload.get('stale'):
            self.plots.set_rotation_reference({'stale':True});return
        if self.signal is None or payload.get('source_hash')!=self.signal.source_hash:return
        if payload.get('channel',self.channel_combo.currentData())!=self.channel_combo.currentData():return
        if self.current_result is not None and not self._source_matches_result(self.current_result):return
        if not self.current_result_valid and payload.get('result') is not None:
            result=payload['result'];plot=self.plots.plots['spectrum'];plot.clear()
            plot.plot(result.frequency,result.spectrum_db,pen='#4cc7d6')
            plot.setLabel('bottom','频率',units='Hz')
            unit=self.signal.channels[self.channel_combo.currentData()].unit
            plot.setLabel('left','幅度',units='dBFS' if unit=='FS' else f'dB re 1 {unit}')
        self.plots.set_rotation_reference(payload)

    def _extension_closed(self):
        if self._close_requested:QTimer.singleShot(0,self.close)

    def open_acquisition(self):
        existing=self._extension_dialogs.get('acquisition')
        if existing is not None and existing.isVisible():
            existing.raise_();existing.activateWindow();return
        from .acquisition_dialog import AcquisitionDialog
        dialog=AcquisitionDialog(parent=self)
        dialog.recordingReady.connect(lambda path:self.open_path(Path(path)) if not self._close_requested else None)
        dialog.finished.connect(self._extension_closed)
        self._extension_dialogs['acquisition']=dialog;dialog.show()

    def open_knowledge(self):
        existing=self._extension_dialogs.get('knowledge')
        if existing is not None and existing.isVisible():
            existing.raise_();existing.activateWindow();return
        from .knowledge_dialog import KnowledgeDialog
        folder=Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation))
        try:
            folder.mkdir(parents=True,exist_ok=True)
            dialog=KnowledgeDialog(folder/'knowledge.sqlite',
                analysis_provider=lambda:self.current_result if self.current_result_valid else None,parent=self)
        except Exception as error:
            self.show_message(f'本地知识库无法打开：{error}');return
        dialog.finished.connect(self._extension_closed)
        self._extension_dialogs['knowledge']=dialog;dialog.show()
    def _restore_preferences(self):
        self._updating=True
        try:
            for key,combo,default in [('fft_size',self.fft_combo,4096),('field_type',self.field_combo,'free')]:
                value=self.preferences.value('analysis/'+key,default,type=type(default))
                index=combo.findData(value)
                if index>=0:combo.setCurrentIndex(index)
            index=self.preferences.value('analysis/stft_preset',0,type=int)
            if 0<=index<self.stft_combo.count():self.stft_combo.setCurrentIndex(index)
            for key,widget,default in [('loudness',self.loudness_check,True),('stationary',self.stationary_check,False),('remove_dc',self.dc_check,False)]:
                widget.setChecked(self.preferences.value('analysis/'+key,default,type=bool))
        except (ValueError,TypeError):pass
        finally:self._updating=False
    def selected_profile(self):return self.profile_combo.currentData()
    def analysis_settings(self):
        if self.signal is None:raise ValueError('请先导入录音。')
        channel=self.channel_combo.currentData()
        stft,hop=self.stft_combo.currentData()
        start=round((self.start_spin.value()-self.signal.time_origin)*self.signal.sample_rate)
        end=round((self.end_spin.value()-self.signal.time_origin)*self.signal.sample_rate)
        settings=AnalysisSettings(channel=channel,start=start,end=end,event_label=EVENTS[self.event_combo.currentText()],
            remove_dc=self.dc_check.isChecked(),fft_size=self.fft_combo.currentData(),stft_size=stft,stft_hop=hop,
            field_type=self.field_combo.currentData(),compute_loudness=self.loudness_check.isChecked(),stationary_loudness=self.stationary_check.isChecked())
        settings.resolve(self.signal);return settings
    def current_item(self):
        if self.signal is None or self.signal.path is None:raise ValueError('请先导入一个有来源路径的录音。')
        return BatchItemSpec(self.signal.path,self.analysis_settings(),self.selected_profile(),self.context,self.mapping,self.signal.source_hash)
    def _show_resolution(self):
        if self.signal is not None:
            size,hop=self.stft_combo.currentData()
            frames=max(1,round((self.end_spin.value()-self.start_spin.value())*self.signal.sample_rate))
            fft=max(frames,self.fft_combo.currentData())
            self.resolution_label.setText(f'事件物理时长 {1000*frames/self.signal.sample_rate:.2f} ms；全事件 FFT 频点间隔 {self.signal.sample_rate/fft:.3f} Hz。\nFFT 最低点数可补零；补零不会提高实际分辨能力。\nSTFT 窗长 {1000*size/self.signal.sample_rate:.2f} ms；频点间隔 {self.signal.sample_rate/size:.3f} Hz。')
    def _channel_changed(self,*_):
        self.player.stop()
        self.invalidate_result()
        channel=self.channel_combo.currentData()
        self.analyze_button.setEnabled(self.signal is not None and channel is not None)
        if self.signal is not None and channel is not None:
            info=self.signal.channels[channel]
            quantity={'digital_amplitude':'数字幅度','digital':'数字幅度','sound_pressure':'声压','voltage':'电压','acceleration':'加速度','unknown':'未知'}.get(info.physical_quantity,info.physical_quantity)
            self.unit_label.setText(f'通道 {channel} · {info.name} · 源单位 {info.unit}\n量纲：{quantity}；源已缩放：{"是" if info.scaled else "否"}\nADC满量程：{info.full_scale if info.full_scale is not None else "未知"}')
            acquisition=info.metadata.get('acquisition_channel',{})
            calibration_record=acquisition.get('calibration_snapshot',{}).get('record')
            if calibration_record:
                self.unit_label.setText(self.unit_label.text()+f'\n采集时校准记录：{calibration_record}（换算配置需明确选择）')
            self.plots.set_signal(self.signal,channel)
            refresh_metric_cards(self,stale=self.current_result is not None)
            self._profile_changed()
        else:self.unit_label.setText('多通道必须明确选择，程序不自动混音。')
        self._sync_rotation_source()
    def _profile_changed(self,*_):
        self.player.stop()
        profile=self.selected_profile()
        if profile:self.calibration_label.setText(f'{profile.source}\n{profile.coefficient:.9g} Pa/{profile.input_unit}；日期：{profile.date or "未知"}')
        else:self.calibration_label.setText('Pa 输入保留文件声明；FS/V 未校准时仅显示相对量。')
        self.invalidate_result()
    def invalidate_result(self,*_):
        if self._updating:return
        if self.current_result is not None:
            self.current_result_valid=False
            self.result_label.setText('设置已变化：当前结果已过期，请重新计算。历史结果快照仍保留。')
            refresh_metric_cards(self,comparison=self.comparison if self._view_comparison else None,
                stale=not self._view_comparison)
        if self._expected_job and self._expected_job[1] in {'analysis','calibration'}:self.cancel_job()
    def _selection_changed(self,*_):
        if self._updating or self.signal is None:return
        # Editing start before end may temporarily reverse the interval. The plot
        # region sorts bounds, so do not feed that temporary state back to widgets.
        if self.start_spin.value()<self.end_spin.value():
            self.plots.set_selection(self.start_spin.value(),self.end_spin.value())
        self.invalidate_result();self.player.stop();self._show_resolution();self._sync_rotation_source()
    def _plot_selection(self,start,end):
        if self._updating or self.signal is None:return
        self._updating=True
        self.start_spin.setValue(max(self.signal.time_origin,start));self.end_spin.setValue(min(self.signal.time_origin+self.signal.duration,end))
        self._updating=False;self.invalidate_result();self.player.stop();self._show_resolution();self._sync_rotation_source()
    def select_full(self):
        if self.signal:
            self.start_spin.setValue(self.signal.time_origin);self.end_spin.setValue(self.signal.time_origin+self.signal.duration);self.event_combo.setCurrentText('全事件')

    def choose_file(self):
        path,_=QFileDialog.getOpenFileName(self,'打开录音',str(self.preferences.value('last_directory','')),
            '录音、采集会话与数值文件 (*.wav *.flac *.mp3 *.m4a *.aac *.ogg *.mp4 *.avi *.mkv *.csv *.txt *.dat *.mat *.h5 *.hdf5 *.hdf *.tdms *.json);;所有文件 (*)')
        if path:self.open_path(Path(path))
    def open_path(self,path,mapping=None,*,restore_result=None,restore_item=None):
        path=Path(path)
        if self._live_import_pending and path.resolve()!=self._live_import_path:self._live_import_pending=False
        self.player.stop();self.invalidate_result()
        self.preferences.setValue('last_directory',str(path.parent))
        self._pending_import=(path,mapping);self._import_options=None
        self._pending_project_result=restore_result;self._pending_event=restore_item
        self._start_job('import',{'path':path,'mapping':mapping})
        self.show_message(f'正在导入：{path.name}。原数据只读，保留采样率与通道。')
    def set_signal(self,signal):
        self.player.stop()
        self._updating=True
        if self._pending_import:self.mapping=self._pending_import[1]
        self.signal=signal;self._pending_import=None;self._signal_channel_hashes={}
        self.file_label.setText(f'{signal.path}\n{signal.channel_count} 通道 · {signal.sample_rate:g} Hz · {signal.duration:.6g} s\n质量：{quality_text(signal.quality) or "未发现数字质量告警；前端过载仍须测量记录"}')
        self.channel_combo.clear()
        if signal.channel_count>1:self.channel_combo.addItem('请选择一个通道',None)
        for index,info in enumerate(signal.channels):self.channel_combo.addItem(f'{index}: {info.name} [{info.unit}]',index)
        self.start_spin.setRange(signal.time_origin,signal.time_origin+signal.duration);self.end_spin.setRange(signal.time_origin,signal.time_origin+signal.duration)
        self.start_spin.setValue(signal.time_origin);self.end_spin.setValue(signal.time_origin+signal.duration)
        self.event_combo.setCurrentText('全事件')
        self._restore_acquisition_context(signal)
        self.current_result=None;self.current_result_valid=False;self.metric_table.setRowCount(0);self.result_label.setText('待分析：确认通道、校准来源和事件。')
        refresh_metric_cards(self)
        self._refresh_repeat_history()
        self._updating=False;self._channel_changed();self._show_resolution()
        self.plots.show_dashboard();self.plots.select_dashboard(time_key='waveform')
        self.show_message('导入完成。确认源单位；已有录音可保留未知工况。')
        if self._pending_project_result:
            result=self._pending_project_result;self._pending_project_result=None
            if result.input_hash!=signal.source_hash:result=replace(result,provenance={**result.provenance,'source_state':'changed'})
            self._restore_controls(result);self.display_result(result)
        if self._pending_event:
            item=self._pending_event;self._pending_event=None;self._restore_item_controls(item)
        if self._live_import_pending:
            self._live_import_pending=False
            if self.channel_combo.currentData() is None:self.channel_combo.setCurrentIndex(1)
            QTimer.singleShot(0,self.start_analysis)
    def mapping_target(self):
        return self._pending_import[0] if self._pending_import else self.signal.path if self.signal else None
    def _restore_acquisition_context(self,signal):
        self.acquisition_markers.clear();self.acquisition_markers.hide();self.marker_button.hide()
        session=signal.metadata.get('acquisition_session')
        if not session:return
        raw=session.get('measurement_context',{})
        values={field.name:raw[field.name] for field in fields(MeasurementContext)
            if field.name in raw and isinstance(raw[field.name],str)}
        for source,target in [('motion_direction','actual_movement'),('cw_reference','rotation_view'),('rotation_reference','rotation_view')]:
            if isinstance(raw.get(source),str) and raw[source].strip():values[target]=raw[source]
        notes=[values.get('notes','')]
        for name,label in [('condition','采集工况'),('operator','操作员')]:
            if isinstance(raw.get(name),str) and raw[name].strip():notes.append(f'{label}：{raw[name]}')
        values['notes']='；'.join(value for value in notes if value)
        self.context=MeasurementContext(**values)
        self._show_context()
        for event in session.get('events',()):
            index=event.get('sample_index')
            if not isinstance(index,int) or not 0<=index<=signal.frames:continue
            kind=event.get('kind','unknown');label=next((key for key,value in EVENTS.items() if value==kind),kind)
            self.acquisition_markers.addItem(f'{label} · 样本 {index} · {index/signal.sample_rate:.6g} s · {event.get("direction") or "旋向未知"}')
            item=self.acquisition_markers.item(self.acquisition_markers.count()-1)
            item.setData(Qt.ItemDataRole.UserRole,dict(event));item.setToolTip(event.get('note',''))
        if self.acquisition_markers.count():self.acquisition_markers.show();self.marker_button.show()
    def use_acquisition_marker(self):
        item=self.acquisition_markers.currentItem()
        if self.signal is None or item is None:return
        event=item.data(Qt.ItemDataRole.UserRole);index=event['sample_index']
        if index>=self.signal.frames:
            self.show_message('该标记位于录音末尾，不能作为非空选段起点。');return
        origin=self.signal.time_origin+index/self.signal.sample_rate
        if self.end_spin.value()<=origin:self.end_spin.setValue(self.signal.time_origin+self.signal.duration)
        self.start_spin.setValue(origin)
        kind=event.get('kind')
        label=next((key for key,value in EVENTS.items() if value==kind),kind if kind in EVENTS else {'稳定':'运行'}.get(kind))
        if label:self.event_combo.setCurrentText(label)
        if event.get('direction') in {'CW','CCW'}:
            self.context=replace(self.context,motor_rotation=event['direction']);self._show_context()
        self.invalidate_result()
        self.show_message('采集标记已设为选段起点；请确认终点后保存事件。点标记不自动划分稳定运行区间。')
    def edit_mapping(self):
        if not self.signal and not self._pending_import:
            self.show_message('请先打开待映射文件。');return
        path=self.mapping_target()
        dialog=ImportDialog(self,options=self._import_options,mapping=self._pending_import[1] if self._pending_import else self.mapping)
        if dialog.exec():
            try:self.open_path(path,dialog.mapping())
            except Exception as error:self.show_message(error)
    def edit_context(self):
        dialog=ContextDialog(self.context,self)
        if dialog.exec():
            self.context=dialog.context();self._show_context()
            self.invalidate_result()
    def _show_context(self):
        context=self.context
        text=(f'试件：{context.specimen}；层级：{context.test_level}\n'
            f'方向：{context.actual_movement}；电机旋向：{context.motor_rotation}；旋向基准：{context.rotation_view}\n'
            f'供电：{context.supply}；负载：{context.load}；测点：{context.mic_position}')
        if context.notes:text+='\n'+context.notes
        self.context_label.setText(text)
    def add_profile(self,profile,select=True):
        if not any(value.profile_id==profile.profile_id for value in self.profiles):
            self.profiles.append(profile);self.profile_combo.addItem(f'{profile.name} · {profile.source}',profile)
            self.batch_panel.profiles=list(self.profiles)
        if select:
            index=next(i+1 for i,value in enumerate(self.profiles) if value.profile_id==profile.profile_id)
            self.profile_combo.setCurrentIndex(index)
    def _load_reference_profile(self):
        from ..resources import resource_root
        path=resource_root()/'profiles'/'head-pa-reference.json'
        if path.is_file():
            try:
                from ..calibration import load_profile
                self.add_profile(load_profile(path),select=False)
            except Exception as error:self.show_message(f'专用配置未载入：{error}')
    def create_profile(self):
        try:settings=self.analysis_settings()
        except Exception as error:self.show_message(error);return
        dialog=CalibrationDialog(self.signal,settings.channel,settings.start,settings.end,self)
        if dialog.exec():
            try:
                if dialog.tabs.currentIndex()==2:
                    self._start_job('calibration',{'path':self.signal.path,'mapping':self.mapping,'input_hash':self.signal.source_hash,'parameters':dialog.worker_parameters()})
                    self.show_message('正在独立进程检查校准音，完成后保留来源与选段记录。')
                else:self.add_profile(dialog.profile())
            except Exception as error:self.show_message(error)
    def choose_profile(self):
        path,_=QFileDialog.getOpenFileName(self,'打开校准配置','','JSON (*.json)')
        if path:
            try:
                from ..calibration import load_profile
                self.add_profile(load_profile(Path(path)))
            except Exception as error:self.show_message(error)
    def save_profile_dialog(self):
        profile=self.selected_profile()
        if profile is None:self.show_message('请先选择一个有来源的配置。');return
        path,_=QFileDialog.getSaveFileName(self,'保存校准配置',profile.name+'.json','JSON (*.json)')
        if path:
            try:
                from ..calibration import save_profile
                save_profile(profile,Path(path));self.show_message('校准配置已保存。')
            except Exception as error:self.show_message(error)

    def add_current_event(self,label=None):
        try:
            if label:self.event_combo.setCurrentText(label)
            item=self.current_item();self.project_items.append(item)
            self.event_list.addItem(f'{self.event_combo.currentText()} · ch{item.settings.channel} · {item.settings.start}:{item.settings.end}')
            self.event_list.item(self.event_list.count()-1).setData(Qt.ItemDataRole.UserRole,item.item_id)
            self.show_message('事件已保存为独立分析条目；未自动判断中间段为稳定段。')
        except Exception as error:self.show_message(error)
    def _select_event(self,row):
        if row<0 or self.signal is None:return
        item_id=self.event_list.item(row).data(Qt.ItemDataRole.UserRole)
        item=next((item for item in self.project_items if item.item_id==item_id),None)
        if item and item.path==self.signal.path:
            self._restore_item_controls(item);self.invalidate_result()
        elif item:self.open_path(item.path,item.mapping,restore_item=item)
    def _restore_item_controls(self,item):
        self._updating=True
        self.context=item.context;self._show_context()
        if item.profile:self.add_profile(item.profile)
        else:self.profile_combo.setCurrentIndex(0)
        self._set_settings(item.settings)
        self._updating=False
        if self.signal:self.plots.set_selection(self.start_spin.value(),self.end_spin.value())
    def _restore_controls(self,result):
        self._restore_item_controls(BatchItemSpec(result.path,result.settings,result.calibration,result.context,input_hash=result.input_hash))
    def _set_settings(self,settings):
        if self.signal:
            self.channel_combo.setCurrentIndex(self.channel_combo.findData(settings.channel))
            self.start_spin.setValue(self.signal.time_origin+settings.start/self.signal.sample_rate)
            self.end_spin.setValue(self.signal.time_origin+(settings.end or self.signal.frames)/self.signal.sample_rate)
        self.event_combo.setCurrentText(next((key for key,value in EVENTS.items() if value==settings.event_label),'异常'))
        self.field_combo.setCurrentIndex(self.field_combo.findData(settings.field_type))
        if self.fft_combo.findData(settings.fft_size)<0:self.fft_combo.addItem(str(settings.fft_size),settings.fft_size)
        self.fft_combo.setCurrentIndex(self.fft_combo.findData(settings.fft_size))
        target=(settings.stft_size,settings.stft_hop);index=self.stft_combo.findData(target)
        if index<0:self.stft_combo.addItem(f'{target[0]} / {target[1]}',target);index=self.stft_combo.count()-1
        self.stft_combo.setCurrentIndex(index)
        self.loudness_check.setChecked(settings.compute_loudness);self.stationary_check.setChecked(settings.stationary_loudness);self.dc_check.setChecked(settings.remove_dc)
    def _start_job(self,kind,payload):
        job=self.jobs.submit(kind,payload);self._expected_job=(job,kind)
        self.cancel_button.setEnabled(True);self.progress.setRange(0,0)
        return job
    def start_analysis(self):
        try:
            item=self.current_item();self._job_snapshot=item
            self._start_job('analysis',dict(path=item.path,mapping=item.mapping,profile=item.profile,settings=item.settings,context=item.context,input_hash=item.input_hash))
            self.show_message('正在计算；关闭或切换设置会取消旧任务，过期输出不会覆盖当前结果。')
        except Exception as error:self.show_message(error)
    def cancel_job(self):
        self.jobs.cancel();self.show_message('已请求取消，正在隔离计算并保留已完成批量条目。')
        self.progress.setRange(0,100);self.progress.setValue(0)
    def poll_jobs(self):
        for event in self.jobs.poll():
            if not self._expected_job or event.job_id!=self._expected_job[0]:continue
            if event.state=='progress':
                self.progress.setRange(0,event.total);self.progress.setValue(event.completed);self.show_message(f'{event.completed}/{event.total} · {event.message}')
            elif event.state in {'complete','failed','cancelled','stale'}:
                self._expected_job=None;self.cancel_button.setEnabled(False);self.progress.setRange(0,100);self.progress.setValue(100 if event.state=='complete' else 0)
                if event.state=='complete':
                    if event.kind=='import':self.set_signal(event.result)
                    elif event.kind=='calibration':self.add_profile(event.result);self.show_message('独立声校准录音检查完成；可保存配置并打开待测录音。')
                    elif event.kind=='analysis':
                        if self._job_snapshot:self.project_items.append(self._job_snapshot)
                        self.accept_result(event.result)
                    elif event.kind=='batch':self.batch_panel.set_result(event.result);self._accept_batch(event.result)
                    elif event.kind in {'report','export_csv'}:
                        self.reports.append(str(event.result));self._refresh_reports();self.show_message(f'已保存：{event.result}')
                elif event.state=='cancelled' and event.kind=='batch' and event.result:
                    self.batch_panel.set_result(event.result);self._accept_batch(event.result)
                if event.state=='failed' and event.kind=='import' and isinstance(event.result,dict):
                    self._import_options=event.result.get('mapping_options')
                if event.state!='complete':self.show_message(event.message+('；点击“修改导入映射”查看可选字段。' if self._import_options else ''))
    def accept_result(self,result):
        index=next((i for i,previous in enumerate(self.results) if previous.result_id==result.result_id),None)
        if index is None:self.results.append(result)
        elif result.detail is not None:self.results[index]=result
        if not any(item.input_hash==result.input_hash and item.settings==result.settings for item in self.project_items):
            self.project_items.append(BatchItemSpec(result.path,result.settings,result.calibration,result.context,self.mapping,result.input_hash))
        self.display_result(result);self._refresh_history();self.show_message('分析完成。检查指标状态、来源、声场和事件，再保存或导出。')
    def display_result(self,result):
        self.current_result=result;self._history_selected=False;self.current_result_valid=result.provenance.get('source_state','ok') not in {'missing','changed','unverified'}
        if self._source_matches_result(result):
            previous=self._updating;self._updating=True
            self.plots.set_signal(self.signal,result.settings.channel)
            self.plots.set_selection(self.signal.time_origin+result.settings.start/self.signal.sample_rate,self.signal.time_origin+(result.settings.end or self.signal.frames)/self.signal.sample_rate)
            self._updating=previous
        else:self.plots.show_result_waveform(result)
        source=result.calibration.source if result.calibration else result.provenance.get('calibration_source','Pa文件声明／未校准相对量')
        self.result_label.setText(f'结果来源：{result.path}\n事件 [{result.settings.start}, {result.settings.end}) · 通道 {result.settings.channel} · 校准：{source} · 声场：'+{'free':'自由场','diffuse':'扩散场'}.get(result.settings.field_type,result.settings.field_type)+'\n'+
            ('历史源文件未验证／已变化：保留原快照，不自动重算。\n' if not self.current_result_valid else '')+quality_text(result.summary.warnings))
        if not self._source_matches_result(result):
            selection=result.provenance.get('source_metadata',{}).get('import_mapping')
            self.result_label.setText(self.result_label.text()+'\n图表来自保存的结果，当前导入选择不同。保存的导入映射：'+str(selection or '文件原有声明'))
        metrics=list(result.summary.metrics.items());self.metric_table.setRowCount(len(metrics))
        for row,(key,metric) in enumerate(metrics):
            for column,text in enumerate([f'{key} · {METRIC_LABELS.get(key,key)}',metric_text(metric),f'{metric.method} {metric.message}']):self.metric_table.setItem(row,column,QTableWidgetItem(text))
        self.metric_table.resizeColumnsToContents();self.plots.set_result(result)
        refresh_metric_cards(self,stale=not self.current_result_valid)
        self._refresh_repeat_history()
    def _refresh_history(self):
        self.left_combo.clear();self.right_combo.clear()
        for index,result in enumerate(self.results):
            label=f'{index+1} · {result.path.name if result.path else "录音"} · ch{result.settings.channel} · {result.settings.event_label} [{result.settings.start}:{result.settings.end}]'
            self.left_combo.addItem(label,index);self.right_combo.addItem(label,index)
        if len(self.results)>1:self.right_combo.setCurrentIndex(len(self.results)-1)
        self._refresh_repeat_history()
    def _refresh_repeat_history(self,*_):
        from ..analysis.repeats import compare_repeat_conditions
        matching=self.repeat_filter.currentData()=='matching';reference=self.current_result
        previous=self.history_list.blockSignals(True)
        self.history_list.clear();visible=0;selected=-1
        reference_match=compare_repeat_conditions(reference,reference) if reference is not None else None
        try:
            for index,result in enumerate(self.results):
                match=compare_repeat_conditions(reference,result) if reference is not None else None
                if matching and (match is None or not match.matched):continue
                label=f'{index+1} · {result.path.name if result.path else "录音"} · ch{result.settings.channel} · {result.settings.event_label} [{result.settings.start}:{result.settings.end}]'
                if result.context.repeat not in ('unknown',''):label+=f' · 重复号 {result.context.repeat}'
                item=QListWidgetItem(label);item.setData(Qt.ItemDataRole.UserRole,index)
                item.setToolTip(f'路径：{result.path}\n输入 SHA256：{result.input_hash}\n结果 ID：{result.result_id}\n备注：{result.context.notes or "无"}')
                self.history_list.addItem(item)
                if reference is not None and result.result_id==reference.result_id:selected=visible
                visible+=1
            if selected>=0:self.history_list.setCurrentRow(selected)
        finally:self.history_list.blockSignals(previous)
        scope='工况（除重复号／备注）、通道／单位／采样率、事件类型、算法设置／版本和校准记录。'
        if not matching:
            self.repeat_filter_status.setText(f'全部 {visible} 个分析条目；选择结果后可按保存条件筛选。')
        elif reference is None:
            self.repeat_filter_status.setText('尚无当前结果；切换“全部结果”并选择参考条目。')
        else:
            missing=reference_match.missing
            prefix=f'显示 {visible}/{len(self.results)} 个分析条目；参考 {reference.result_id[:12]}。'
            if missing:prefix+=f'仅记录匹配（{len(missing)} 项缺失）；重复试验未确认。'
            else:prefix+='保存条件匹配；重复试验及独立校准未确认。'
            self.repeat_filter_status.setText(prefix)
        details=[scope,'列表数量不是独立重复试验次数；文件声明 Pa 不代表传感器链或绝对校准已验证。','每条使用各自样本区间，不证明同一瞬态。']
        if reference_match is not None:
            for key,value in reference_match.reference_conditions.items():
                label=CONTEXT_LABELS.get(key.removeprefix('context.'),key)
                if key=='calibration.record':
                    record=reference.provenance.get('calibration')
                    source=reference.calibration.source if reference.calibration else record.get('source','未记录') if isinstance(record,Mapping) else '未记录'
                    value=f'来源：{source}；完整记录保存在结果／项目中'
                details.append(f'{label} ({key})：{value}')
            if reference_match.missing:details.append('缺项：'+', '.join(reference_match.missing))
        self.repeat_filter_status.setToolTip('\n'.join(details))
    def show_history_result(self,row):
        item=self.history_list.item(row) if row>=0 else None
        index=item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        # Preserve direct all-history callers that supply results before the
        # list is refreshed. Filtered rows must always resolve via their IDs.
        if item is None and self.repeat_filter.currentData()=='all':index=row
        if isinstance(index,int) and 0<=index<len(self.results):
            self.player.stop()
            result=self.results[index]
            if self._source_matches_result(result):
                self._restore_controls(result)
            self.display_result(result);self._history_selected=True

    def _source_matches_result(self,result):
        signal=self.signal;source=result.provenance
        if signal is None or signal.source_hash!=result.input_hash:return False
        channel=result.settings.channel
        if channel is None or not 0<=channel<signal.channel_count:return False
        if (source.get('sample_rate')!=signal.sample_rate or source.get('source_frames')!=signal.frames or
            source.get('time_origin_seconds')!=signal.time_origin or source.get('source_metadata')!=signal.metadata or
            source.get('source_unit')!=signal.channels[channel].unit or source.get('channel_name')!=signal.channels[channel].name):return False
        expected=source.get('selected_samples_sha256')
        if not expected:return False
        if channel not in self._signal_channel_hashes:
            digest=hashlib.sha256()
            for left in range(0,signal.frames,131072):
                digest.update(memoryview(np.ascontiguousarray(signal.samples[channel,left:left+131072])))
            self._signal_channel_hashes[channel]=digest.hexdigest()
        return self._signal_channel_hashes[channel]==expected
    def _result_from_combo(self,combo):
        index=combo.currentData();return self.results[index] if index is not None and index<len(self.results) else None
    def compare_selected(self):
        left,right=self._result_from_combo(self.left_combo),self._result_from_combo(self.right_combo)
        if left is None or right is None:self.show_message('请先完成两个结果。');return
        try:
            from ..analysis.comparison import compare_results
            self.comparison=compare_results(left,right)
            conditions='条件未对齐：\n'+'\n'.join(self.comparison.condition_differences) if self.comparison.condition_differences else '已检查条件一致；不输出自动改善或故障结论。'
            self.comparison_label.setText(f'A：{left.path.name} · 样本[{left.settings.start}, {left.settings.end or "末尾"})\nB：{right.path.name} · 样本[{right.settings.start}, {right.settings.end or "末尾"})\n'+conditions)
            self.comparison_label.setToolTip(f'A：{left.path}\nB：{right.path}\n'+conditions)
            self.diff_table.setRowCount(len(self.comparison.differences))
            for row,(key,metric) in enumerate(self.comparison.differences.items()):
                for column,text in enumerate([key,metric_text(metric),metric.message]):self.diff_table.setItem(row,column,QTableWidgetItem(text))
            self.plots.overlay(left,right)
            self.workspace.setCurrentIndex(0);self.plots.show_dashboard()
            self.plots.select_dashboard(frequency_key='spectrum')
            refresh_metric_cards(self,comparison=self.comparison)
            self.analysis_tabs.setCurrentWidget(self.comparison_detail_page)
            self.show_message('已叠图：青色 A、橙色 B，各自频率/时间轴保留；差值为 B−A。')
        except Exception as error:self.show_message(error)

    def play_visible(self):
        if self._view_comparison and self.comparison is not None:self.play_result(self.comparison.right)
        elif self._history_selected and self.current_result is not None and not self._source_matches_result(self.current_result):
            self.play_result(self.current_result)
        else:self.play_current()
    def play_current(self):
        try:
            settings=self.analysis_settings();profile=self.selected_profile()
            key=('source',id(self.signal),settings.channel,settings.start,settings.end,
                 profile,self.playback_gain.value())
            if self._playback_key==key and self.player.state in {'paused','playing'}:
                self.player.play();return
            values=self.signal.samples[settings.channel]
            if profile or self.signal.channels[settings.channel].unit=='Pa':
                from ..calibration import apply_calibration
                values=apply_calibration(self.signal,profile,channel=settings.channel).samples[0]
            self.player.set_segment(values,self.signal.sample_rate,settings.start,settings.end,self.playback_gain.value(),time_origin=self.signal.time_origin)
            self._playback_key=key;self.player.play()
        except Exception as error:self.player.stop();self.show_message(error)
    def play_result(self,result):
        try:
            if result is None or result.detail is None:raise ValueError('此结果没有试听详情；需明确重算或恢复已存缓存。')
            key=('result',id(result.detail),result.result_id,self.playback_gain.value())
            if self._playback_key==key and self.player.state in {'paused','playing'}:
                self.player.play();return
            arrays=result.detail.arrays;time=np.asarray(arrays['wave_time']);values=np.asarray(arrays['waveform'])
            if len(time)<2:raise ValueError('试听区间过短。')
            rate=result.provenance.get('sample_rate')
            if rate is None:rate=1/(time[1]-time[0])
            self.player.set_segment(values,rate,gain=self.playback_gain.value());self.player.origin=float(time[0])
            self._playback_key=key;self.player.play()
        except Exception as error:self.player.stop();self.show_message(error)

    def add_to_batch(self):
        try:self.batch_panel.add_item(self.current_item());self.workspace.setCurrentWidget(self.batch_panel)
        except Exception as error:self.show_message(error)
    def choose_batch_files(self):
        paths,_=QFileDialog.getOpenFileNames(self,'加入批处理文件',str(self.preferences.value('last_directory','')),'所有文件 (*)')
        for path in paths:
            # Common preset is explicit; invalid channels/settings fail per row, never fall back.
            try:
                settings=self.analysis_settings() if self.signal is not None else AnalysisSettings(channel=None)
                self.batch_panel.add_item(BatchItemSpec(Path(path),settings,self.selected_profile(),self.context,self.mapping))
            except Exception as error:self.show_message(error)
    def start_batch(self):
        items=self.batch_panel.items()
        if not items:self.show_message('请先加入分析条目并检查逐行配置。');return
        if any(item.settings.channel is None for item in items):self.show_message('有条目未指定通道，请双击该行配置。');return
        self.project_items.extend(item for item in items if not any(existing.item_id==item.item_id for existing in self.project_items))
        self._start_job('batch',{'items':items});self.show_message(f'开始 {len(items)} 条独立分析；相同路径不会覆盖不同事件。')
    def _accept_batch(self,batch):
        for item in batch.items:
            if item.result is not None and not any(result.result_id==item.result.result_id for result in self.results):self.results.append(item.result)
        self._refresh_history()

    def project(self):
        items=list(self.project_items)
        for item in self.batch_panel.items():
            if not any(old.item_id==item.item_id for old in items):items.append(item)
        if self.signal is not None:
            try:
                current=self.current_item()
                if not any(old.path==current.path and old.settings==current.settings for old in items):items.append(current)
            except Exception:pass
        return MeasurementProject(self.project_path.stem if self.project_path else '声学测量项目',tuple(items),
            tuple(ResultSnapshot(result) for result in self.results),tuple(self.reports),
            project_id=self._project.project_id if self._project else MeasurementProject('temp').project_id)
    def save_project_path(self,path):
        path=Path(path);project=self.project();save_project(project,path);self.project_path=path;self._project=project
        self.show_message(f'项目及不可变结果/详情已保存：{path}');return path
    def save_project_dialog(self):
        path,_=QFileDialog.getSaveFileName(self,'保存项目',str(self.project_path or '声学测量项目.json'),'JSON (*.json)')
        if path:
            try:self.save_project_path(path)
            except Exception as error:self.show_message(error)
    def choose_project(self):
        path,_=QFileDialog.getOpenFileName(self,'打开项目','','JSON (*.json)')
        if path:
            try:self.open_project_path(path)
            except Exception as error:self.show_message(error)
    def open_project_path(self,path):
        self.cancel_job();self._expected_job=None;self.player.stop()
        path=Path(path);project=load_project(path);self._project=project;self.project_path=path
        self.project_items=list(project.items);self.reports=list(project.reports);self.results=[]
        self.current_result=None;self.current_result_valid=False;self._history_selected=False;self.comparison=None
        self._pending_project_result=None;self._pending_event=None;self._pending_import=None;self._job_snapshot=None
        self.metric_table.setRowCount(0);self.diff_table.setRowCount(0)
        refresh_metric_cards(self)
        self.result_label.setText('项目尚无当前分析结果。');self.comparison_label.setText('选择两个已完成的结果。')
        self.plots.clear_rotation_reference();self._sync_rotation_source()
        for plot in self.plots.plots.values():plot.clear()
        for bar in self.plots.colorbars.values():bar.setImageItem([]);bar.hide()
        for item in project.items:
            if item.profile:self.add_profile(item.profile,select=False)
        self._refresh_reports()
        warnings=[]
        for snapshot in project.results:
            result=snapshot.result
            try:result=replace(result,detail=load_result_detail(snapshot,path))
            except Exception as error:warnings.append(str(error))
            self.results.append(result)
        self.batch_panel.specifications=list(project.items);self.batch_panel.refresh()
        self.event_list.clear()
        for item in project.items:
            self.event_list.addItem(f'{item.path.name} · ch{item.settings.channel} · {item.settings.event_label} [{item.settings.start}:{item.settings.end}]')
            self.event_list.item(self.event_list.count()-1).setData(Qt.ItemDataRole.UserRole,item.item_id)
        self._refresh_history()
        if self.results:self.display_result(self.results[-1])
        states=source_states(project)
        self.show_message('项目已重开。'+('；'.join(warnings) if warnings else '')+' 源状态：'+str(list(states.values())))
        if project.items:
            item=next((item for item in reversed(project.items) if self.results and item.input_hash==self.results[-1].input_hash),project.items[-1])
            if states[item.item_id]=='ok':
                self.open_path(item.path,item.mapping,restore_result=self.results[-1] if self.results else None)
    def relink_dialog(self):
        project=self.project();item=None
        if project.items:
            row=self.batch_panel.table.currentRow();item=project.items[row] if 0<=row<len(project.items) else project.items[-1]
        if item is None:self.show_message('项目没有可重新定位的原文件。');return
        path,_=QFileDialog.getOpenFileName(self,'重新定位：'+item.path.name,'','所有文件 (*)')
        if path:
            try:
                project=relink_source(project,item.item_id,Path(path));self._project=project;self.project_items=list(project.items)
                self.results=[replace(snapshot.result,detail=next((old.detail for old in self.results if old.result_id==snapshot.result.result_id),None)) for snapshot in project.results]
                self.batch_panel.specifications=list(project.items);self.batch_panel.refresh();self._refresh_history();self.open_path(path,item.mapping)
            except Exception as error:self.show_message(error)

    def _export(self,result,kind,title,extension):
        if result is None:self.show_message('没有可导出的完成结果。');return
        path,_=QFileDialog.getSaveFileName(self,title,'声学分析.'+extension,extension.upper()+f' (*.{extension})')
        if path:self._start_job(kind,{'result':result,'destination':Path(path),
            'project_name':self.project_path.stem if self.project_path else '声学测量项目','project_version':1})
    def export_visible_report(self):
        if self._view_comparison and self.comparison is not None:self.export_comparison_report()
        else:self.export_current_report()
    def export_current_report(self):
        if not self.current_result_valid and not self._history_selected:self.show_message('当前设置或来源已变化，请重算；历史结果可在结果列表中选择后导出原快照。');return
        self._export(self.current_result,'report','单文件报告','docx')
    def export_comparison_report(self):self._export(self.comparison,'report','比较报告','docx')
    def export_batch_report(self):self._export(self.batch_panel.result,'report','批量报告','docx')
    def export_batch_csv(self):self._export(self.batch_panel.result,'export_csv','批量 CSV','csv')
    def _refresh_reports(self):
        self.report_combo.clear()
        for path in self.reports:self.report_combo.addItem(Path(path).name,path)
    def open_saved_report(self):
        path=self.report_combo.currentData()
        if not path or not Path(path).is_file():self.show_message('报告文件不存在，请检查保存位置。');return
        if Path(path).suffix.lower() not in {'.docx','.csv'}:self.show_message('只打开本程序支持的 DOCX/CSV 报告。');return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).resolve()))):self.show_message('系统未能打开报告，请安装相应查看程序。')
    def closeEvent(self,event):
        self._close_requested=True
        for panel in (self.soundcard_panel,self.rotation_panel):
            if panel is not None and not panel.shutdown():
                self.show_message('正在停止采集并保存原始录音，或等待 NVH 分析线程结束。')
                event.ignore();QTimer.singleShot(250,self.close);return
        for dialog in self._extension_dialogs.values():
            if dialog.isVisible() and not dialog.close():
                self.show_message('正在等待采集停止与最后数据排空保存；完成后自动关闭。')
                event.ignore();QTimer.singleShot(500,self.close);return
        self.timer.stop();self.player.stop();self.jobs.close();self.preferences.setValue('geometry',self.saveGeometry())
        save_workbench(self)
        self.preferences.setValue('analysis/fft_size',self.fft_combo.currentData());self.preferences.setValue('analysis/field_type',self.field_combo.currentData())
        self.preferences.setValue('analysis/stft_preset',self.stft_combo.currentIndex())
        for key,widget in [('loudness',self.loudness_check),('stationary',self.stationary_check),('remove_dc',self.dc_check)]:self.preferences.setValue('analysis/'+key,widget.isChecked())
        super().closeEvent(event)
