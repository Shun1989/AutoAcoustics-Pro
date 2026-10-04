from dataclasses import replace
from pathlib import Path
from PyQt6.QtCore import Qt,pyqtSignal
from PyQt6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QTableWidget,QTableWidgetItem,
    QPushButton,QLabel,QDialog,QFormLayout,QSpinBox,QLineEdit,QComboBox,QCheckBox,QDialogButtonBox,
    QFrame,QHeaderView,QPlainTextEdit)
from ..model import AnalysisSettings,CalibrationProfile,MeasurementContext,BatchItemSpec,MetricResult
from .context_dialog import ContextDialog
from .import_dialog import ImportDialog
from .formatting import metric_text,STATUS_LABELS

class BatchItemDialog(QDialog):
    def __init__(self,item,profiles,parent=None):
        super().__init__(parent)
        profiles=list(profiles)
        if item.profile and not any(profile.profile_id==item.profile.profile_id for profile in profiles):profiles.append(item.profile)
        self.item=item;self.profile_values=[None]+profiles
        self.context_value=item.context;self.mapping_value=item.mapping
        self.setWindowTitle('逐条分析配置（不会自动回退）');self.resize(630,540)
        layout=QVBoxLayout(self);form=QFormLayout();layout.addLayout(form)
        form.addRow('原文件',QLabel(str(item.path)))
        self.channel=QSpinBox();self.channel.setRange(-1,999);self.channel.setSpecialValueText('必须指定');self.channel.setValue(item.settings.channel if item.settings.channel is not None else -1)
        self.start=QLineEdit(str(item.settings.start));self.end=QLineEdit('' if item.settings.end is None else str(item.settings.end))
        self.event=QLineEdit(item.settings.event_label)
        self.fft=QSpinBox();self.fft.setRange(2,1048576);self.fft.setValue(item.settings.fft_size)
        self.stft=QSpinBox();self.stft.setRange(2,1048576);self.stft.setValue(item.settings.stft_size)
        self.hop=QSpinBox();self.hop.setRange(1,1048576);self.hop.setValue(item.settings.stft_hop)
        self.field=QComboBox();self.field.addItems(['free','diffuse']);self.field.setCurrentText(item.settings.field_type)
        self.profile=QComboBox()
        self.profile.addItem('不使用校准（Pa直通／数字相对量）')
        for profile in profiles:self.profile.addItem(profile.name)
        if item.profile:
            for index,profile in enumerate(self.profile_values):
                if profile and profile.profile_id==item.profile.profile_id:self.profile.setCurrentIndex(index)
        self.dc=QCheckBox('去直流');self.dc.setChecked(item.settings.remove_dc)
        self.loudness=QCheckBox('时变响度');self.loudness.setChecked(item.settings.compute_loudness)
        self.stationary=QCheckBox('额外稳态响度');self.stationary.setChecked(item.settings.stationary_loudness)
        for title,widget in [('通道索引（从0开始）',self.channel),('原样本起点',self.start),('原样本终点（空=文件末尾）',self.end),
            ('事件名称',self.event),('FFT 最低点数 / Welch 段长',self.fft),('STFT 窗长',self.stft),('STFT 步长',self.hop),('响度声场',self.field),('校准配置',self.profile)]:form.addRow(title,widget)
        form.addRow(self.dc);form.addRow(self.loudness);form.addRow(self.stationary)
        context=QPushButton('编辑本条工况');context.clicked.connect(self.edit_context);form.addRow(context)
        mapping=QPushButton('编辑本条导入映射');mapping.clicked.connect(self.edit_mapping);form.addRow(mapping)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)

    def edit_context(self):
        dialog=ContextDialog(self.context_value,self)
        if dialog.exec():self.context_value=dialog.context()
    def edit_mapping(self):
        dialog=ImportDialog(self,mapping=self.mapping_value)
        if dialog.exec():self.mapping_value=dialog.mapping()
    def specification(self):
        channel=self.channel.value()
        if channel<0:raise ValueError('必须明确指定本条分析通道。')
        settings=replace(self.item.settings,channel=channel,start=int(self.start.text()),
            end=int(self.end.text()) if self.end.text().strip() else None,event_label=self.event.text(),
            fft_size=self.fft.value(),stft_size=self.stft.value(),stft_hop=self.hop.value(),field_type=self.field.currentText(),
            remove_dc=self.dc.isChecked(),compute_loudness=self.loudness.isChecked(),stationary_loudness=self.stationary.isChecked())
        return replace(self.item,settings=settings,profile=self.profile_values[self.profile.currentIndex()],
            context=self.context_value,mapping=self.mapping_value)

class BatchPanel(QWidget):
    runRequested=pyqtSignal();addFilesRequested=pyqtSignal();csvRequested=pyqtSignal();reportRequested=pyqtSignal()
    error=pyqtSignal(str)
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setObjectName('batch_workspace')
        self.specifications=[];self.profiles=[];self.result=None
        layout=QVBoxLayout(self);layout.setContentsMargins(18,18,18,18);layout.setSpacing(14)
        title=QLabel('批量分析工作区');title.setObjectName('batch_title');title.setProperty('role','sectionTitle')
        layout.addWidget(title)
        hint=QLabel('按文件、通道和事件逐条配置。声级与响度保留各自的单位、状态和缺值原因；双击条目编辑设置。')
        hint.setObjectName('batch_hint');hint.setProperty('role','muted');hint.setWordWrap(True);layout.addWidget(hint)
        cards=QHBoxLayout();cards.setSpacing(10);layout.addLayout(cards)
        self.count_labels={}
        for name,caption in [('files','文件数'),('items','分析条目'),('completed','已完成'),
            ('partial','部分指标失败'),('failed','失败'),('cancelled','已取消')]:
            card=QFrame();card.setObjectName(f'batch_{name}_card');card.setProperty('role','metricCard')
            card_layout=QVBoxLayout(card);card_layout.setContentsMargins(14,10,14,10);card_layout.setSpacing(4)
            value=QLabel('0');value.setObjectName(f'batch_{name}_count');value.setProperty('role','metricValue')
            label=QLabel(caption);label.setProperty('role','muted')
            card_layout.addWidget(value);card_layout.addWidget(label);cards.addWidget(card,1)
            self.count_labels[name]=value
        bar=QHBoxLayout();bar.setSpacing(8);layout.addLayout(bar)
        for title,name,callback in [('加入文件','batch_add_files',self.addFilesRequested.emit),('编辑选中条目','batch_edit',self.edit_current),
            ('删除条目','batch_remove',self.remove_current),('开始批处理','batch_run',self.runRequested.emit),
            ('导出 CSV','batch_csv',self.csvRequested.emit),('批量 DOCX','batch_report',self.reportRequested.emit)]:
            button=QPushButton(title);button.setObjectName(name);button.clicked.connect(callback);bar.addWidget(button)
            if name=='batch_run':button.setProperty('role','primary')
            if name=='batch_remove':bar.addStretch(1)
        self.table=QTableWidget(0,12);self.table.setObjectName('batch_items')
        self.table.setHorizontalHeaderLabels(['文件','通道','起始样本','结束样本','事件','校准来源','设置／工况','状态',
            'LAeq · 平均 A 声级','Nmean · 平均响度','Nmax · 最大响度','结果说明'])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.cellDoubleClicked.connect(lambda *_:self.edit_current())
        self.table.verticalHeader().setVisible(False);self.table.verticalHeader().setDefaultSectionSize(48)
        header=self.table.horizontalHeader();header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for column,width in enumerate((150,55,85,95,100,125,200,145,175,150,150,300)):
            self.table.setColumnWidth(column,width)
        header.setStretchLastSection(True);layout.addWidget(self.table,1)
        self.result_readout=QPlainTextEdit();self.result_readout.setObjectName('batch_result_readout')
        self.result_readout.setReadOnly(True);self.result_readout.setMinimumHeight(82);self.result_readout.setMaximumHeight(108)
        self.result_readout.setPlaceholderText('单击条目，在此查看声级、响度和完整结果说明。')
        layout.addWidget(self.result_readout)
        self.table.currentCellChanged.connect(lambda *_:self._update_readout())
        self.summary=QLabel('0 文件 · 0 分析条目');self.summary.setObjectName('batch_summary')
        self.summary.setProperty('role','muted');self.summary.setWordWrap(True);layout.addWidget(self.summary)

    def add_item(self,item):
        self.specifications.append(item);self.refresh()
    def items(self):return tuple(self.specifications)
    def refresh(self):
        self.result=None
        self.table.clearContents()
        self.table.setRowCount(len(self.specifications))
        for row,item in enumerate(self.specifications):
            settings=item.settings
            values=[item.path.name,str(settings.channel) if settings.channel is not None else '未指定',str(settings.start),
                str(settings.end) if settings.end is not None else '文件末尾',settings.event_label,
                item.profile.source if item.profile else '未校准／Pa直通',
                f'FFT={settings.fft_size}, STFT={settings.stft_size}/{settings.stft_hop}, {settings.field_type}; {item.context.specimen}',
                '待运行','— 待运行','— 待运行','— 待运行','尚未运行本条分析']
            for column,text in enumerate(values):self._set_cell(row,column,text)
            path_cell=self.table.item(row,0);path_cell.setToolTip(str(item.path))
            path_cell.setData(Qt.ItemDataRole.UserRole,str(item.path))
        self._update_summary()
        self._update_readout()

    def _set_cell(self,row,column,text):
        cell=QTableWidgetItem(text);cell.setToolTip(text)
        cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
        if column in (1,2,3):cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row,column,cell)

    def _update_readout(self):
        row=self.table.currentRow()
        columns=(0,4,7,8,9,10,11)
        cells=[self.table.item(row,column) for column in columns] if row>=0 else []
        if not cells or any(cell is None for cell in cells):
            self.result_readout.clear();return
        file,event,status,laeq,nmean,nmax,notes=(cell.text() for cell in cells)
        self.result_readout.setPlainText(f'{file} · {event} · {status}\n'
            f'LAeq  {laeq}    |    Nmean  {nmean}    |    Nmax  {nmax}\n{notes}')

    def _update_summary(self,by_id=None):
        counts={'files':len({str(item.path) for item in self.specifications}),
            'items':len(self.specifications),'completed':0,'partial':0,'failed':0,'cancelled':0}
        groups={'ok':'completed','complete':'completed','partial':'partial','failed':'failed','cancelled':'cancelled'}
        for spec in self.specifications:
            item=(by_id or {}).get(spec.item_id)
            if item and item.status in groups:counts[groups[item.status]]+=1
        for name,value in counts.items():self.count_labels[name].setText(str(value))
        text=f'{counts["files"]} 文件 · {counts["items"]} 分析条目'
        if self.result is not None:
            text+=f' · {counts["completed"]} 已完成 · {counts["partial"]} 部分指标失败 · {counts["failed"]} 失败 · {counts["cancelled"]} 已取消'
        self.summary.setText(text)

    def _result_metrics(self,item):
        metrics=item.result.summary.metrics if item.result else {}
        values=[];notes=[]
        for key,unit in [('LAeq','dB re 20 µPa'),('Nmean','sone'),('Nmax','sone')]:
            metric=metrics.get(key)
            if metric is None:
                status=item.status if item.status in {'failed','cancelled'} else 'unavailable'
                metric=MetricResult(None,unit,status,message=item.error or '本条结果未提供该指标。')
            values.append(metric_text(metric))
            if metric.message:
                notes.append(f'{key}：{metric.message}')
            elif metric.value is None:
                notes.append(f'{key}：{STATUS_LABELS.get(metric.status,metric.status)}')
        if item.error:notes.insert(0,item.error)
        if item.result:notes.extend(item.result.summary.warnings)
        return values,'；'.join(dict.fromkeys(notes)) or '本条分析已完成，指标状态见数值列。'
    def remove_current(self):
        row=self.table.currentRow()
        if row>=0:self.specifications.pop(row);self.refresh()
    def edit_current(self):
        row=self.table.currentRow()
        if row<0:return
        dialog=BatchItemDialog(self.specifications[row],self.profiles,self)
        if dialog.exec():
            try:self.specifications[row]=dialog.specification();self.refresh()
            except Exception as error:self.error.emit(str(error))
    def set_result(self,result):
        self.result=result
        by_id={item.item_id:item for item in result.items}
        for row,spec in enumerate(self.specifications):
            item=by_id.get(spec.item_id)
            if item:
                text={'ok':'已完成','complete':'已完成','partial':'部分指标失败','failed':'失败','cancelled':'已取消'}.get(item.status,item.status)
                self._set_cell(row,7,text+('：'+item.error if item.error else ''))
                metrics,notes=self._result_metrics(item)
                for column,value in enumerate(metrics,8):self._set_cell(row,column,value)
                self._set_cell(row,11,notes)
            else:
                for column,value in enumerate(['待运行','— 待运行','— 待运行','— 待运行','本次批处理未返回该条目'],7):
                    self._set_cell(row,column,value)
        self._update_summary(by_id)
        if self.table.currentRow()<0 and self.table.rowCount():self.table.setCurrentCell(0,0)
        self._update_readout()
