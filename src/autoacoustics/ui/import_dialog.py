from PyQt6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QLineEdit, QComboBox, QLabel, QDialogButtonBox
from ..model import ImportMapping

class ImportDialog(QDialog):
    def __init__(self, parent=None, options=None, mapping=None):
        super().__init__(parent)
        self.setWindowTitle('确认数据字段、单位与时间轴')
        self.resize(570, 530)
        layout = QVBoxLayout(self)
        hint = QLabel('科学文件和表格须明确数值字段。单位声明须来自文件或测量记录；未知单位仍可看相对图。\n字段未确定时保留空白，导入器会给出可选字段。视频多音轨需明确音轨序号。')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        if options:
            hint2 = QLabel(str(options))
            hint2.setWordWrap(True)
            layout.addWidget(hint2)
        form = QFormLayout()
        layout.addLayout(form)
        self.rate = QLineEdit('' if not mapping or mapping.sample_rate is None else str(mapping.sample_rate))
        self.dataset = QLineEdit('' if not mapping else mapping.dataset or '')
        self.columns = QLineEdit('' if not mapping else ','.join(map(str,mapping.data_columns)))
        self.time_column = QLineEdit('' if not mapping or mapping.time_column is None else str(mapping.time_column))
        self.time_unit = QComboBox()
        for title, value in [('文件有声明时识别；否则必须选择',None),('秒 / s','s'),('毫秒 / ms','ms'),('微秒 / us','us')]:
            self.time_unit.addItem(title,value)
        if mapping:self.time_unit.setCurrentIndex(self.time_unit.findData(mapping.time_unit))
        self.names = QLineEdit('' if not mapping else ','.join(mapping.channels))
        self.units = QLineEdit('' if not mapping else ','.join(mapping.channel_units))
        self.delimiter = QLineEdit('' if not mapping else mapping.delimiter or '')
        self.axis = QComboBox()
        for text, value in [('自动识别（无歧义时）',None),('行是通道',0),('列是通道',1)]:
            self.axis.addItem(text,value)
        self.unit = QComboBox()
        self.unit.addItems(['unknown','FS','V','Pa','m/s²','g'])
        if mapping:
            self.unit.setCurrentText(mapping.unit)
            self.axis.setCurrentIndex(self.axis.findData(mapping.channel_axis))
        self.stream = QLineEdit('' if not mapping or mapping.audio_stream_index is None else str(mapping.audio_stream_index))
        for label, widget in [('采样率 / Hz（已有时间列可留空）',self.rate),('MAT 变量／HDF5 数据集路径',self.dataset),
            ('表格数值列名或序号（逗号分隔）',self.columns),('时间列名或序号',self.time_column),('时间列单位',self.time_unit),('矩阵通道名称／TDMS通道全路径',self.names),
            ('公共源单位',self.unit),('逐通道单位（逗号分隔，可留空）',self.units),('通道轴',self.axis),
            ('表格分隔符（空为自动）',self.delimiter),('媒体音轨 stream_index',self.stream)]:
            form.addRow(label,widget)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def mapping(self):
        def column(text):
            text = text.strip()
            return int(text) if text.isdigit() else text
        return ImportMapping(sample_rate=float(self.rate.text()) if self.rate.text().strip() else None,
            unit=self.unit.currentText(), channel_units=tuple(x.strip() for x in self.units.text().split(',') if x.strip()),
            dataset=self.dataset.text().strip() or None,
            channels=tuple(x.strip() for x in self.names.text().split(',') if x.strip()),
            time_column=column(self.time_column.text()) if self.time_column.text().strip() else None,
            data_columns=tuple(column(x) for x in self.columns.text().split(',') if x.strip()),
            channel_axis=self.axis.currentData(), audio_stream_index=int(self.stream.text()) if self.stream.text().strip() else None,
            delimiter=self.delimiter.text() or None,time_unit=self.time_unit.currentData())
