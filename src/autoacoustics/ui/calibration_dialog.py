from datetime import datetime, timezone
from PyQt6.QtWidgets import (QDialog,QVBoxLayout,QFormLayout,QLineEdit,QDoubleSpinBox,
    QTabWidget,QWidget,QLabel,QDialogButtonBox)
from ..model import CalibrationProfile

def numeric(value, maximum=1e12):
    spin=QDoubleSpinBox()
    spin.setDecimals(9)
    spin.setRange(1e-12,maximum)
    spin.setValue(value)
    return spin

class CalibrationDialog(QDialog):
    def __init__(self, signal, channel, start=0, end=None, parent=None):
        super().__init__(parent)
        self.signal,self.channel,self.start,self.end=signal,channel,start,end
        self.setWindowTitle('建立有来源的校准配置')
        self.resize(580,460)
        layout=QVBoxLayout(self)
        self.name=QLineEdit('测量链校准配置')
        form=QFormLayout()
        form.addRow('配置名称',self.name)
        form.addRow('源单位',QLabel(signal.channels[channel].unit))
        layout.addLayout(form)
        self.tabs=QTabWidget()
        layout.addWidget(self.tabs)
        manual=QWidget(); mf=QFormLayout(manual)
        self.coefficient=numeric(1)
        self.source=QLineEdit()
        self.source.setPlaceholderText('填写测量记录、证书或参考数据来源')
        mf.addRow('Pa / 原始单位',self.coefficient)
        mf.addRow('换算来源（必填）',self.source)
        mf.addRow(QLabel('人工系数不是独立声校准证明。已为 Pa 的数据保留文件声明，不能重复缩放。'))
        self.tabs.addTab(manual,'有记录的换算系数')
        sensor=QWidget(); sf=QFormLayout(sensor)
        self.sensitivity=numeric(50)
        self.gain=numeric(1)
        self.full_scale=numeric(10)
        self.sensor_ref=QLineEdit()
        sf.addRow('灵敏度 / mV/Pa',self.sensitivity)
        sf.addRow('测量链线性增益',self.gain)
        sf.addRow('数字信号 V/full-scale',self.full_scale)
        sf.addRow('传感器／证书引用',self.sensor_ref)
        self.tabs.addTab(sensor,'传感器链')
        tone=QWidget(); tf=QFormLayout(tone)
        self.level=numeric(94,200)
        self.frequency=numeric(1000,100000)
        tf.addRow('校准器证书参考声级 / dB',self.level)
        tf.addRow('校准音频率 / Hz',self.frequency)
        tf.addRow(QLabel('使用当前录音与选段；必须是至少0.5秒稳定校准音。\n请先打开校准录音，建立并保存配置，再打开待测录音使用。'))
        self.tabs.addTab(tone,'独立声校准录音')
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def profile(self):
        from ..calibration import make_sensitivity_profile,make_calibrator_profile
        info=self.signal.channels[self.channel]
        mode=self.tabs.currentIndex()
        if mode==1:
            return make_sensitivity_profile(self.name.text(),self.sensitivity.value(),self.gain.value(),
                full_scale_v=self.full_scale.value() if info.unit=='FS' else None,input_unit=info.unit,
                channel=self.channel,channel_name=info.name,sensor_ref=self.sensor_ref.text())
        if mode==2:
            return make_calibrator_profile(self.signal,self.level.value(),channel=self.channel,
                start=self.start,end=self.end,frequency=self.frequency.value(),name=self.name.text())
        if not self.source.text().strip():
            raise ValueError('必须填写校准来源。')
        if info.unit=='Pa' and self.coefficient.value()!=1:
            raise ValueError('文件已经为 Pa，不能再次换算。')
        return CalibrationProfile(self.name.text(),self.source.text().strip(),self.coefficient.value(),info.unit,
            channel=self.channel,channel_name=info.name,date=datetime.now(timezone.utc).isoformat(),
            details={'manual':True,'source_record':self.source.text().strip()})

    def worker_parameters(self):
        """Only the file-based worker performs reference-tone FFT and RMS."""
        return dict(known_level_db=self.level.value(),channel=self.channel,start=self.start,
            end=self.end,frequency=self.frequency.value(),name=self.name.text())
