from dataclasses import fields
from PyQt6.QtWidgets import QDialog,QVBoxLayout,QFormLayout,QLineEdit,QScrollArea,QWidget,QDialogButtonBox
from ..model import MeasurementContext

CONTEXT_LABELS={'specimen':'试件／编号','version':'产品版本','test_level':'测试层级（裸电机／传动／座椅／装车）',
    'mechanism':'机构','supply':'供电 / 电压','supply_source':'供电来源','load':'负载','load_layout':'负载布置',
    'fixture':'安装／夹具','fastening':'紧固方案','environment':'环境','temperature':'温度状态',
    'mic_position':'测点位置','mic_distance':'测点距离','mic_direction':'麦克风方向','background':'背景记录',
    'actual_movement':'实际运动方向','motor_rotation':'电机旋向 CW/CCW','rotation_view':'旋向观察基准',
    'stroke':'行程位置／区间','cycle':'循环号','repeat':'重复号','notes':'备注'}

class ContextDialog(QDialog):
    def __init__(self, context=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('测量工况（缺失项保持未知）')
        self.resize(650,690)
        layout=QVBoxLayout(self)
        scroll=QScrollArea();scroll.setWidgetResizable(True)
        body=QWidget();form=QFormLayout(body)
        self.inputs={}
        context=context or MeasurementContext()
        for field in fields(context):
            value=getattr(context,field.name)
            widget=QLineEdit('' if value=='unknown' else value)
            widget.setObjectName('context_'+field.name)
            widget.setPlaceholderText('未知' if field.name!='notes' else '可留空')
            self.inputs[field.name]=widget
            form.addRow(CONTEXT_LABELS[field.name],widget)
        scroll.setWidget(body);layout.addWidget(scroll)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)

    def context(self):
        return MeasurementContext(**{key:widget.text().strip() or ('' if key=='notes' else 'unknown')
            for key,widget in self.inputs.items()})
