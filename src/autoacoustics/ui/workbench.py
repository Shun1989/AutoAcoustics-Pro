"""Compact measurement workbench, using the existing workflow controls."""
import math
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QScrollArea, QSizePolicy, QTabWidget, QFrame, QToolBar)
from .formatting import METRIC_LABELS, STATUS_LABELS

WORKBENCH_STYLE = '''
QMainWindow,QWidget{background:#141523;color:#dce3ef;font-size:13px;}
QLabel{background:transparent;border:0;}
QLabel[role="brand"]{color:#65d6e6;font-size:21px;font-weight:600;}
QLabel[role="muted"]{color:#9ba9be;}
QLabel[role="sectionTitle"]{color:#65d6e6;font-size:19px;font-weight:600;}
QFrame[role="metricCard"]{background:#1b1e2f;border:1px solid #30354c;border-radius:5px;}
QLabel[role="metricValue"]{color:#74dbe8;font-size:23px;font-weight:600;}
QToolBar{background:#1b1e2f;border:1px solid #30354c;spacing:4px;padding:5px;}
QGroupBox{border:1px solid #30354c;border-radius:4px;margin-top:16px;padding:8px;}
QGroupBox::title{subcontrol-origin:margin;left:9px;color:#70d4e3;}
QPushButton{background:#232a3f;border:1px solid #394760;border-radius:4px;padding:5px 8px;}
QPushButton:hover{background:#304258;}
QPushButton[role="primary"]{background:#087e94;border-color:#20b8c9;color:white;}
QPushButton:disabled{color:#77849a;background:#202334;}
QComboBox,QDoubleSpinBox,QSpinBox,QLineEdit{background:#202334;padding:4px;border:1px solid #394760;border-radius:3px;}
QTableWidget,QListWidget,QPlainTextEdit{background:#10131f;alternate-background-color:#1c2335;border:1px solid #30354c;}
QHeaderView::section{background:#252c42;color:#c9d6e7;padding:5px;border:0;}
QTabWidget::pane{border:1px solid #30354c;}
QTabBar::tab{background:#202438;padding:6px 10px;color:#aab8cd;}
QTabBar::tab:selected{background:#30465b;color:#75dce9;}
QProgressBar{border:1px solid #394760;text-align:center;}
QProgressBar::chunk{background:#0d92a7;}
QScrollArea{border:0;}
QSplitter::handle{background:#30354c;}
'''


class SourceLabel(QLabel):
    """Keep the full source text for provenance, elide only its painting."""
    def setText(self, text):
        super().setText(text)
        self.setToolTip(text)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(self.palette().windowText().color())
        lines = self.text().splitlines()
        visible = ' · '.join([Path(lines[0]).name] + lines[1:]) if lines else ''
        text = self.fontMetrics().elidedText(visible,
            Qt.TextElideMode.ElideMiddle, max(0, self.width()-8))
        painter.drawText(self.rect().adjusted(4, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter, text)


class MetricCard(QFrame):
    def __init__(self, key):
        super().__init__()
        self.key = key
        self.setProperty('role', 'metricCard')
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(1)
        self.title_label = QLabel(key)
        self.value_label = QLabel('—'); self.value_label.setProperty('role', 'metricValue')
        self.status_label = QLabel('等待分析'); self.status_label.setProperty('role', 'muted')
        for label in (self.title_label, self.value_label, self.status_label):
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            layout.addWidget(label)
        self.setToolTip(METRIC_LABELS.get(key, key))


def refresh_metric_cards(window, comparison=None, stale=False):
    if not hasattr(window, 'metric_cards'):
        return
    window._view_comparison = comparison is not None
    index = window.analysis_tabs.indexOf(window.comparison_detail_page)
    if comparison is None and window.analysis_tabs.currentIndex() == index:
        window.analysis_tabs.setCurrentIndex(0)
    window.analysis_tabs.setTabVisible(index, comparison is not None)
    export = window.findChild(QPushButton, 'export_report')
    play = window.findChild(QPushButton, 'play_segment')
    if export: export.setText('导出 A/B 报告' if comparison else '导出单文件报告')
    if play: play.setText('试听 B' if comparison else '播放选段')
    metrics = comparison.differences if comparison else (
        window.current_result.summary.metrics if window.current_result else {})
    for key, card in window.metric_cards.items():
        metric = metrics.get(key)
        card.title_label.setText(key + (' · B−A' if comparison else ''))
        if metric is None:
            card.value_label.setText('—')
            card.status_label.setText('不适用' if comparison else '等待分析')
            card.setToolTip(METRIC_LABELS.get(key, key))
            continue
        status = STATUS_LABELS.get(metric.status, metric.status)
        valid = metric.value is not None and metric.status not in {
            'failed', 'uncalibrated', 'not_applicable', 'cancelled', 'incompatible', 'undefined'}
        if valid and not math.isnan(metric.value):
            number = f'{metric.value:.4g}' if math.isfinite(metric.value) else ('−∞' if metric.value < 0 else '+∞')
            unit = 'dB' if metric.unit.startswith('dB re 20') else metric.unit
            card.value_label.setText(number + (' '+unit if unit else ''))
        else:
            card.value_label.setText('—')
        card.status_label.setText('已过期 · 请重新分析' if stale else status)
        card.setToolTip(f'{METRIC_LABELS.get(key,key)} · {metric.unit}\n{metric.method}\n{metric.message}')


def _page(*widgets):
    scroll = QScrollArea(); scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    body = QWidget(); body.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    layout = QVBoxLayout(body); layout.setContentsMargins(6, 6, 6, 6)
    for widget in widgets:
        widget.setParent(body); layout.addWidget(widget)
    layout.addStretch()
    scroll.setWidget(body)
    return scroll


def install_workbench(window):
    """Reparent controls once; retain all existing slots, models and object names."""
    old = window.takeCentralWidget()
    groups = {int(group.title()[0]): group for group in window.controls.parentWidget().findChildren(QGroupBox)}
    central = QWidget(); outer = QVBoxLayout(central)
    outer.setContentsMargins(10, 8, 10, 6); outer.setSpacing(7)
    header = QHBoxLayout()
    brand = QLabel('AutoAcoustics Pro'); brand.setProperty('role', 'brand')
    header.addWidget(brand)
    subtitle = QLabel('座椅电机 · 现场 NVH'); subtitle.setProperty('role', 'muted');
    subtitle.setMinimumWidth(0);subtitle.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred);header.addWidget(subtitle)
    header.addStretch()
    live_button=window._button('麦克风 / 声卡','soundcard_entry',window.open_soundcard)
    live_button.setProperty('role','primary');header.addWidget(live_button)
    header.addWidget(window._button('转频 / 阶次','rotation_entry',window.open_rotation))
    for name in ('daq_entry', 'knowledge_entry'):
        button=old.findChild(QPushButton,name)
        if name=='daq_entry':button.setText('NI / HEAD')
        header.addWidget(button)
    outer.addLayout(header)

    toolbar = QToolBar(); toolbar.setObjectName('measurement_toolbar'); toolbar.setMovable(False)
    titles = {'open_file':'导入录音','open_project':'打开项目','save_project':'保存项目',
        'analyze':'开始分析','cancel_job':'取消','play_segment':'播放选段','pause_audio':'暂停',
        'stop_audio':'停止','export_report':'导出报告','add_batch_item':'加入批处理'}
    for name, title in titles.items():
        button = old.findChild(QPushButton, name); button.setText(title)
        if name == 'analyze': button.setProperty('role', 'primary')
        toolbar.addWidget(button)
    window.progress.setMaximumWidth(105); window.progress.setMinimumWidth(70)
    toolbar.addSeparator(); toolbar.addWidget(window.progress)
    outer.addWidget(toolbar)

    source = QHBoxLayout(); source.setSpacing(8)
    original = window.file_label
    label = SourceLabel(original.text()); label.setObjectName('source_path')
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    label.setMinimumWidth(0); original.setParent(None); original.deleteLater()
    window.file_label = label; source.addWidget(label, 1)
    source.addWidget(QLabel('通道'))
    window.channel_combo.setMinimumWidth(100); window.channel_combo.setMaximumWidth(220)
    source.addWidget(window.channel_combo)
    source.addWidget(QLabel('校准'))
    window.profile_combo.setMinimumWidth(160); window.profile_combo.setMaximumWidth(310)
    source.addWidget(window.profile_combo)
    outer.addLayout(source)

    window.analysis_tabs = QTabWidget(); window.analysis_tabs.setObjectName('analysis_panel_tabs')
    window.analysis_tabs.setUsesScrollButtons(True)
    result_page = QWidget(); result_layout = QVBoxLayout(result_page)
    result_layout.setContentsMargins(4, 4, 4, 4)
    provenance = QScrollArea(); provenance.setWidgetResizable(True); provenance.setMaximumHeight(54)
    window.result_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    provenance.setWidget(window.result_label); result_layout.addWidget(provenance)
    window.metric_table.setMaximumHeight(16777215)
    window.metric_table.verticalHeader().setVisible(False)
    window.metric_table.verticalHeader().setDefaultSectionSize(25)
    window.metric_table.setAlternatingRowColors(True)
    result_layout.addWidget(window.metric_table, 1)
    window.analysis_tabs.addTab(result_page, '结果 / 质量')
    window.analysis_tabs.addTab(_page(groups[4]), '选段 / 工况')
    window.analysis_tabs.addTab(_page(groups[1], groups[2], groups[3]), '校准 / 来源')
    window.analysis_tabs.addTab(_page(groups[5]), '参数 / 试听')
    window.analysis_tabs.addTab(_page(groups[6]), '项目 / 报告')
    window.comparison_detail_page = QWidget()
    comparison_layout = QVBoxLayout(window.comparison_detail_page)
    comparison_layout.setContentsMargins(4, 4, 4, 4)
    playback = QHBoxLayout()
    for name in ('play_a','play_b','comparison_report'):
        playback.addWidget(old.findChild(QPushButton, name))
    comparison_layout.addLayout(playback)
    comparison_notes = QScrollArea(); comparison_notes.setWidgetResizable(True)
    comparison_notes.setMaximumHeight(40)
    comparison_notes.setToolTip('滚动查看 A/B 来源与完整条件记录')
    window.comparison_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    comparison_notes.setWidget(window.comparison_label)
    comparison_layout.addWidget(comparison_notes)
    comparison_layout.addWidget(window.diff_table, 1)
    window.diff_table.verticalHeader().setVisible(False)
    window.diff_table.verticalHeader().setDefaultSectionSize(25)
    window.analysis_tabs.addTab(window.comparison_detail_page, 'A/B 差值')
    window.plots.set_analysis_panel(window.analysis_tabs)

    analysis = window.workspace.widget(0)
    layout = analysis.layout()
    while layout.count(): layout.takeAt(0)
    layout.setContentsMargins(0, 7, 0, 0); layout.setSpacing(7)
    cards = QHBoxLayout(); cards.setSpacing(7)
    window.metric_cards = {}
    for key in ('duration','LAeq','LZeq','LAFmax','Nmean','Nmax'):
        card = MetricCard(key); window.metric_cards[key] = card; cards.addWidget(card)
    layout.addLayout(cards)
    window.plots.setParent(analysis); layout.addWidget(window.plots, 1)
    window.workspace.setParent(central); outer.addWidget(window.workspace, 1)
    window.workspace.setTabText(1, 'A/B 比较 · 结果库')
    window.workspace.setTabText(2, '批处理 · 报告')
    window.soundcard_page=QWidget();live_layout=QVBoxLayout(window.soundcard_page)
    live_layout.setContentsMargins(0,4,0,0)
    window.rotation_page=QWidget();nvh_layout=QVBoxLayout(window.rotation_page)
    nvh_layout.setContentsMargins(0,4,0,0)
    window.workspace.addTab(window.soundcard_page,'现场采集 · 声卡')
    window.workspace.addTab(window.rotation_page,'转频 · 阶次诊断')
    window.workspace.currentChanged.connect(lambda index:
        window.open_soundcard() if window.workspace.widget(index) is window.soundcard_page
        else window.open_rotation() if window.workspace.widget(index) is window.rotation_page else None)
    window.status_label.setParent(central)
    window.status_label.setWordWrap(False)
    window.status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    window.status_label.setProperty('role', 'muted')
    outer.addWidget(window.status_label)
    window.setCentralWidget(central)
    window.statusBar().hide()
    old.deleteLater()
    refresh_metric_cards(window)


def restore_workbench(window):
    for name in ('dashboard_splitter', 'top_splitter', 'bottom_splitter'):
        state = window.preferences.value('workbench/'+name)
        if state is not None:
            getattr(window.plots, name).restoreState(state)
    time = window.preferences.value('workbench/time', 'waveform')
    frequency = window.preferences.value('workbench/frequency', 'spectrum')
    if time in ('waveform','spl','loudness') and frequency in ('spectrum','psd','octave','bark'):
        window.plots.select_dashboard(time_key=time, frequency_key=frequency)


def save_workbench(window):
    for name in ('dashboard_splitter', 'top_splitter', 'bottom_splitter'):
        window.preferences.setValue('workbench/'+name, getattr(window.plots, name).saveState())
    window.preferences.setValue('workbench/time', window.plots.time_combo.currentData())
    window.preferences.setValue('workbench/frequency', window.plots.frequency_combo.currentData())
