"""Consume shared details; display decimation preserves extrema."""
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal, Qt, QEvent
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QStackedWidget,
    QSplitter, QFrame, QLabel, QComboBox, QPushButton, QDialog,
)


PLOT_TITLES = {
    'waveform': '波形 / 选段', 'spectrum': '幅度谱', 'psd': 'PSD',
    'stft': '语谱图', 'octave': '1/3 倍频程', 'spl': 'SPL 历程',
    'loudness': '响度历程', 'bark': '特征响度 / Bark',
}
TIME_PLOTS = ('waveform', 'spl', 'loudness')
FREQUENCY_PLOTS = ('spectrum', 'psd', 'octave', 'bark')


class _OctaveFrequencyAxis(pg.AxisItem):
    """Label logarithmic positions as frequencies in Hz, without powers of ten."""
    def logTickStrings(self, values, scale, spacing):
        if spacing is None or spacing < 1:
            # Wide axes can expose only pyqtgraph's minor tick level. Keep its
            # integer decades, so 100/1000/10000 remain without dense text.
            return [f'{10. ** value:g}' if np.isclose(value, round(value), rtol=0, atol=1e-9)
                    else '' for value in values]
        return [f'{10. ** value:g}' for value in values]


class _PlotTabs(QTabWidget):
    """Honor existing setCurrentIndex calls even when that index is selected."""
    focusRequested = pyqtSignal(int)

    def setCurrentIndex(self, index):
        previous = self.currentIndex()
        super().setCurrentIndex(index)
        if previous == index and 0 <= index < self.count():
            self.focusRequested.emit(index)


class _DashboardSection(QFrame):
    def __init__(self, title, name, keys=(), parent=None):
        super().__init__(parent)
        self.setObjectName('dashboard_' + name)
        self.setProperty('dashboardSection', True)
        self.setMinimumSize(180, 150)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(7, 5, 7, 7)
        layout.setSpacing(5)
        header = QHBoxLayout()
        self.title = QLabel(title)
        self.title.setObjectName('dashboardTitle')
        header.addWidget(self.title)
        header.addStretch()
        self.selector = None
        if keys:
            self.selector = QComboBox()
            self.selector.setObjectName(name + '_plot_selector')
            for key in keys:
                self.selector.addItem(PLOT_TITLES[key], key)
            header.addWidget(self.selector)
        self.expand_button = QPushButton('放大')
        self.expand_button.setObjectName('expand_' + name)
        self.expand_button.setToolTip('在独立窗口中放大；关闭后返回当前视图')
        header.addWidget(self.expand_button)
        layout.addLayout(header)
        self.content = QWidget()
        self.content.setObjectName(name + '_plot_host')
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.content, 1)

def envelope_plot_data(samples, sample_rate, max_points=20000, origin=0.):
    values = np.asarray(samples)
    if values.size <= max_points:
        return origin + np.arange(values.size) / sample_rate, values
    bins = max(1, max_points // 2)
    edges = np.linspace(0, len(values), bins + 1, dtype=int)
    indices = []
    for left, right in zip(edges[:-1], edges[1:]):
        block = values[left:right]
        indices.extend(sorted((left + int(np.argmin(block)), left + int(np.argmax(block)))))
    indices = np.asarray(indices)
    return origin + indices / sample_rate, values[indices]

def envelope_xy(x, y, max_points=20000):
    """Display-only min/max envelope; retain peak coordinates on any x grid."""
    x,y=np.asarray(x),np.asarray(y)
    if len(x)<=max_points:
        return x,y
    edges=np.linspace(0,len(x),max_points//2+1,dtype=int)
    indices=[]
    for left,right in zip(edges[:-1],edges[1:]):
        block=y[left:right]
        indices.extend(sorted((left+int(np.argmin(block)),left+int(np.argmax(block)))))
    indices=np.asarray(indices)
    return x[indices],y[indices]

def compact_image(times, values, max_columns=1200):
    """Bound raster work, retain transient maxima within each display time bin."""
    times,values=np.asarray(times),np.asarray(values)
    if len(times)<=max_columns:
        return times,values
    edges=np.linspace(0,len(times),max_columns+1,dtype=int)
    result=np.empty((values.shape[0],max_columns),dtype=values.dtype)
    centers=np.empty(max_columns)
    for column,(left,right) in enumerate(zip(edges[:-1],edges[1:])):
        result[:,column]=np.max(values[:,left:right],axis=1)
        centers[column]=(times[left]+times[right-1])/2
    return centers,result

def _finite_line(x,y):
    x,y=envelope_xy(x,y)
    mask=np.isfinite(x)&np.isfinite(y)
    return x[mask],y[mask]

class PlotPanel(QWidget):
    selectionChanged = pyqtSignal(float, float)
    def __init__(self, parent=None):
        super().__init__(parent)
        pg.setConfigOptions(antialias=True, background='#101018', foreground='#d9dfed')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setObjectName('plotPanel')
        self.setStyleSheet('''
            QWidget#plotPanel { background: #141523; }
            QDialog { background: #141523; color: #d9dfed; }
            QLabel { color: #d9dfed; }
            QFrame[dashboardSection="true"] { background: #141523; border: 1px solid #303246; }
            QLabel#dashboardTitle { color: #66d2e8; font-weight: 600; border: none; }
            QComboBox { background: #202334; color: #d9dfed; border: 1px solid #3a4257; padding: 3px 7px; }
            QPushButton { background: #22283b; color: #77d8e8; border: 1px solid #39445b; padding: 3px 9px; }
            QPushButton:hover { background: #303d52; }
            QTabWidget::pane { border: 1px solid #303246; background: #141523; }
            QTabBar::tab { background: #222438; color: #bfcadd; padding: 6px 10px; }
            QTabBar::tab:selected { background: #304257; color: #77d8e8; }
            QSplitter::handle { background: #25283a; }
        ''')
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)
        self.dashboard = QWidget()
        dashboard_layout = QVBoxLayout(self.dashboard)
        dashboard_layout.setContentsMargins(0, 0, 0, 0)
        self.dashboard_splitter = QSplitter(Qt.Orientation.Vertical)
        self.top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.sections = {
            'time': _DashboardSection('TIME HISTORY  时间历程', 'time', TIME_PLOTS),
            'frequency': _DashboardSection('FREQUENCY  频域分析', 'frequency', FREQUENCY_PLOTS),
            'stft': _DashboardSection('SPECTROGRAM  语谱图', 'stft'),
            'analysis': _DashboardSection('ANALYSIS  分析面板', 'analysis'),
        }
        self.top_splitter.addWidget(self.sections['time'])
        self.top_splitter.addWidget(self.sections['frequency'])
        self.bottom_splitter.addWidget(self.sections['stft'])
        self.bottom_splitter.addWidget(self.sections['analysis'])
        self.dashboard_splitter.addWidget(self.top_splitter)
        self.dashboard_splitter.addWidget(self.bottom_splitter)
        for splitter in (self.dashboard_splitter, self.top_splitter, self.bottom_splitter):
            splitter.setChildrenCollapsible(False)
            splitter.setHandleWidth(5)
            splitter.setSizes([500, 500])
        dashboard_layout.addWidget(self.dashboard_splitter)
        self.stack.addWidget(self.dashboard)
        self.tabs = _PlotTabs()
        self.stack.addWidget(self.tabs)
        self.plots = {}
        self._tab_layouts = {}
        self._mode = 'dashboard'
        self._time_key = 'waveform'
        self._frequency_key = 'spectrum'
        self._dialogs = {}
        self._popout_windows = {}
        self._observed_window = None
        for key, title in PLOT_TITLES.items():
            plot = pg.PlotWidget(axisItems={'bottom': _OctaveFrequencyAxis('bottom')}) if key == 'octave' else pg.PlotWidget()
            plot.setObjectName('plot_' + key)
            plot.setBackground('#101018')
            plot.showGrid(x=True, y=True, alpha=.18)
            self.plots[key] = plot
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(7, 5, 7, 7)
            header = QHBoxLayout()
            title_label = QLabel(title)
            title_label.setObjectName('dashboardTitle')
            header.addWidget(title_label)
            header.addStretch()
            dashboard_button = QPushButton('同屏视图')
            dashboard_button.setObjectName('dashboard_from_' + key)
            dashboard_button.clicked.connect(self.show_dashboard)
            header.addWidget(dashboard_button)
            expand_button = QPushButton('放大')
            expand_button.setObjectName('expand_' + key + '_single')
            expand_button.clicked.connect(lambda checked=False, key=key: self.pop_out(key))
            header.addWidget(expand_button)
            page_layout.addLayout(header)
            host = QWidget()
            host_layout = QVBoxLayout(host)
            host_layout.setContentsMargins(0, 0, 0, 0)
            host_layout.addWidget(plot)
            self._tab_layouts[key] = host_layout
            page_layout.addWidget(host, 1)
            self.tabs.addTab(page, title)
        self.plots['octave'].setLogMode(x=True)
        self.plots['octave'].getAxis('bottom').enableAutoSIPrefix(False)
        self._octave_legend = self.plots['octave'].addLegend(
            offset=(8, 8), labelTextColor='#d9dfed',
            brush=pg.mkBrush(20, 21, 35, 210), pen=pg.mkPen('#303246'))
        self._octave_legend.hide()
        self.region = pg.LinearRegionItem([0, 1], brush=pg.mkBrush(48,160,220,35))
        self.region.sigRegionChangeFinished.connect(lambda: self.selectionChanged.emit(*self.region.getRegion()))
        self.cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('#f6c65b', width=2))
        self._overlay_items = []
        self._rotation_items = []
        self.colorbars = {}
        self.time_combo = self.sections['time'].selector
        self.frequency_combo = self.sections['frequency'].selector
        self.time_combo.currentIndexChanged.connect(
            lambda index: self.select_dashboard(time_key=self.time_combo.itemData(index)))
        self.frequency_combo.currentIndexChanged.connect(
            lambda index: self.select_dashboard(frequency_key=self.frequency_combo.itemData(index)))
        self.sections['time'].expand_button.clicked.connect(lambda: self.pop_out(self._time_key))
        self.sections['frequency'].expand_button.clicked.connect(lambda: self.pop_out(self._frequency_key))
        self.sections['stft'].expand_button.clicked.connect(lambda: self.pop_out('stft'))
        self.sections['analysis'].expand_button.clicked.connect(lambda: self.pop_out('analysis'))
        self.tabs.currentChanged.connect(self._focus_tab)
        self.tabs.focusRequested.connect(self._focus_tab)
        self.analysis_panel = QLabel('导入信号并运行分析后，在此查看分析结果。')
        self.analysis_panel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.analysis_panel.setWordWrap(True)
        self.sections['analysis'].content_layout.addWidget(self.analysis_panel)
        self.show_dashboard()

    @staticmethod
    def _place_widget(widget, destination):
        """Move a live widget, keeping its graphics scene and ViewBox intact."""
        if destination.indexOf(widget) >= 0:
            return
        parent = widget.parentWidget()
        if parent is not None and parent.layout() is not None:
            parent.layout().removeWidget(widget)
        destination.addWidget(widget)
        widget.show()

    def _dock_plots(self):
        selected = {self._time_key: 'time', self._frequency_key: 'frequency', 'stft': 'stft'}
        for key, plot in self.plots.items():
            if key in self._dialogs:
                continue
            if self._mode == 'dashboard' and key in selected:
                destination = self.sections[selected[key]].content_layout
            else:
                destination = self._tab_layouts[key]
            self._place_widget(plot, destination)

    def set_analysis_panel(self, widget):
        """Mount the application's analysis controls in the fourth quadrant."""
        if not isinstance(widget, QWidget):
            raise TypeError('分析面板必须为 QWidget。')
        if widget is self.analysis_panel:
            return
        if 'analysis' in self._dialogs:
            self._dialogs['analysis'].close()
        previous = self.analysis_panel
        self.sections['analysis'].content_layout.removeWidget(previous)
        previous.hide()
        previous.setParent(self)
        self.analysis_panel = widget
        self._place_widget(widget, self.sections['analysis'].content_layout)

    def show_dashboard(self):
        """Return selected time, frequency and STFT graphs to the workbench."""
        self._mode = 'dashboard'
        self._dock_plots()
        self.stack.setCurrentWidget(self.dashboard)

    def select_dashboard(self, time_key=None, frequency_key=None):
        """Choose existing graphs for the two selectable dashboard regions."""
        if time_key is not None and time_key not in TIME_PLOTS:
            raise ValueError('时间历程请选择 waveform、spl 或 loudness。')
        if frequency_key is not None and frequency_key not in FREQUENCY_PLOTS:
            raise ValueError('频域分析请选择 spectrum、psd、octave 或 bark。')
        for key, combo, attribute in (
            (time_key, self.time_combo, '_time_key'),
            (frequency_key, self.frequency_combo, '_frequency_key'),
        ):
            if key is not None:
                setattr(self, attribute, key)
                previous = combo.blockSignals(True)
                combo.setCurrentIndex(combo.findData(key))
                combo.blockSignals(previous)
        self.show_dashboard()

    def _focus_tab(self, index):
        if 0 <= index < len(PLOT_TITLES):
            self.focus_plot(tuple(PLOT_TITLES)[index])

    def focus_plot(self, key):
        """Show a graph in its original full size, eight-tab single view."""
        if key not in self.plots:
            raise ValueError('未知图表：' + str(key))
        self._mode = 'tabs'
        if key in self._dialogs:
            self._dialogs[key].close()
        previous = self.tabs.blockSignals(True)
        self.tabs.setCurrentIndex(tuple(PLOT_TITLES).index(key))
        self.tabs.blockSignals(previous)
        self._dock_plots()
        self.stack.setCurrentWidget(self.tabs)

    def pop_out(self, key):
        """Move a graph or analysis controls into one nonmodal window."""
        if key not in self.plots and key != 'analysis':
            raise ValueError('未知面板：' + str(key))
        self._observe_window()
        if key in self._dialogs:
            dialog = self._dialogs[key]
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            return dialog
        widget = self.analysis_panel if key == 'analysis' else self.plots[key]
        dialog = self._popout_windows.get(key)
        if dialog is None:
            dialog = QDialog(self)
            dialog.setObjectName('popout_' + key)
            dialog.setWindowTitle('分析面板' if key == 'analysis' else PLOT_TITLES[key])
            dialog.setModal(False)
            dialog.resize(1040, 700)
            popup_layout = QVBoxLayout(dialog)
            popup_layout.setContentsMargins(8, 8, 8, 8)
            return_button = QPushButton('返回原视图')
            return_button.clicked.connect(dialog.close)
            popup_layout.addWidget(return_button, 0, Qt.AlignmentFlag.AlignRight)
            dialog.finished.connect(lambda result, key=key: self._restore_popout(key))
            self._popout_windows[key] = dialog
        self._dialogs[key] = dialog
        self._place_widget(widget, dialog.layout())
        dialog.show()
        return dialog

    def _restore_popout(self, key):
        if self._dialogs.pop(key, None) is None:
            return
        if key == 'analysis':
            self._place_widget(self.analysis_panel, self.sections['analysis'].content_layout)
        else:
            self._dock_plots()

    def _close_popouts(self):
        for dialog in tuple(getattr(self, '_dialogs', {}).values()):
            dialog.close()

    def showEvent(self, event):
        super().showEvent(event)
        self._observe_window()

    def _observe_window(self):
        """Watch the owner even when a graph is opened before its first show."""
        window = self.window()
        if window is not self._observed_window:
            if self._observed_window is not None:
                self._observed_window.removeEventFilter(self)
            self._observed_window = window
            if window is not self:
                window.installEventFilter(self)

    def eventFilter(self, watched, event):
        # Qt may deliver a parent event while Python is clearing a cyclic
        # widget's state. Exceptions escaping a Qt callback abort the process.
        if watched is getattr(self, '_observed_window', None) and event.type() == QEvent.Type.Close:
            self._close_popouts()
        return super().eventFilter(watched, event)

    def closeEvent(self, event):
        self._close_popouts()
        super().closeEvent(event)

    def clear(self):
        self.clear_rotation_reference()
        for plot in self.plots.values():
            plot.clear()
        for bar in self.colorbars.values():bar.setImageItem([]);bar.hide()
        self._octave_legend.hide()
        self._overlay_items.clear()

    def set_signal(self, signal, channel):
        self.clear()
        plot = self.plots['waveform']
        time, values = envelope_plot_data(signal.samples[channel], signal.sample_rate,origin=signal.time_origin)
        plot.plot(time, values, pen=pg.mkPen('#4cc7d6'))
        plot.setLabel('bottom', '时间', units='s')
        plot.setLabel('left', '幅值', units=signal.channels[channel].unit)
        self.region.setBounds((signal.time_origin, signal.time_origin+signal.duration))
        self.region.setRegion((signal.time_origin, signal.time_origin+signal.duration))
        self.cursor.setValue(signal.time_origin)
        plot.addItem(self.region)
        plot.addItem(self.cursor)

    def set_selection(self, start_s, end_s):
        self.region.setRegion((start_s, end_s))

    def set_result(self, result):
        self.clear_rotation_reference()
        for key in ('spectrum','psd','stft','octave','spl','loudness','bark'):
            self.plots[key].clear()
        for bar in self.colorbars.values():bar.setImageItem([]);bar.hide()
        self._octave_legend.hide()
        if result.detail is None:return
        arrays, units = result.detail.arrays, result.detail.units
        for kind, xkey, ykey, xlabel, ylabel in [
            ('spectrum','frequency','spectrum_db','频率 / Hz','幅度 / dB'),
            ('psd','psd_frequency','psd','频率 / Hz','PSD'),
            ('spl','spl_time','laf','时间 / s','Fast A 声级 / dB'),
            ('loudness','loudness_time','loudness','时间 / s','时变响度 / sone')]:
            if xkey in arrays and ykey in arrays:
                x, y = np.asarray(arrays[xkey]), np.asarray(arrays[ykey])
                x,y=_finite_line(x,y)
                self.plots[kind].plot(x,y,pen=pg.mkPen('#4cc7d6'))
            self.plots[kind].setLabel('bottom', xlabel)
            self.plots[kind].setLabel('left', ylabel, units=units.get(ykey, ''))
        self.plots['octave'].setLabel('bottom', '精确中心频率', units='Hz')
        self.plots['octave'].setLabel('left', '声级', units=units.get('octave_levels', 'dB'))
        if 'octave_centers' in arrays and 'octave_levels' in arrays:
            self._octave_bars(arrays['octave_centers'], arrays['octave_levels'])
        if 'spl_time' in arrays and 'las' in arrays:
            x,y=np.asarray(arrays['spl_time']),np.asarray(arrays['las'])
            x,y=_finite_line(x,y)
            self.plots['spl'].plot(x,y,pen=pg.mkPen('#91c975',style=Qt.PenStyle.DashLine),name='LAS')
        self.plots['spl'].setLabel('left','LAF 实线 / LAS 虚线',units='dB(A)')
        for key,xkey,ykey,zkey,ylabel,yunit in [('stft','stft_time','stft_frequency','stft_db','频率','Hz'),
            ('bark','loudness_time','bark','specific_loudness','临界带率','Bark')]:
            if all(k in arrays for k in (xkey,ykey,zkey)):
                self._image(key,arrays[xkey],arrays[ykey],arrays[zkey],ylabel,yunit,units.get(zkey,''))

    def clear_rotation_reference(self):
        for key,item in self._rotation_items:
            self.plots[key].removeItem(item)
        self._rotation_items.clear()

    def set_rotation_reference(self,payload):
        """Physical reference frequencies, kept separate from observed peaks."""
        self.clear_rotation_reference()
        if payload.get('stale'):return
        times=np.asarray(payload.get('rpm_times',[]),dtype=float)
        rpm=np.asarray(payload.get('rpm_values',[]),dtype=float)
        tracked=payload.get('rpm_mode') in {'measured','tracked'} and len(times)>1 and times.shape==rpm.shape and np.isfinite(times).all() and np.isfinite(rpm).all()
        result=payload.get('result')
        nyquist=float(getattr(result,'sample_rate',payload.get('sample_rate',float('inf'))))/2
        colors=('#f6c65b','#e79a63','#bf8ee5','#93c975','#e582ba')
        for index,marker in enumerate(payload.get('markers',[])[:12]):
            frequency=float(marker['frequency']);order=float(marker['order'])
            if not np.isfinite(frequency) or frequency<=0 or frequency>nyquist:continue
            label=str(marker['label'])+(' · 中位转速' if tracked else '')
            color=colors[index%len(colors)];pen=pg.mkPen(color,width=1,style=Qt.PenStyle.DashLine)
            vertical=pg.InfiniteLine(pos=frequency,angle=90,pen=pen,label=label,
                labelOpts={'position':.92,'color':color,'rotateAxis':(1,0)})
            self.plots['spectrum'].addItem(vertical);self._rotation_items.append(('spectrum',vertical))
            if tracked:
                item=self.plots['stft'].plot(times,order*rpm/60,pen=pen,name=str(marker['label']))
            else:
                item=pg.InfiniteLine(pos=frequency,angle=0,pen=pen,label=str(marker['label']),
                    labelOpts={'position':.88,'color':color})
                self.plots['stft'].addItem(item)
            self._rotation_items.append(('stft',item))

    def _octave_bars(self, centers, levels):
        """Display each provided band level; positions alone use log10(Hz)."""
        centers, levels = np.asarray(centers), np.asarray(levels)
        valid = np.isfinite(centers) & (centers > 0) & np.isfinite(levels)
        if not valid.any():
            return
        self.plots['octave'].setLogMode(x=True, y=False)
        bars = pg.BarGraphItem(
            x=np.log10(centers[valid]), height=levels[valid], width=.08,
            brush=pg.mkBrush(76, 199, 214, 190), pen=pg.mkPen('#70d4e3'), name='声级')
        self.plots['octave'].addItem(bars)
        self._octave_legend.show()

    def _image(self,key,xaxis,yaxis,values,ylabel,yunit,zunit):
        from PyQt6.QtCore import QRectF
        x,y,data=np.asarray(xaxis),np.asarray(yaxis),np.asarray(values)
        if not data.size or not len(x) or not len(y):return
        if data.shape!=(len(y),len(x)):raise ValueError('图像数据与独立时间/频率轴不一致。')
        original_columns=len(x)
        x,data=compact_image(x,data)
        if not np.isfinite(data).any():return
        image=pg.ImageItem(data,axisOrder='row-major')
        dx=(x[-1]-x[0])/(len(x)-1) if len(x)>1 else .002
        dy=(y[-1]-y[0])/(len(y)-1) if len(y)>1 else 1
        image.setRect(QRectF(x[0]-dx/2,y[0]-dy/2,dx*len(x),dy*len(y)))
        image.setLookupTable(pg.colormap.get('viridis').getLookupTable())
        self.plots[key].addItem(image)
        self.plots[key].setLabel('bottom','时间',units='s')
        self.plots[key].setLabel('left',ylabel,units=yunit)
        self.plots[key].setTitle('显示按时间分箱取最大值；分析数据保留原分辨率' if original_columns>len(x) else '')
        finite=data[np.isfinite(data)];low=float(np.min(finite));high=float(np.max(finite))
        if high<=low:high=low+1
        bar=self.colorbars.get(key)
        if bar is None:
            bar=pg.ColorBarItem(values=(low,high),colorMap=pg.colormap.get('viridis'),interactive=False)
            bar.setImageItem(image,insert_in=self.plots[key].getPlotItem());self.colorbars[key]=bar
        else:bar.setImageItem(image)
        bar.axis.setLabel(text=zunit);bar.setLevels((low,high));bar.show()

    def show_result_waveform(self,result):
        plot=self.plots['waveform'];plot.clear()
        if result.detail and all(key in result.detail.arrays for key in ('wave_time','waveform')):
            x,y=np.asarray(result.detail.arrays['wave_time']),np.asarray(result.detail.arrays['waveform'])
            if len(x)>1:x,y=envelope_plot_data(y,1/(x[1]-x[0]),origin=x[0])
            plot.plot(x,y,pen=pg.mkPen('#4cc7d6'))
            plot.setLabel('bottom','原录音时间',units='s');plot.setLabel('left','幅值',units=result.detail.units.get('waveform',''))
            plot.addItem(self.cursor)

    def overlay(self, left, right):
        self.set_result(left)
        if left.detail is None or right.detail is None:
            return
        if left.detail.units.get('waveform')==right.detail.units.get('waveform'):
            plot=self.plots['waveform'];plot.clear()
            for result,color in [(left,'#4cc7d6'),(right,'#ffbb66')]:
                arrays=result.detail.arrays
                if 'wave_time' in arrays and 'waveform' in arrays:
                    x,y=np.asarray(arrays['wave_time']),np.asarray(arrays['waveform'])
                    if len(x)>1:x,y=envelope_plot_data(y,1/(x[1]-x[0]),origin=x[0])
                    plot.plot(x,y,pen=pg.mkPen(color))
            plot.setLabel('bottom','原录音时间',units='s');plot.setLabel('left','幅值',units=left.detail.units.get('waveform',''));plot.addItem(self.cursor)
        for kind, xkey, ykey in [('spectrum','frequency','spectrum_db'),('psd','psd_frequency','psd'),
            ('spl','spl_time','laf'),('spl','spl_time','las'),('loudness','loudness_time','loudness')]:
            if left.detail.units.get(ykey)!=right.detail.units.get(ykey):continue
            arrays = right.detail.arrays
            if xkey in arrays and ykey in arrays:
                x, y = np.asarray(arrays[xkey]), np.asarray(arrays[ykey])
                x,y=_finite_line(x,y)
                self.plots[kind].plot(x,y,pen=pg.mkPen('#ffbb66',width=2,style=Qt.PenStyle.DashLine if ykey=='las' else Qt.PenStyle.SolidLine))
        if left.detail.units.get('octave_levels') == right.detail.units.get('octave_levels'):
            arrays = right.detail.arrays
            if 'octave_centers' in arrays and 'octave_levels' in arrays:
                x, y = _finite_line(arrays['octave_centers'], arrays['octave_levels'])
                valid = x > 0
                if valid.any():
                    plot = self.plots['octave']
                    self._octave_legend.clear()
                    for item in plot.items():
                        if isinstance(item, pg.BarGraphItem):
                            self._octave_legend.addItem(item, 'A 基准')
                    plot.plot(x[valid], y[valid], pen=pg.mkPen('#ffbb66', width=2),
                              symbol='o', symbolSize=5, symbolBrush='#ffbb66',
                              symbolPen='#ffbb66', name='B 比较')
                    self._octave_legend.show()
