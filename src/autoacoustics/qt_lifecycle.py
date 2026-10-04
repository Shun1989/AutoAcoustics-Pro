"""Release graph scenes on the GUI thread while QApplication is still alive."""
import time

import pyqtgraph as pg
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, Qt
from PyQt6.QtWidgets import QMainWindow, QMenu


def dispose_gui(application, windows=None, timeout=20.):
    # MainWindow.closeEvent stops capture and waits for writers/analysis. Never
    # delete a parent whose close was deferred while those workers are active.
    deadline = time.monotonic() + timeout
    while True:
        candidates = list(windows) if windows is not None else application.topLevelWidgets()
        roots = [widget for widget in candidates if not sip.isdeleted(widget)
                 and widget.parentWidget() is None and not isinstance(widget, QMenu)
                 and widget.windowType() not in (Qt.WindowType.Popup, Qt.WindowType.ToolTip)
                 and (windows is not None or isinstance(widget, (QMainWindow, pg.PlotWidget))
                      or type(widget).__module__.startswith('autoacoustics.')
                      or widget.findChildren(pg.PlotWidget))]
        # PlotItem's QWidgetAction controls can be parentless native windows.
        # Their owner must release them; deleting all topLevelWidgets double
        # destroys them when the graph's Python wrappers are later collected.
        pending = [widget for widget in roots
                   if not sip.isdeleted(widget) and widget.close() is False]
        application.processEvents()
        if not pending:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('等待采集和分析线程停止超时，未强制销毁正在使用的界面。')
        time.sleep(.01)
    plots = {}
    for root in roots:
        if sip.isdeleted(root):
            continue
        children = ([root] if isinstance(root, pg.PlotWidget) else [])
        children += root.findChildren(pg.PlotWidget)
        for plot in children:
            plots[id(plot)] = plot
    for plot in plots.values():
        if not sip.isdeleted(plot):
            # PlotWidget.close removes ViewBox/axes and clears its scene; parent
            # QObject deletion alone skips this pyqtgraph-specific operation.
            if plot.plotItem is not None:
                plot.close()
            plot.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    application.processEvents()
    for root in roots:
        if not sip.isdeleted(root):
            root.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    application.processEvents()
