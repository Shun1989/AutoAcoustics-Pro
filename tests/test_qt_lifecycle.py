"""Real native widget destruction occurs before QApplication destruction."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import pyqtgraph as pg
from PyQt6 import sip
from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout

from autoacoustics.qt_lifecycle import dispose_gui
from autoacoustics.ui.plots import PlotPanel


def test_plot_event_filter_tolerates_python_state_released_during_gc():
    panel = PlotPanel()
    window = QWidget()
    panel._observed_window = window
    # A native event can arrive during Python cyclic teardown after its state
    # dictionary has been cleared, as observed in qt_lifecycle_targeted.log.
    del panel._observed_window
    assert not panel.eventFilter(window, QEvent(QEvent.Type.Close))
    dispose_gui(QApplication.instance())


def test_disposal_closes_plot_scene_before_destroying_parent():
    application = QApplication.instance()
    for _ in range(10):
        window = QWidget()
        layout = QVBoxLayout(window)
        plot = pg.PlotWidget()
        layout.addWidget(plot)
        plot.plot([1., 2., 3.])
        scene = plot.scene()
        window.show()
        application.processEvents()
        dispose_gui(application)
        assert sip.isdeleted(plot)
        assert sip.isdeleted(window)
        assert sip.isdeleted(scene)
        assert QApplication.instance() is application


def test_plot_cleanup_process_exits_without_native_memory_fault():
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent('''
        import ctypes, faulthandler, gc, sys
        if sys.platform == 'win32':
            ctypes.windll.kernel32.SetErrorMode(3)
        faulthandler.enable()
        from PyQt6.QtWidgets import QApplication
        from autoacoustics.ui.plots import PlotPanel
        from autoacoustics.qt_lifecycle import dispose_gui
        app = QApplication([])
        app.setQuitOnLastWindowClosed(False)
        for _ in range(10):
            panel = PlotPanel()
            panel.show()
            app.processEvents()
            dispose_gui(app)
            del panel
            gc.collect()
        # Import allocations triggered the original post-cleanup native fault.
        from autoacoustics.ui.soundcard_panel import SoundCardPanel
        print('native graph cleanup completed', flush=True)
    ''')
    environment = dict(os.environ, QT_QPA_PLATFORM='offscreen', PYTHONPATH=str(root/'src'))
    completed = subprocess.run([sys.executable, '-X', 'utf8', '-c', script],
                               cwd=root, env=environment, capture_output=True,
                               text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'native graph cleanup completed' in completed.stdout
