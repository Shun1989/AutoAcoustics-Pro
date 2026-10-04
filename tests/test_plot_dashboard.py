"""Plot dashboard behavior: one live graph survives each view and window."""
import os
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QDialog, QLabel, QPushButton, QWidget, QVBoxLayout

from autoacoustics.model import AnalysisDetail, ChannelInfo, SignalData
from autoacoustics.ui.plots import PlotPanel


@pytest.fixture
def panel():
    app = QApplication.instance() or QApplication([])
    widget = PlotPanel()
    widget.resize(1100, 760)
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    app.processEvents()


def process_events():
    QApplication.instance().processEvents()


def result(offset=0.):
    return SimpleNamespace(detail=AnalysisDetail({
        'wave_time': np.array([2., 2.1, 2.2, 2.3]),
        'waveform': np.array([0., 1., -1., 0.]) + offset,
        'frequency': np.array([0., 10., 20.]),
        'spectrum_db': np.array([1., 2., 3.]) + offset,
        'psd_frequency': np.array([0., 5.]),
        'psd': np.array([2., 4.]) + offset,
        'stft_time': np.array([2., 2.1, 2.2, 2.3]),
        'stft_frequency': np.array([100., 200., 300.]),
        'stft_db': np.arange(12.).reshape(3, 4),
        'octave_centers': np.array([100., 200., 400.]),
        'octave_levels': np.array([40., 50., 60.]) + offset,
        'spl_time': np.array([2., 2.1]),
        'laf': np.array([50., 60.]) + offset,
        'las': np.array([45., 55.]) + offset,
        'loudness_time': np.array([2., 2.1, 2.2, 2.3]),
        'loudness': np.array([1., 2., 3., 4.]) + offset,
        'bark': np.array([1., 2., 3.]),
        'specific_loudness': np.arange(12.).reshape(3, 4),
    }, {'waveform': 'Pa', 'psd': 'Pa²/Hz', 'stft_db': 'dB',
        'spectrum_db': 'dB', 'octave_levels': 'dB', 'laf': 'dB(A)',
        'las': 'dB(A)', 'loudness': 'sone', 'specific_loudness': 'sone/Bark'}))


def test_default_dashboard_shows_time_spectrum_and_spectrogram_together(panel):
    visible = {key for key, plot in panel.plots.items() if plot.isVisible()}
    assert visible == {'waveform', 'spectrum', 'stft'}
    assert panel.tabs.count() == 8
    waveform = panel.plots['waveform'].mapTo(panel, panel.plots['waveform'].rect().topLeft())
    spectrum = panel.plots['spectrum'].mapTo(panel, panel.plots['spectrum'].rect().topLeft())
    stft = panel.plots['stft'].mapTo(panel, panel.plots['stft'].rect().topLeft())
    assert waveform.x() < spectrum.x()
    assert waveform.y() < stft.y()


def test_dashboard_selectors_reuse_existing_graphs_and_data(panel):
    original = dict(panel.plots)
    panel.set_result(result())
    panel.select_dashboard(time_key='loudness', frequency_key='psd')
    process_events()
    assert panel.plots == original
    assert {key for key, plot in panel.plots.items() if plot.isVisible()} == {'loudness', 'psd', 'stft'}
    np.testing.assert_array_equal(panel.plots['psd'].listDataItems()[0].getData()[0], [0., 5.])
    panel.select_dashboard(time_key='spl', frequency_key='bark')
    process_events()
    assert panel.plots['spl'].isVisible() and panel.plots['bark'].isVisible()
    assert len(panel.plots['spl'].listDataItems()) == 2
    assert any(isinstance(item, pg.ImageItem) for item in panel.plots['bark'].items())
    with pytest.raises(ValueError):
        panel.select_dashboard(time_key='spectrum')


def test_legacy_tabs_focus_actual_graph_and_return_to_dashboard(panel):
    panel.tabs.setCurrentIndex(6)
    process_events()
    assert panel.tabs.isVisible()
    assert panel.tabs.widget(6).isAncestorOf(panel.plots['loudness'])
    assert panel.plots['loudness'].isVisible()
    panel.focus_plot('bark')
    process_events()
    assert panel.tabs.currentIndex() == 7
    assert panel.tabs.widget(7).isAncestorOf(panel.plots['bark'])
    panel.show_dashboard()
    process_events()
    assert panel.plots['waveform'].isVisible() and panel.plots['stft'].isVisible()
    # Index zero is already selected in a fresh panel, but requesting it must
    # still enter the single graph view.
    panel.tabs.setCurrentIndex(0)
    panel.show_dashboard()
    panel.tabs.setCurrentIndex(0)
    process_events()
    assert panel.tabs.isVisible() and panel.plots['waveform'].isVisible()


def test_popout_restores_same_graph_selection_cursor_and_zoom(panel):
    signal = SignalData(np.arange(1000.).reshape(1, -1), 1000,
                        (ChannelInfo('microphone', 'Pa'),))
    panel.set_signal(signal, 0)
    panel.set_selection(.2, .6)
    panel.cursor.setValue(.4)
    original = panel.plots['waveform']
    original.setXRange(.1, .7, padding=0)
    original.setYRange(100., 700., padding=0)
    before = np.asarray(original.viewRange())
    panel.findChild(QPushButton, 'expand_time').click()
    process_events()
    dialog = original.window()
    assert isinstance(dialog, QDialog) and dialog.isVisible()
    assert not dialog.isModal()
    assert dialog.isAncestorOf(original)
    dialog.close()
    process_events()
    assert panel.plots['waveform'] is original and original.isVisible()
    assert original.window() is panel
    assert tuple(panel.region.getRegion()) == (.2, .6)
    assert panel.cursor.value() == .4
    np.testing.assert_allclose(original.viewRange(), before)


@pytest.mark.parametrize('key', ['waveform', 'spectrum', 'psd', 'stft', 'octave', 'spl', 'loudness', 'bark'])
def test_every_graph_can_popout_and_restore_to_its_single_page(panel, key):
    graph = panel.plots[key]
    panel.focus_plot(key)
    dialog = panel.pop_out(key)
    process_events()
    assert dialog.isAncestorOf(graph) and graph.isVisible()
    dialog.close()
    process_events()
    assert panel.plots[key] is graph
    assert panel.tabs.currentWidget().isAncestorOf(graph) and graph.isVisible()


def test_analysis_panel_moves_to_popout_and_back(panel):
    analysis = QWidget()
    layout = QVBoxLayout(analysis)
    value = QLabel('LAeq: 62.4 dB(A)')
    layout.addWidget(value)
    panel.set_analysis_panel(analysis)
    process_events()
    assert analysis.isVisible()
    panel.findChild(QPushButton, 'expand_analysis').click()
    process_events()
    dialog = analysis.window()
    assert isinstance(dialog, QDialog) and dialog.isAncestorOf(analysis)
    dialog.close()
    process_events()
    assert analysis.window() is panel and analysis.isVisible()
    assert value.text() == 'LAeq: 62.4 dB(A)'


@pytest.mark.parametrize('show_parent', [True, False])
def test_closing_parent_window_restores_popouts_before_hiding(show_parent):
    app = QApplication.instance() or QApplication([])
    window = QWidget()
    layout = QVBoxLayout(window)
    panel = PlotPanel()
    layout.addWidget(panel)
    if show_parent:
        window.show()
    app.processEvents()
    graph = panel.plots['waveform']
    dialog = panel.pop_out('waveform')
    app.processEvents()
    window.close()
    app.processEvents()
    assert not dialog.isVisible()
    assert panel.isAncestorOf(graph) and graph.window() is window
    window.show()
    app.processEvents()
    assert panel.plots['waveform'] is graph and graph.isVisible()
    window.close()
    app.processEvents()


def test_signal_result_and_overlay_still_render_real_axes_and_units(panel):
    signal = SignalData(np.arange(1000.).reshape(1, -1), 1000,
                        (ChannelInfo('microphone', 'Pa'),),
                        metadata={'time_origin_seconds': 2.})
    panel.set_signal(signal, 0)
    x, y = panel.plots['waveform'].listDataItems()[0].getData()
    assert x[0] == 2. and x[-1] == 2.999 and y[-1] == 999.
    emitted = []
    panel.selectionChanged.connect(lambda a, b: emitted.append((a, b)))
    panel.set_selection(2.1, 2.6)
    assert emitted[-1] == (2.1, 2.6)
    panel.set_result(result())
    image = next(item for item in panel.plots['stft'].items() if isinstance(item, pg.ImageItem))
    assert image.image.shape == (3, 4)
    assert panel.colorbars['stft'].isVisible()
    assert panel.colorbars['stft'].axis.labelText == 'dB'
    panel.overlay(result(), result(2.))
    assert len(panel.plots['waveform'].listDataItems()) == 2
    assert len(panel.plots['spectrum'].listDataItems()) == 2
    assert len(panel.plots['spl'].listDataItems()) == 4
    np.testing.assert_array_equal(panel.plots['spectrum'].listDataItems()[-1].getData()[1], [3., 4., 5.])


def test_octave_bars_keep_computed_band_levels_and_readable_hz_axis(panel):
    from autoacoustics.analysis.octave import compute_octave
    samples = np.sin(2 * np.pi * 1000 * np.arange(24000) / 48000)
    bands = compute_octave(samples, 48000)
    detail = AnalysisDetail({'octave_centers': bands['exact'], 'octave_levels': bands['levels']},
                            {'octave_levels': 'dB'})
    panel.set_result(SimpleNamespace(detail=detail))
    panel.select_dashboard(frequency_key='octave')
    process_events()
    plot = panel.plots['octave']
    bars = [item for item in plot.items() if isinstance(item, pg.BarGraphItem)]
    assert len(bars) == 1
    np.testing.assert_allclose(10 ** np.asarray(bars[0].opts['x']), bands['exact'])
    np.testing.assert_array_equal(bars[0].opts['height'], bands['levels'])
    # True one-third octave bands occupy equal space instead of squeezing the
    # lower frequencies into the left edge of a linear frequency graph.
    np.testing.assert_allclose(np.diff(bars[0].opts['x']), .1)
    assert np.all(np.asarray(bars[0].opts['width']) < .1)
    axis = plot.getAxis('bottom')
    assert axis.logMode and axis.labelUnits == 'Hz'
    assert axis.tickStrings([2., 3., 4.], 1., 1.) == ['100', '1000', '10000']
    assert plot.getAxis('left').labelUnits == 'dB'
    assert plot.isVisible()


def test_octave_comparison_keeps_separate_centers_levels_and_legend(panel):
    left = SimpleNamespace(detail=AnalysisDetail({
        'octave_centers': np.array([100., 200., 400.]),
        'octave_levels': np.array([40., 50., 60.]),
    }, {'octave_levels': 'dB'}))
    right = SimpleNamespace(detail=AnalysisDetail({
        'octave_centers': np.array([125., 250., 500.]),
        'octave_levels': np.array([31., 43., 52.]),
    }, {'octave_levels': 'dB'}))
    panel.overlay(left, right)
    plot = panel.plots['octave']
    bars = [item for item in plot.items() if isinstance(item, pg.BarGraphItem)]
    assert len(bars) == 1
    np.testing.assert_allclose(10 ** np.asarray(bars[0].opts['x']), [100., 200., 400.])
    np.testing.assert_array_equal(bars[0].opts['height'], [40., 50., 60.])
    comparison = next(item for item in plot.listDataItems() if isinstance(item, pg.PlotDataItem))
    np.testing.assert_array_equal(comparison.xData, [125., 250., 500.])
    np.testing.assert_array_equal(comparison.yData, [31., 43., 52.])
    assert [label.text for sample, label in plot.getPlotItem().legend.items] == ['A 基准', 'B 比较']
    panel.set_result(left)
    assert not any(isinstance(item, pg.PlotDataItem) for item in plot.listDataItems())
    assert [label.text for sample, label in plot.getPlotItem().legend.items] == ['声级']


@pytest.mark.parametrize('spacing', [None, .1, .5])
def test_octave_secondary_ticks_do_not_generate_dense_frequency_text(panel, spacing):
    axis = panel.plots['octave'].getAxis('bottom')
    minor_positions = np.log10([20., 30., 40., 50., 60., 70., 80., 90.])
    assert axis.tickStrings(minor_positions, 1., spacing) == [''] * 8
    assert axis.tickStrings([2., 3., 4.], 1., 1.) == ['100', '1000', '10000']


@pytest.mark.parametrize('width', [410, 620])
def test_octave_axis_keeps_sparse_primary_labels_at_both_dashboard_widths(panel, width):
    axis = panel.plots['octave'].getAxis('bottom')
    ticks = axis.tickValues(np.log10(20.), np.log10(20000.), width)
    labels_by_position = {float(position): label for spacing, values in ticks
                          for position, label in zip(values, axis.tickStrings(values, 1., spacing))
                          if label}
    assert labels_by_position == {2.: '100', 3.: '1000', 4.: '10000'}
