import importlib
import numpy as np

def test_waveform_envelope_preserves_short_impulse():
    module = importlib.import_module('autoacoustics.ui.plots')
    signal = np.zeros(100000)
    signal[54321] = 9
    signal[54322] = -7
    time, values = module.envelope_plot_data(signal, 1000, max_points=1000)
    assert values.max() == 9 and values.min() == -7
    assert len(values) <= 1000
    assert np.all(np.diff(time) >= 0)

def test_metric_display_does_not_forge_zero():
    module = importlib.import_module('autoacoustics.ui.formatting')
    from autoacoustics.model import MetricResult
    assert '未校准' in module.metric_text(MetricResult(None, 'dB', 'uncalibrated'))
    assert '失败' in module.metric_text(MetricResult(None, 'sone', 'failed'))
    assert '静音' in module.metric_text(MetricResult(float('-inf'), 'dB', 'silent'))

def test_result_plots_use_independent_axes_and_image_orientation():
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PyQt6.QtWidgets import QApplication
    import pyqtgraph as pg
    from types import SimpleNamespace
    from autoacoustics.model import AnalysisDetail
    from autoacoustics.ui.plots import PlotPanel
    app=QApplication.instance() or QApplication([])
    panel=PlotPanel()
    try:
        details=AnalysisDetail({'frequency':np.array([0,10,20]),'spectrum_db':np.array([1,2,3]),
            'psd_frequency':np.array([0,5]),'psd':np.array([2.,4.]),
            'stft_time':np.array([.1,.2,.3,.4]),'stft_frequency':np.array([100,200,300]),
            'stft_db':np.arange(12).reshape(3,4),
            'spl_time':np.array([.1,.2]),'laf':np.array([50,60]),'las':np.array([45,55]),
            'loudness_time':np.array([.1,.2,.3,.4]),'bark':np.array([1,2,3]),
            'specific_loudness':np.arange(12).reshape(3,4)}, {'psd':'Pa²/Hz'})
        panel.set_result(SimpleNamespace(detail=details))
        x,y=panel.plots['psd'].listDataItems()[0].getData()
        assert np.array_equal(x,[0,5]) and np.array_equal(y,[2,4])
        assert len(panel.plots['spl'].listDataItems())==2
        for key in ('stft','bark'):
            image=next(item for item in panel.plots[key].items() if isinstance(item,pg.ImageItem))
            assert image.image.shape==(3,4)
            assert image.boundingRect().width()==4 and image.boundingRect().height()==3
    finally:
        panel.close();app.processEvents()
def test_display_decimation_preserves_frequency_peak_and_time_heatmap_burst():
    import numpy as np
    from autoacoustics.ui.plots import envelope_xy, compact_image
    x=np.arange(100000.)*.125
    y=np.zeros(100000);y[43217]=99
    dx,dy=envelope_xy(x,y,max_points=2000)
    assert len(dx)<=2000 and len(dx)==len(dy)
    assert dx[np.argmax(dy)]==x[43217]
    matrix=np.zeros((3,10000));matrix[1,999]=17
    original=matrix.copy()
    times,data=compact_image(np.arange(10000)*.002,matrix,max_columns=100)
    assert data.shape==(3,100)
    assert data.max()==17
    assert len(times)==100
    np.testing.assert_array_equal(matrix,original)

