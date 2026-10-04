import numpy as np
from autoacoustics.model import CancellationToken,CancelledError

def test_event_clock_is_selected_from_full_context_and_no_spl_smoothing():
    from autoacoustics.analysis.loudness import compute_loudness
    fs=48000
    x=np.zeros(19200);x[2400:2880]=np.sin(2*np.pi*1000*np.arange(480)/fs)*.1
    full=compute_loudness(x,fs)
    event=compute_loudness(x,fs,start=2400,end=14400)
    mask=(full['time']>=.05)&(full['time']<.3)
    np.testing.assert_array_equal(event['time'],full['time'][mask])
    np.testing.assert_array_equal(event['total'],full['total'][mask])
    np.testing.assert_array_equal(event['specific'],full['specific'][:,mask])

def test_lower_rate_is_estimate_and_clock_has_no_end_point_drift():
    from autoacoustics.analysis.loudness import compute_loudness
    result=compute_loudness(np.zeros(12000),24000)
    assert result['status']=='estimated'
    assert result['metadata']['resampling_ratio']==[2,1]
    np.testing.assert_allclose(result['time'],np.arange(250)*.002,atol=1e-15)
    assert result['time'][-1]<.5
