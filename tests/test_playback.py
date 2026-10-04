import importlib
import numpy as np
import pytest

def test_playback_uses_one_explicit_gain_without_normalization():
    module = importlib.import_module('autoacoustics.ui.playback')
    signal = np.asarray([.01, -.02, .03])
    before = signal.copy()
    audio, clipped = module.prepare_playback(signal, gain=2)
    np.testing.assert_allclose(audio, [.02, -.04, .06])
    np.testing.assert_array_equal(signal, before)
    assert clipped == 0
    a, _ = module.prepare_playback(signal, gain=3)
    b, _ = module.prepare_playback(signal * 2, gain=3)
    np.testing.assert_allclose(b, a * 2)

def test_playback_flags_clipping_and_invalid_input():
    module = importlib.import_module('autoacoustics.ui.playback')
    audio, clipped = module.prepare_playback([.75, -.75], gain=2)
    assert clipped == 1
    np.testing.assert_allclose(audio, [1, -1])
    with pytest.raises(ValueError):
        module.prepare_playback([np.nan], gain=1)

@pytest.mark.parametrize('rate',[48000,51200])
def test_real_audio_sink_segment_pause_resume_and_original_cursor(rate):
    import os,time
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtMultimedia import QAudio,QMediaDevices
    from autoacoustics.ui.playback import SegmentPlayer
    app=QApplication.instance() or QApplication([])
    player=SegmentPlayer()
    messages=[];positions=[]
    player.message.connect(messages.append);player.positionChanged.connect(positions.append)
    samples=.015*np.sin(2*np.pi*440*np.arange(rate*2)/rate)
    player.set_segment(samples,rate,start=round(.2*rate),end=round(1.2*rate),gain=1)
    assert len(player.samples)==rate and player.origin==.2
    try:
        player.play()
        if QMediaDevices.defaultAudioOutput().isNull():
            assert player.sink is None and player.state=='stopped'
            assert any('没有可用播放设备' in message for message in messages)
            pytest.skip('此运行环境没有物理音频输出；已验证明确提示。')
        if player.sink is None:
            pytest.fail('真实设备未能启动：'+'；'.join(messages))
        assert player.buffer.size()==rate*4
        deadline=time.monotonic()+2
        # Backend progress and the 40 ms Qt cursor timer advance independently.
        while time.monotonic()<deadline and (
            player.sink.processedUSecs()<100000 or
            not positions or not .2<max(positions)<=1.2
        ):
            app.processEvents();time.sleep(.01)
        assert player.sink.error()==QAudio.Error.NoError,'；'.join(messages)
        assert player.sink.processedUSecs()>=100000
        assert positions and .2<max(positions)<=1.2
        player.pause();app.processEvents()
        assert player.state=='paused' and player.sink.state()==QAudio.State.SuspendedState
        before=player.sink.processedUSecs()
        for _ in range(8):app.processEvents();time.sleep(.01)
        assert player.sink.processedUSecs()-before<30000
        player.play();app.processEvents()
        assert player.state=='playing'
        player.stop();app.processEvents()
        assert player.sink is None and player.buffer is None and player.state=='stopped'
    finally:
        player.stop();app.processEvents()
