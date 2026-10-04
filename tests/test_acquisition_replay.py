import hashlib

import h5py
import numpy as np

from autoacoustics.acquisition.device import AcquisitionConfig, AcquisitionStatus, ChannelConfig
from autoacoustics.acquisition.engine import RecordingEngine
from autoacoustics.acquisition.replay import ReplayBackend
from autoacoustics.acquisition.storage import open_acquisition_session


def ten_minute_pattern(first, count, channels):
    values = (np.arange(first, first + count, dtype=np.int64) % 4096) / 4096
    return np.vstack([values, 1 - values])


def test_ten_minute_equivalent_two_channel_recording_is_bounded_and_exact(tmp_path):
    frames = 51200 * 600
    config = AcquisitionConfig(channels=(ChannelConfig("Replay/0", "PCB电压", "V"),
                                          ChannelConfig("Replay/1", "GRAS电压", "V")),
                               requested_sample_rate=48000, target_frames=frames,
                               block_frames=51200, queue_blocks=3, preview_frames=4096)
    engine = RecordingEngine(ReplayBackend(actual_sample_rate=51200, generator=ten_minute_pattern),
                             config, tmp_path / "ten-minute")
    engine.configure()
    engine.start()
    result = engine.wait(timeout=60)
    assert result.status == AcquisitionStatus.COMPLETE and result.is_simulated
    assert result.frames_expected == result.frames_read == result.frames_written == frames
    assert engine.maximum_queue_depth <= 3 and engine.preview().samples.shape[1] <= 4096
    with h5py.File(result.data_path, "r") as data:
        source = data["samples"]
        assert source.shape == (2, frames) and source.attrs["sample_rate"] == 51200
        for start in (0, 51200-3, frames // 2, frames - 17):
            count = min(17, frames-start)
            np.testing.assert_array_equal(source[:, start:start+count], ten_minute_pattern(start, count, 2))
    signal = open_acquisition_session(result.manifest_path)
    assert signal.frames == frames and signal.duration == 600 and signal.sample_rate == 51200
    assert signal.source_hash == hashlib.sha256(result.manifest_path.read_bytes()).hexdigest()
    assert signal.metadata["acquisition_data_sha256"] == hashlib.sha256(result.data_path.read_bytes()).hexdigest()
    del signal
