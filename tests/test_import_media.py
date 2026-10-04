from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from autoacoustics.importers.registry import import_signal
from autoacoustics.model import ImportError, ImportMapping, MappingRequired


def ffmpeg(*args):
    executable = shutil.which("ffmpeg")
    assert executable, "Required fixture encoder ffmpeg is missing"
    command = [executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *map(str, args)]
    subprocess.run(command, check=True, capture_output=True, timeout=30)


@pytest.fixture
def stereo_source(tmp_path):
    path = tmp_path / "源 双通道.wav"
    times = np.arange(22050) / 44100
    values = np.column_stack((0.25 * np.sin(2 * np.pi * 440 * times),
                              0.125 * np.sin(2 * np.pi * 997 * times)))
    sf.write(path, values, 44100, subtype="PCM_16")
    return path, sf.read(path, always_2d=True)[0]


@pytest.mark.parametrize("suffix,codec,has_video,lossy", [
    ("mp3", "libmp3lame", False, True), ("m4a", "aac", False, True),
    ("aac", "aac", False, True), ("ogg", "libvorbis", False, True),
    ("mp4", "aac", True, True), ("avi", "pcm_s16le", True, False),
    ("mkv", "pcm_s16le", True, False),
])
def test_each_real_encoded_media_format_preserves_channels_rate_and_provenance(
    tmp_path, stereo_source, suffix, codec, has_video, lossy
):
    source, original = stereo_source
    path = tmp_path / f"实际 样例.{suffix}"
    args = []
    if has_video:
        args += ["-f", "lavfi", "-i", "color=c=black:s=32x32:r=10:d=0.5"]
    args += ["-i", source, "-c:a", codec]
    if has_video:
        args += ["-c:v", "mpeg4", "-shortest"]
    ffmpeg(*args, path)
    result = import_signal(path)
    assert result.sample_rate == 44100
    assert result.channel_count == 2
    assert result.metadata["lossy"] is lossy
    assert result.metadata["codec"]
    assert result.metadata["audio_stream_index"] == (1 if has_video else 0)
    assert "encoder_padding" in result.metadata
    assert result.metadata["decoded_time_basis"] == "sample_index_from_decoded_frame_zero"
    if lossy:
        assert "lossy_source" in result.quality
        assert abs(result.frames - original.shape[0]) < 3072
        assert np.std(result.samples[0]) > np.std(result.samples[1]) * 1.4
    else:
        np.testing.assert_array_equal(result.samples, original.T)


def test_multi_audio_stream_requires_absolute_stream_index_and_preserves_mapping(tmp_path, stereo_source):
    source, original = stereo_source
    second = tmp_path / "second.wav"
    sf.write(second, original * 0.5, 44100, subtype="PCM_16")
    path = tmp_path / "two_streams.mkv"
    ffmpeg("-i", source, "-i", second, "-map", "0:a", "-map", "1:a", "-c:a", "pcm_s16le", path)
    with pytest.raises(MappingRequired) as caught:
        import_signal(path)
    options = caught.value.options["audio_streams"]
    assert len(options) == 2 and [item["index"] for item in options] == [0, 1]
    result = import_signal(path, ImportMapping(audio_stream_index=1))
    expected = sf.read(second, always_2d=True)[0].T
    np.testing.assert_array_equal(result.samples, expected)
    assert result.metadata["import_mapping"]["audio_stream_index"] == 1
    with pytest.raises(ImportError, match="音轨"):
        import_signal(path, ImportMapping(audio_stream_index=5))


def test_media_with_no_audio_is_error(tmp_path):
    path = tmp_path / "silent_video.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=s=32x32:r=10:d=0.1", "-an", "-c:v", "mpeg4", path)
    with pytest.raises(ImportError, match="没有音轨"):
        import_signal(path)


def test_source_time_origin_is_recorded_without_inserting_silence(tmp_path, stereo_source):
    source, original = stereo_source
    path = tmp_path / "offset.mkv"
    ffmpeg("-itsoffset", "0.25", "-i", source, "-c:a", "pcm_s16le", path)
    result = import_signal(path)
    assert result.metadata["time_origin_seconds"] == pytest.approx(0.25, abs=0.001)
    np.testing.assert_array_equal(result.samples, original.T)


def test_corrupt_media_or_missing_tools_raise_typed_error(tmp_path, stereo_source, monkeypatch):
    path = tmp_path / "damaged.mp3"
    path.write_bytes(b"not a compressed audio file")
    with pytest.raises(ImportError):
        import_signal(path)
    source, _ = stereo_source
    path = tmp_path / "valid.mp3"
    ffmpeg("-i", source, "-c:a", "libmp3lame", path)
    from autoacoustics import resources
    def missing(_):
        raise ImportError("测试环境缺少内置媒体工具")
    monkeypatch.setattr(resources, "media_tool", missing)
    with pytest.raises(ImportError, match="媒体工具"):
        import_signal(path)


def test_frozen_build_does_not_fall_back_to_path_codecs(tmp_path, stereo_source, monkeypatch):
    import sys
    source, _ = stereo_source
    path = tmp_path / "valid.mp3"
    ffmpeg("-i", source, "-c:a", "libmp3lame", path)
    assert shutil.which("ffprobe")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "broken_bundle"), raising=False)
    with pytest.raises(ImportError, match="交付包缺少"):
        import_signal(path)
