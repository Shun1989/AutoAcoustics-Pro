from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from autoacoustics.model import ImportError


def import_signal(*args, **kwargs):
    try:
        from autoacoustics.importers.registry import import_signal as implementation
    except ModuleNotFoundError as error:
        pytest.fail(f"Audio import implementation is missing: {error}")
    return implementation(*args, **kwargs)


@pytest.mark.parametrize("subtype", ["PCM_U8", "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"])
def test_wav_bit_depth_amplitude_channel_order_and_chinese_path(tmp_path, subtype):
    path = tmp_path / f"中文 双通道 {subtype}.wav"
    frames = np.asarray([[0.125, -0.5], [-0.25, 0.75], [0.0, -0.125], [0.5, 0.25]])
    sf.write(path, frames, 51200, subtype=subtype)
    signal = import_signal(path)
    np.testing.assert_array_equal(signal.samples, frames.T)
    assert signal.sample_rate == 51200
    assert signal.frames == 4 and signal.channel_count == 2
    assert [channel.name for channel in signal.channels] == ["Channel 1", "Channel 2"]
    assert all(channel.unit == "FS" for channel in signal.channels)
    assert signal.source_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    assert signal.metadata["subtype"] == subtype
    assert not signal.samples.flags.writeable
    with pytest.raises(TypeError):
        signal.metadata["subtype"] = "changed"


def test_pcm_8_unsigned_normalization_and_rail_quality(tmp_path):
    path = tmp_path / "unsigned.wav"
    values = np.asarray([-1.0, 0.0, 1 / 128, 127 / 128])
    sf.write(path, values, 22050, subtype="PCM_U8")
    signal = import_signal(path)
    np.testing.assert_array_equal(signal.samples[0], values)
    assert "possible_clipping" in signal.quality
    assert signal.channels[0].full_scale == 1.0


def test_float_wav_keeps_values_beyond_nominal_digital_range(tmp_path):
    path = tmp_path / "float.wav"
    values = np.asarray([-1.25, 0.25, 1.5])
    sf.write(path, values, 48000, subtype="FLOAT")
    signal = import_signal(path)
    np.testing.assert_array_equal(signal.samples[0], values)
    assert signal.channels[0].full_scale is None
    assert "floating_full_scale_unknown" in signal.quality
    assert "possible_clipping" not in signal.quality


def test_flac_preserves_amplitudes_rate_and_channels(tmp_path):
    path = tmp_path / "lossless.flac"
    values = np.asarray([[0.125, 0.5], [0.25, -0.75], [-0.5, 0.0]])
    sf.write(path, values, 44100, subtype="PCM_24")
    signal = import_signal(path)
    np.testing.assert_array_equal(signal.samples, values.T)
    assert signal.sample_rate == 44100
    assert signal.metadata["lossy"] is False


def test_wav_companded_encoding_is_labelled_as_lossy(tmp_path):
    path = tmp_path / "companded.wav"
    sf.write(path, [0.25, -0.5], 48000, subtype="ALAW")
    result = import_signal(path)
    assert result.metadata["lossy"] is True
    assert "lossy_source" in result.quality


def test_rf64_wav_valid_and_truncated_payload(tmp_path):
    path = tmp_path / "large_container.wav"
    sf.write(path, [0.25, -0.5, 0.125], 48000, subtype="PCM_16", format="RF64")
    result = import_signal(path)
    np.testing.assert_array_equal(result.samples, [[0.25, -0.5, 0.125]])
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ImportError):
        import_signal(path)


@pytest.mark.parametrize("damage", ["truncated", "nonfinite", "not_audio"])
def test_invalid_audio_is_typed_error(tmp_path, damage):
    path = tmp_path / "invalid.wav"
    if damage == "not_audio":
        path.write_bytes(b"not an audio container")
    else:
        sf.write(path, [0.25, np.nan if damage == "nonfinite" else -0.5], 48000, subtype="FLOAT")
        if damage == "truncated":
            path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ImportError):
        import_signal(path)


def test_missing_file_and_zip_receive_actionable_error(tmp_path):
    with pytest.raises(ImportError):
        import_signal(tmp_path / "missing.wav")
    path = tmp_path / "压缩.zip"
    path.write_bytes(b"PK")
    with pytest.raises(ImportError, match="解压"):
        import_signal(path)


def test_input_changed_during_decode_does_not_get_a_misleading_source_hash(tmp_path, monkeypatch):
    from autoacoustics.importers import audio
    path = tmp_path / "changed.wav"
    sf.write(path, [0.25, -0.5], 48000, subtype="PCM_16")
    original = audio.make_signal

    def change_file_after_samples_read(*args, **kwargs):
        result = original(*args, **kwargs)
        sf.write(path, [0.125, -0.25], 48000, subtype="PCM_16")
        return result

    monkeypatch.setattr(audio, "make_signal", change_file_after_samples_read)
    with pytest.raises(ImportError, match="变化"):
        import_signal(path)


def test_committed_actual_format_fixtures_support_independent_package_probes():
    from autoacoustics.model import ImportMapping
    fixture_dir = Path(__file__).resolve().parent / "fixtures/imports"
    manifest_path = fixture_dir / "fixture_manifest.json"
    assert manifest_path.is_file(), "All-format standalone fixture manifest has not been generated"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kinds = {case["kind"] for case in manifest["cases"]}
    assert {"wav", "flac", "mp3", "m4a", "aac", "ogg", "mp4", "avi", "mkv",
            "csv", "txt", "dat", "mat", "mat73", "hdf5", "tdms"} <= kinds
    for case in manifest["cases"]:
        path = fixture_dir / case["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case["sha256"]
        mapping = ImportMapping(**case["mapping"]) if case["mapping"] else None
        result = import_signal(path, mapping)
        assert result.sample_rate == pytest.approx(case["sample_rate"])
        assert result.channel_count == case["channels"]
        assert [channel.unit for channel in result.channels] == case["units"]
        assert abs(result.frames - case["source_frames"]) <= case["frame_tolerance"]
