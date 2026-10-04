"""Product WAV decode against an independent PCM reference, using read-only ZIP."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import wave
import zipfile

import numpy as np

import pytest


def import_signal(*args, **kwargs):
    try:
        from autoacoustics.importers.registry import import_signal as implementation
    except ModuleNotFoundError as error:
        pytest.fail(f"Audio import implementation is missing: {error}")
    return implementation(*args, **kwargs)


def test_all_64_real_wav_imports_preserve_samples_frames_rate_and_hash(tmp_path):
    archive_path = Path(__file__).resolve().parents[1] / "压缩.zip"
    assert archive_path.is_file(), "Required real corpus is missing"
    before = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = sorted(info for info in archive.namelist() if info.lower().endswith(".wav"))
        assert len(members) == 64
        for name in members:
            data = archive.read(name)
            path = tmp_path / Path(name).name
            path.write_bytes(data)
            signal = import_signal(path)
            with wave.open(io.BytesIO(data), "rb") as reference:
                frames = reference.getnframes()
                values = np.frombuffer(reference.readframes(frames), dtype="<i2").astype(float) / 32768
                assert signal.sample_rate == reference.getframerate() == 48000
                assert signal.channel_count == reference.getnchannels() == 1
                assert signal.frames == frames
            np.testing.assert_array_equal(signal.samples[0], values)
            assert signal.source_hash == hashlib.sha256(data).hexdigest()
            assert signal.channels[0].unit == "FS"
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == before
