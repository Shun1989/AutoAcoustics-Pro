from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

from autoacoustics.importers.registry import import_signal


ROOT = Path(__file__).resolve().parents[1]


def independent_reader():
    name = "real_head_reference"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests/reference/head_pa_reader.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.read_head_pa_bytes


def test_all_66_real_head_files_match_reference_values_counts_rate_units_and_hash(tmp_path):
    original = ROOT / "压缩.zip"
    assert original.is_file(), "Required real HEAD corpus is missing"
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    manifest = json.loads((ROOT / "docs/validation/corpus_manifest.json").read_text(encoding="utf-8"))
    assert before == manifest["archive"]["sha256"]
    reader = independent_reader()
    members = {entry["path"]: entry for entry in manifest["members"]}
    paired_wavs = {entry["id"]: entry for entry in manifest["members"] if entry["kind"] == "wav"}
    with zipfile.ZipFile(original, "r") as archive:
        paths = sorted(name for name in archive.namelist() if name.lower().endswith(".hdf"))
        assert len(paths) == 66
        paired_ids = set(manifest["paired_ids"])
        paired_count, extra_count = 0, 0
        for name in paths:
            blob = archive.read(name)
            path = tmp_path / Path(name).name
            path.write_bytes(blob)
            actual, reference = import_signal(path), reader(blob)
            np.testing.assert_array_equal(actual.samples, reference.samples)
            assert actual.frames == reference.frames == members[name]["frames"]
            assert actual.sample_rate == reference.fs_hz == 48000
            assert actual.channels[0].unit == reference.unit == "Pa"
            assert actual.source_hash == members[name]["sha256"] == hashlib.sha256(blob).hexdigest()
            assert actual.channels[0].name == "Ch. 8@SQuadriga III"
            assert actual.channels[0].metadata["sensor"]["Manufacturer"] == "PCB"
            assert actual.channels[0].metadata["sensor"]["CalibrationFactor"] == "0.9"
            assert actual.metadata["calibration_applied_by_importer"] is False
            if Path(name).stem in paired_ids:
                wav_entry = paired_wavs[Path(name).stem]
                wav_path = tmp_path / Path(wav_entry["path"]).name
                wav_path.write_bytes(archive.read(wav_entry["path"]))
                wav = import_signal(wav_path)
                assert wav.frames == actual.frames
                assert wav.sample_rate == actual.sample_rate
                assert wav.source_hash == wav_entry["sha256"]
                paired_count += 1
            else:
                extra_count += 1
        assert paired_count == 64 and extra_count == 2
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before
