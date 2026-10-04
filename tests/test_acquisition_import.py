"""Acquisition source envelopes use the same desktop/batch/project importer."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np
import pytest

from autoacoustics.acquisition.head_recorder import copy_completed_recording
from autoacoustics.analysis.pipeline import analyze
from autoacoustics.batch import run_batch
from autoacoustics.importers import import_signal
from autoacoustics.model import (AnalysisSettings, BatchItemSpec, ImportError, ImportMapping,
    MeasurementContext, MeasurementProject, ResultSnapshot, ValidationError)
from autoacoustics.project import load_project, load_result_detail, save_project


def head_handoff(tmp_path):
    with zipfile.ZipFile(Path(__file__).resolve().parents[1] / "压缩.zip") as archive:
        original = archive.read("HDF/01-01CW.hdf")
    source = tmp_path / "设备已停止.hdf"
    source.write_bytes(original)
    session = copy_completed_recording(source, tmp_path / "交接会话", device_stopped=True,
        measurement_context={"specimen": "software-handoff-validation", "motor_rotation": "CW",
                             "rotation_view": "unknown"},
        events=({"kind": "startup", "sample_index": 4800, "time_seconds": .1,
                 "direction": "CW", "note": "operator annotation"},),
        device_info={"model": "SQuadriga III", "firmware": "unknown"},
        stability_interval=.001)
    return source, session


def test_real_head_completed_manifest_is_importable_with_manifest_identity_and_provenance(tmp_path):
    source, session = head_handoff(tmp_path)
    signal = import_signal(session.manifest_path)
    assert signal.path == session.manifest_path.resolve()
    assert signal.source_hash == hashlib.sha256(session.manifest_path.read_bytes()).hexdigest()
    assert signal.metadata["acquisition_data_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert signal.metadata["acquisition_manifest_sha256"] == signal.source_hash
    assert Path(signal.metadata["acquisition_data_path"]) == session.data_path
    assert signal.metadata["acquisition_session"]["measurement_context"]["motor_rotation"] == "CW"
    assert signal.metadata["acquisition_session"]["events"][0]["sample_index"] == 4800
    assert signal.channels[0].metadata["acquisition_channel"]["unit"] == "Pa"
    np.testing.assert_array_equal(signal.samples, import_signal(source).samples)
    assert signal.metadata["calibration_applied_by_importer"] is False


def test_real_head_manifest_analyze_batch_project_reopen_keep_same_identity(tmp_path):
    _, session = head_handoff(tmp_path)
    signal = import_signal(session.manifest_path)
    settings = AnalysisSettings(channel=0, start=4800, end=9600, compute_loudness=False)
    context = MeasurementContext(**signal.metadata["acquisition_session"]["measurement_context"])
    result = analyze(signal, None, settings, context)
    assert result.summary.metrics["LZeq"].status == "ok"
    assert result.provenance["calibration"]["coefficient"] == 1.
    item = BatchItemSpec(signal.path, settings, context=context, input_hash=signal.source_hash)
    batch = run_batch((item,))
    assert batch.items[0].status == "ok"
    assert batch.items[0].result.result_id == result.result_id
    project_path = tmp_path / "声音项目.json"
    save_project(MeasurementProject("HEAD C2 software", (item,), (ResultSnapshot(result),)), project_path)
    restored = load_project(project_path)
    cached = load_result_detail(restored.results[0], project_path)
    np.testing.assert_array_equal(cached.arrays["waveform"], result.detail.arrays["waveform"])
    reopened = import_signal(restored.items[0].path, restored.items[0].mapping)
    assert reopened.source_hash == result.input_hash
    repeated = analyze(reopened, None, restored.items[0].settings, restored.items[0].context,
                       include_detail=False)
    assert repeated.result_id == result.result_id
    assert repeated.provenance["source_metadata"]["acquisition_session"]["events"][0]["direction"] == "CW"


def recovery_file(tmp_path, completion):
    path = tmp_path / "recovered.h5"
    with h5py.File(path, "w") as output:
        dataset = output.create_dataset("samples", data=np.sin(2*np.pi*1000*np.arange(4800)/48000)[None])
        dataset.attrs.update({"sample_rate": 48000, "channel_axis": 0, "unit": "Pa",
            "acquisition_complete": completion, "acquisition_session_id": "software-recovery-fixture"})
    return path


@pytest.mark.parametrize("completion", [False, 0])
def test_direct_recovered_hdf5_cannot_bypass_incomplete_recording_measurement_gate(tmp_path, completion):
    signal = import_signal(recovery_file(tmp_path, completion))
    assert "incomplete_recording" in signal.quality
    with pytest.raises(ValidationError, match="录制未完成"):
        analyze(signal, None, AnalysisSettings(channel=0, compute_loudness=False))


@pytest.mark.parametrize("completion", [True, 1])
def test_direct_completed_hdf5_remains_analyzable(tmp_path, completion):
    signal = import_signal(recovery_file(tmp_path, completion))
    assert "incomplete_recording" not in signal.quality
    result = analyze(signal, None, AnalysisSettings(channel=0, compute_loudness=False), include_detail=False)
    assert result.summary.metrics["LZeq"].status == "ok"


@pytest.mark.parametrize("completion", ["false", 2])
def test_ambiguous_acquisition_completion_attribute_is_not_guessed(tmp_path, completion):
    with pytest.raises(ImportError, match="完整|状态"):
        import_signal(recovery_file(tmp_path, completion))


def test_acquisition_manifest_refuses_mapping_that_would_relabel_recorded_units(tmp_path):
    _, session = head_handoff(tmp_path)
    with pytest.raises(ImportError, match="映射"):
        import_signal(session.manifest_path, ImportMapping(unit="V"))


@pytest.mark.parametrize("payload", [{"schema_version": 1, "status": "complete"},
    {"schema_version": 1, "name": "ordinary project", "items": []}, [], {"schema_version": 2}])
def test_generic_or_incomplete_json_is_not_accepted_as_acquisition(tmp_path, payload):
    path = tmp_path / "ordinary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ImportError):
        import_signal(path)


def test_manifest_rejects_corrupted_data_and_recursive_json_data_reference(tmp_path):
    _, session = head_handoff(tmp_path)
    original = json.loads(session.manifest_path.read_text(encoding="utf-8"))
    bad = {**original, "data_file": "session.json",
           "data_sha256": hashlib.sha256(session.manifest_path.read_bytes()).hexdigest()}
    session.manifest_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ImportError):
        import_signal(session.manifest_path)
    session.manifest_path.write_text(json.dumps(original), encoding="utf-8")
    with session.data_path.open("ab") as output:
        output.write(b"broken trailing bytes")
    with pytest.raises(ImportError, match="哈希"):
        import_signal(session.manifest_path)
