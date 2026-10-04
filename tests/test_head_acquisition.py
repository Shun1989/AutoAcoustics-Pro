import zipfile
from pathlib import Path

import numpy as np
import pytest

from autoacoustics.acquisition.device import AcquisitionError, AcquisitionStatus
from autoacoustics.acquisition.head_recorder import HeadManagedRecorder, copy_completed_recording
from autoacoustics.acquisition import head_recorder
from autoacoustics.acquisition.replay import SimulatedManagedRecorder
from autoacoustics.acquisition.storage import open_acquisition_session
from autoacoustics.importers.registry import import_signal


def real_head(tmp_path):
    with zipfile.ZipFile(Path(__file__).resolve().parents[1] / "压缩.zip") as archive:
        blob = archive.read("HDF/01-01CW.hdf")
    source = tmp_path / "设备录制.hdf"
    source.write_bytes(blob)
    return source


def test_head_file_handoff_is_explicit_readonly_stable_and_uses_product_import(tmp_path):
    source = real_head(tmp_path)
    before = source.read_bytes()
    session = copy_completed_recording(source, tmp_path / "handoff", device_stopped=True,
                                       device_info={"model": "SQuadriga III", "firmware": "unknown"},
                                       operator="tester", stability_interval=0.001)
    assert session.status == AcquisitionStatus.COMPLETE and not session.is_simulated
    assert session.backend == "HEAD_FILE_HANDOFF"
    actual = open_acquisition_session(session.manifest_path)
    np.testing.assert_array_equal(actual.samples, import_signal(source).samples)
    assert actual.channels[0].unit == "Pa" and source.read_bytes() == before
    assert actual.metadata["acquisition_session"]["device"]["model"] == "SQuadriga III"
    assert actual.metadata["acquisition_session"]["controlled_recording"] is False


def test_head_file_requires_explicit_stop_and_rejects_incomplete_payload(tmp_path):
    source = real_head(tmp_path)
    with pytest.raises(AcquisitionError, match="停止"):
        copy_completed_recording(source, tmp_path / "unstopped", device_stopped=False)
    source.write_bytes(source.read_bytes()[:-1])
    with pytest.raises(AcquisitionError, match="完整"):
        copy_completed_recording(source, tmp_path / "truncated", device_stopped=True, stability_interval=0.001)


def test_head_official_control_without_confirmed_sdk_reports_precise_unavailable_reason():
    backend = HeadManagedRecorder()
    result = backend.discover()
    assert not result.available and result.devices == ()
    assert "ASX" in result.reason and "SDK" in result.reason
    with pytest.raises(AcquisitionError, match="官方"):
        backend.start()
    assert not backend.supports_realtime_samples


def test_changing_device_file_is_not_published_as_complete(tmp_path, monkeypatch):
    source = real_head(tmp_path)
    original_copy = head_recorder._copy_bytes
    def moving_copy(incoming, destination):
        original_copy(incoming, destination)
        with incoming.open("ab") as output:
            output.write(b"injected concurrent change")
    monkeypatch.setattr(head_recorder, "_copy_bytes", moving_copy)
    destination = tmp_path / "changing"
    with pytest.raises(AcquisitionError, match="变化"):
        copy_completed_recording(source, destination, device_stopped=True, stability_interval=0.001)
    assert not (destination / source.name).exists()
    with pytest.raises(AcquisitionError, match="完整"):
        open_acquisition_session(destination / "session.json")


def test_simulated_managed_recorder_contract_is_labelled_and_has_no_realtime_samples(tmp_path):
    source = real_head(tmp_path)
    recorder = SimulatedManagedRecorder((source,))
    assert recorder.discover().devices[0].is_simulated
    assert not recorder.supports_realtime_samples
    assert recorder.completed_files() == ()
    recorder.configure({"purpose": "software contract test"})
    recorder.start()
    recorder.start()
    recorder.stop()
    recorder.stop()
    assert recorder.completed_files() == (source,)
    session = copy_completed_recording(source, tmp_path / "simulated-managed", device_stopped=True,
                                       is_simulated=True, stability_interval=0.001)
    assert session.is_simulated
    assert open_acquisition_session(session.manifest_path).channels[0].unit == "Pa"


@pytest.mark.parametrize("fault", ["disconnect", "license", "stop"])
def test_simulated_managed_errors_never_claim_complete_files(tmp_path, fault):
    recorder = SimulatedManagedRecorder((real_head(tmp_path),), fault=fault)
    recorder.configure()
    with pytest.raises(AcquisitionError):
        if fault == "stop":
            recorder.start()
            recorder.stop()
        else:
            recorder.start()
    assert recorder.completed_files() == ()
