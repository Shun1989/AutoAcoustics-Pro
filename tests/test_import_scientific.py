from __future__ import annotations

import h5py
import numpy as np
from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter
import pytest
from scipy.io import savemat

from autoacoustics.importers.registry import import_signal
from autoacoustics.model import ImportError, ImportMapping, MappingRequired


def test_classic_mat_vector_and_known_fields(tmp_path):
    path = tmp_path / "known.mat"
    values = np.asarray([0.25, -0.5, 2.0])
    savemat(path, {"signal": values, "fs": 51200.0, "unit": "Pa"})
    result = import_signal(path)
    np.testing.assert_array_equal(result.samples, [values])
    assert result.sample_rate == 51200
    assert result.channels[0].unit == "Pa" and result.channels[0].scaled
    assert result.metadata["dataset"] == "signal"


def test_classic_mat_nested_matrix_explicit_axis_and_mixed_units(tmp_path):
    path = tmp_path / "nested.mat"
    values = np.asarray([[0.25, 2.0], [-0.5, -1], [0.125, 3]])
    savemat(path, {"measurement": {"samples": values, "sample_rate": 1000,
                                  "units": ["V", "Pa"], "channel_names": ["Voltage", "Pressure"],
                                  "channel_axis": 1}})
    result = import_signal(path)
    np.testing.assert_array_equal(result.samples, values.T)
    assert [channel.name for channel in result.channels] == ["Voltage", "Pressure"]
    assert [channel.unit for channel in result.channels] == ["V", "Pa"]
    assert result.metadata["dataset"] == "measurement.samples"


def test_classic_mat_single_frame_dimensions_not_squeezed_away(tmp_path):
    path = tmp_path / "single_frame.mat"
    savemat(path, {"samples": np.asarray([[0.25], [2.0]]), "fs": 1000,
                  "units": ["V", "Pa"], "channel_axis": 0})
    result = import_signal(path)
    np.testing.assert_array_equal(result.samples, [[0.25], [2.0]])


def test_mat_ambiguous_variable_and_matrix_axis_require_mapping(tmp_path):
    path = tmp_path / "ambiguous.mat"
    savemat(path, {"first": np.ones((2, 3)), "second": np.asarray([[1, 2, 3], [4, 5, 6]])})
    with pytest.raises(MappingRequired) as caught:
        import_signal(path)
    assert set(caught.value.options["datasets"]) == {"first", "second"}
    with pytest.raises(MappingRequired):
        import_signal(path, ImportMapping(dataset="second", sample_rate=1000, unit="V"))
    result = import_signal(path, ImportMapping(dataset="second", sample_rate=1000, unit="V", channel_axis=0))
    np.testing.assert_array_equal(result.samples, [[1, 2, 3], [4, 5, 6]])
    assert result.metadata["source_dtype"].startswith("int")


@pytest.mark.parametrize("suffix", ["h5", "hdf5", "hdf"])
def test_generic_hdf5_nested_dataset_attributes_dtype_and_mixed_units(tmp_path, suffix):
    path = tmp_path / f"generic.{suffix}"
    values = np.asarray([[0.25, -0.5, 0.125], [2, -1, 3]], dtype="float32")
    with h5py.File(path, "w") as output:
        dataset = output.create_dataset("measurement/samples", data=values)
        dataset.attrs["sample_rate"] = 51200
        dataset.attrs["units"] = np.asarray(["V", "Pa"], dtype=h5py.string_dtype())
        dataset.attrs["channel_names"] = np.asarray(["Voltage", "Pressure"], dtype=h5py.string_dtype())
        dataset.attrs["channel_axis"] = 0
    result = import_signal(path)
    np.testing.assert_array_equal(result.samples, values)
    assert result.sample_rate == 51200
    assert [channel.unit for channel in result.channels] == ["V", "Pa"]
    assert result.metadata["source_dtype"] == "float32"
    assert result.metadata["dataset"] == "/measurement/samples"


def test_mat_73_matlab_storage_transpose_then_logical_channel_axis(tmp_path):
    path = tmp_path / "v73.mat"
    logical = np.asarray([[0.25, -0.5, 0.125], [2, -1, 3]], dtype="float64")
    with h5py.File(path, "w", userblock_size=512) as output:
        dataset = output.create_dataset("signal", data=logical.T)
        dataset.attrs["MATLAB_class"] = np.bytes_("double")
        dataset.attrs["sample_rate"] = 48000
        dataset.attrs["unit"] = "Pa"
    with path.open("r+b") as stream:
        stream.write(b"MATLAB 7.3 MAT-file, Platform: synthetic generic validation\0")
    result = import_signal(path, ImportMapping(dataset="/signal", channel_axis=0))
    np.testing.assert_array_equal(result.samples, logical)
    assert result.metadata["matlab_storage_transpose"] is True
    assert result.sample_rate == 48000


def test_hdf5_mapping_missing_rate_units_and_axis_then_explicit_mapping(tmp_path):
    path = tmp_path / "unknown.h5"
    values = np.asarray([[1, 4], [2, 5], [3, 6]], dtype="int16")
    with h5py.File(path, "w") as output:
        output.create_dataset("anything", data=values)
    with pytest.raises(MappingRequired):
        import_signal(path)
    result = import_signal(path, ImportMapping(dataset="/anything", channel_axis=1, sample_rate=2000,
                                               channel_units=("V", "Pa")))
    np.testing.assert_array_equal(result.samples, values.T)
    assert result.metadata["source_dtype"] == "int16"
    assert [channel.unit for channel in result.channels] == ["V", "Pa"]


def test_explicit_channel_axis_preserves_one_frame_multichannel_matrix(tmp_path):
    path = tmp_path / "one_frame.h5"
    with h5py.File(path, "w") as output:
        dataset = output.create_dataset("samples", data=[[0.25], [2.0]])
        dataset.attrs["sample_rate"] = 1000
        dataset.attrs["units"] = np.asarray(["V", "Pa"], dtype=h5py.string_dtype())
    result = import_signal(path, ImportMapping(dataset="/samples", channel_axis=0))
    np.testing.assert_array_equal(result.samples, [[0.25], [2.0]])
    assert result.channel_count == 2 and result.frames == 1


def test_ambiguous_scale_application_state_is_never_interpreted_by_truthiness(tmp_path):
    path = tmp_path / "ambiguous_scale.h5"
    with h5py.File(path, "w") as output:
        dataset = output.create_dataset("samples", data=[1, 2, 3])
        dataset.attrs.update({"sample_rate": 1000, "unit": "V", "scaling_convention": "linear",
                              "scale_factor": 0.5, "scale_applied": "False"})
    with pytest.raises(ImportError, match="缩放"):
        import_signal(path)


def test_hdf5_declared_unapplied_scale_is_applied_once_and_provenance_kept(tmp_path):
    path = tmp_path / "scaled.h5"
    with h5py.File(path, "w") as output:
        dataset = output.create_dataset("signal", data=np.asarray([1, 2, 3], dtype="int16"))
        dataset.attrs.update({"sample_rate": 1000, "unit": "V", "scaling_convention": "linear",
                              "scale_factor": 0.5, "add_offset": -1.0, "scale_applied": False})
    result = import_signal(path)
    np.testing.assert_array_equal(result.samples, [[-0.5, 0, 0.5]])
    assert result.metadata["scaling"]["applied_by_importer"] is True
    with h5py.File(path, "a") as output:
        output["signal"][:] = np.asarray([-1, 0, 1], dtype="int16")
        output["signal"].attrs["scale_applied"] = True
    result = import_signal(path)
    np.testing.assert_array_equal(result.samples, [[-1, 0, 1]])
    assert result.metadata["scaling"]["applied_by_importer"] is False


def test_head_text_header_is_detected_before_hdf5_and_invalid_scientific_is_typed(tmp_path):
    path = tmp_path / "HEAD.hdf"
    path.write_bytes(b"; Copyright HEAD acoustics\n; HEAD acoustics datafile format\n")
    with pytest.raises(ImportError, match="HEAD"):
        import_signal(path)
    path = tmp_path / "broken.mat"
    path.write_bytes(b"not a MATLAB file")
    with pytest.raises(ImportError):
        import_signal(path)


def write_tdms(path, *, mismatch=False, already_scaled=False):
    scale = {"NI_Number_Of_Scales": 1, "NI_Scale[0]_Scale_Type": "Linear",
             "NI_Scale[0]_Linear_Slope": 0.5, "NI_Scale[0]_Linear_Y_Intercept": -1.0,
             "NI_Scaling_Status": "scaled" if already_scaled else "unscaled"}
    with TdmsWriter(path) as output:
        output.write_segment([RootObject(), GroupObject("Motor"),
                              ChannelObject("Motor", "Voltage", np.asarray([1, 2, 3], dtype="int16"),
                                            {"wf_increment": 1 / 51200, "unit_string": "V", **scale}),
                              ChannelObject("Motor", "Pressure", np.asarray([2, -1, 3]),
                                            {"wf_increment": 1 / (48000 if mismatch else 51200), "unit_string": "Pa"})])


def test_tdms_per_channel_units_native_scaling_and_order(tmp_path):
    path = tmp_path / "native.tdms"
    write_tdms(path)
    result = import_signal(path)
    np.testing.assert_array_equal(result.samples, [[-0.5, 0, 0.5], [2, -1, 3]])
    assert result.sample_rate == 51200
    assert [channel.unit for channel in result.channels] == ["V", "Pa"]
    assert result.channels[0].metadata["scaling_provider"] == "nptdms"
    result = import_signal(path, ImportMapping(channels=("/'Motor'/'Pressure'", "/'Motor'/'Voltage'")))
    np.testing.assert_array_equal(result.samples, [[2, -1, 3], [-0.5, 0, 0.5]])
    assert result.metadata["import_mapping"]["channels"][0] == "/'Motor'/'Pressure'"


def test_tdms_already_scaled_does_not_scale_again(tmp_path):
    path = tmp_path / "already.tdms"
    write_tdms(path, already_scaled=True)
    result = import_signal(path)
    np.testing.assert_array_equal(result.samples[0], [1, 2, 3])


def test_tdms_differing_sample_rates_require_selected_channels_and_truncation_rejected(tmp_path):
    path = tmp_path / "mismatched.tdms"
    write_tdms(path, mismatch=True)
    with pytest.raises(MappingRequired):
        import_signal(path)
    selected = import_signal(path, ImportMapping(channels=("/'Motor'/'Voltage'",)))
    assert selected.sample_rate == 51200 and selected.channel_count == 1
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ImportError):
        import_signal(path)
