from __future__ import annotations

import hashlib

import h5py
import numpy as np
import pytest

from autoacoustics.importers.registry import import_signal
from autoacoustics.model import ImportError, ImportMapping


def head_fixture(values=(0.125, -0.5, 1.25), *, kind="FLOAT32", order="Intel", unit="Pa",
                 quantity="sound pressure", offset=8192, extra_fields=None, sensor_xml=None):
    global_fields = {"version": "4", "release": "6", "byte order": order, "kind": "Time data",
                     "start of data": str(offset), "nbr of abscissa": "1", "nbr of channel": "1",
                     "nbr of extra fields": "0", "idx order": "1", "ch order": "1",
                     "data org": "a1b1 a2b2", "channel attributes": "private", "scan mode": "simultaneous"}
    axis = {"abscissa definition": "1", "name str": "Time", "physical quantity": "time",
            "physical unit": "s", "absc sort": "calc", "first value": "0.25",
            "delta value": "2.08333333333333e-005", "nbr of scans": str(len(values)),
            "distribution func": "linear"}
    channel = {"channel definition": "1", "name str": "Mic", "ch sort": "all data",
               "physical quantity": quantity, "physical unit": unit, "calibration": "83.084850197733005",
               "quantisation func": "linear", "composition": "sample", "implementation type": kind}
    for (section, key), value in (extra_fields or {}).items():
        target = {"file": global_fields, "axis": axis, "channel": channel}[section]
        if value is None:
            target.pop(key, None)
        else:
            target[key] = str(value)
    lines = [";", "; Copyright 1999 HEAD acoustics GmbH, Germany", "; HEAD acoustics datafile format",
             ";#code page: 65001"]
    for fields in (global_fields, axis, channel):
        lines += [f"{key}: {value}" for key, value in fields.items()]
    lines += ["; comment", "comment 0 :[Dataset]"]
    if sensor_xml:
        lines += ["[Channel0]", f"_SI_Sensor = string ( {sensor_xml} )"]
    prefix = ("\r\n".join(lines) + "\r\n").encode("utf-8")
    dtype = {"FLOAT32": "f4", "FLOAT64": "f8", "INT16": "i2", "INT32": "i4"}.get(kind, "f4")
    payload = np.asarray(values, dtype=("<" if order == "Intel" else ">") + dtype).tobytes()
    marker = f"data1 {len(payload)} :".encode("ascii")
    assert offset >= len(prefix) + len(marker)
    return prefix + b"\t" * (offset - len(prefix) - len(marker)) + marker + payload


def test_evidenced_mono_float32_layout_pa_passes_through_and_records_evidence(tmp_path):
    kind, order = "FLOAT32", "Intel"
    path = tmp_path / "独立 合成.hdf"
    blob = head_fixture(kind=kind, order=order)
    path.write_bytes(blob)
    result = import_signal(path)
    np.testing.assert_array_equal(result.samples, [[0.125, -0.5, 1.25]])
    assert result.sample_rate == 48000
    assert result.channels[0].unit == "Pa" and result.channels[0].scaled
    assert result.channels[0].metadata["head_calibration"] == 83.084850197733005
    assert result.metadata["format"] == "HEAD_HDF"
    assert result.metadata["calibration_applied_by_importer"] is False
    assert result.metadata["time_origin_seconds"] == 0.25
    assert result.metadata["head"]["data_offset"] == 8192
    assert result.metadata["head"]["implementation_type"] == kind
    assert result.metadata["head"]["byte_order"] == order
    assert result.source_hash == hashlib.sha256(blob).hexdigest()


@pytest.mark.parametrize("kind,order", [("FLOAT64", "Intel"), ("FLOAT32", "Motorola"),
                                       ("FLOAT64", "Motorola")])
def test_unevidenced_float64_and_big_endian_variants_are_rejected(tmp_path, kind, order):
    path = tmp_path / "unverified_variant.hdf"
    path.write_bytes(head_fixture(kind=kind, order=order))
    with pytest.raises(ImportError, match="HEAD"):
        import_signal(path)


def test_signature_routes_renamed_head_and_hdf5_independently_of_extension(tmp_path):
    renamed = tmp_path / "HEAD.bin"
    renamed.write_bytes(head_fixture())
    assert import_signal(renamed).metadata["format"] == "HEAD_HDF"
    renamed_h5 = tmp_path / "generic.wav"
    with h5py.File(renamed_h5, "w") as output:
        dataset = output.create_dataset("samples", data=[0.25, -0.5])
        dataset.attrs.update({"sample_rate": 1000, "unit": "Pa"})
    result = import_signal(renamed_h5)
    np.testing.assert_array_equal(result.samples, [[0.25, -0.5]])
    assert result.metadata["format"] == "HDF5"


@pytest.mark.parametrize("fields", [
    {("file", "version"): 5}, {("file", "release"): 7}, {("file", "nbr of channel"): 2},
    {("file", "nbr of abscissa"): 2}, {("file", "nbr of extra fields"): 1},
    {("file", "data org"): "unknown"}, {("file", "byte order"): "native"},
    {("axis", "delta value"): "nan"}, {("axis", "delta value"): 0},
    {("axis", "nbr of scans"): "9999999999999999999999999"},
    {("channel", "implementation type"): "UINT24"}, {("channel", "quantisation func"): "logarithmic"},
    {("channel", "physical quantity"): "acceleration"},
    {("channel", "unverified payload scale"): "2"}, {("channel", "emphasis"): "True"},
    {("channel", "delta_headroom"): "12"},
])
def test_unknown_layout_version_scaling_or_invalid_fields_are_not_guessed(tmp_path, fields):
    path = tmp_path / "unknown.hdf"
    path.write_bytes(head_fixture(extra_fields=fields))
    with pytest.raises(ImportError, match="HEAD"):
        import_signal(path)


@pytest.mark.parametrize("kind", ["INT16", "INT32"])
def test_integer_pa_without_verified_quantisation_semantics_is_rejected(tmp_path, kind):
    path = tmp_path / "integer.hdf"
    path.write_bytes(head_fixture(values=(1, -2, 3), kind=kind))
    with pytest.raises(ImportError, match="HEAD"):
        import_signal(path)


@pytest.mark.parametrize("damage", ["truncated_header", "truncated_payload", "extra_tail", "duplicate", "nonfinite", "bad_magic"])
def test_damaged_header_or_payload_is_typed_error(tmp_path, damage):
    blob = head_fixture()
    if damage == "truncated_header":
        blob = blob[:512]
    elif damage == "truncated_payload":
        blob = blob[:-1]
    elif damage == "extra_tail":
        blob += b"\0"
    elif damage == "duplicate":
        extra = b"\r\nbyte order: Intel"
        blob = blob.replace(b"byte order: Intel", b"byte order: Intel" + extra, 1).replace(b"\t" * len(extra), b"", 1)
    elif damage == "nonfinite":
        blob = head_fixture(values=(float("nan"),))
    else:
        blob = blob.replace(b"HEAD acoustics", b"FAKE acoustics")
    path = tmp_path / "damaged.hdf"
    path.write_bytes(blob)
    with pytest.raises(ImportError):
        import_signal(path)


def test_sensor_fields_are_metadata_without_implicit_voltage_calibration(tmp_path):
    xml = ('<SensorInfo Name="Mic" Manufacturer="PCB" SerialNumber="123" OutputUnit="mV" '
           'Sensitivity="50" CalibrationFactor="0.9"><Quantity ID="SoundPressure" Unit="Pa" /></SensorInfo>')
    path = tmp_path / "voltage.hdf"
    path.write_bytes(head_fixture(values=(1.25, -0.5), unit="V", sensor_xml=xml))
    result = import_signal(path)
    np.testing.assert_array_equal(result.samples, [[1.25, -0.5]])
    assert result.channels[0].unit == "V"
    assert result.channels[0].metadata["sensor"]["Sensitivity"] == "50"
    assert result.metadata["calibration_applied_by_importer"] is False


def test_millivolt_unit_conversion_is_explicit_without_sensor_gain_or_sensitivity(tmp_path):
    path = tmp_path / "millivolt.hdf"
    path.write_bytes(head_fixture(values=(125, -50), unit="mV"))
    result = import_signal(path)
    np.testing.assert_array_equal(result.samples, [[0.125, -0.05]])
    assert result.channels[0].unit == "V"
    assert result.channels[0].metadata["unit_conversion_factor"] == 0.001
    assert result.channels[0].metadata["source_unit"] == "mV"


def test_missing_or_unknown_unit_remains_unknown_and_pa_cannot_be_relabelled(tmp_path):
    path = tmp_path / "unknown_unit.hdf"
    path.write_bytes(head_fixture(extra_fields={("channel", "physical unit"): None}))
    result = import_signal(path)
    assert result.channels[0].unit == "unknown"
    assert "unknown_source_unit" in result.quality
    path.write_bytes(head_fixture())
    with pytest.raises(ImportError, match="单位"):
        import_signal(path, ImportMapping(unit="V"))


def test_xml_external_entities_or_script_fields_are_never_executed(tmp_path):
    xml = '<!DOCTYPE x [<!ENTITY external SYSTEM "file:///secret">]><SensorInfo Name="&external;" />'
    path = tmp_path / "entity.hdf"
    path.write_bytes(head_fixture(sensor_xml=xml))
    with pytest.raises(ImportError, match="XML"):
        import_signal(path)


@pytest.mark.parametrize("damage", ["missing_offset", "oversized_header", "bad_code_page", "bad_encoding",
                                   "wrong_byte_count", "missing_axis", "missing_sensor_end"])
def test_missing_fields_unknown_encoding_and_oversized_header_fail_early(tmp_path, damage):
    extra = {("file", "start of data"): None} if damage == "missing_offset" else None
    blob = head_fixture(extra_fields=extra)
    if damage == "oversized_header":
        blob = blob.replace(b"start of data: 8192", b"start of data: 99999999999999999999")
    elif damage == "bad_code_page":
        blob = blob.replace(b"code page: 65001", b"code page: 99999")
    elif damage == "bad_encoding":
        blob = blob.replace(b"name str: Mic", b"name str: M\xffc")
    elif damage == "wrong_byte_count":
        blob = blob.replace(b"data1 12 :", b"data1 11 :")
    elif damage == "missing_axis":
        blob = blob.replace(b"abscissa definition: 1", b"abscissa definition: 2")
    elif damage == "missing_sensor_end":
        blob = head_fixture(sensor_xml='<SensorInfo Name="PCB"')
    path = tmp_path / "unsafe.hdf"
    path.write_bytes(blob)
    with pytest.raises(ImportError, match="HEAD"):
        import_signal(path)


def test_source_rate_conflict_and_invalid_channel_mapping_are_rejected(tmp_path):
    path = tmp_path / "axis.hdf"
    path.write_bytes(head_fixture())
    with pytest.raises(ImportError, match="采样率"):
        import_signal(path, ImportMapping(sample_rate=44100))
    with pytest.raises(ImportError, match="通道"):
        import_signal(path, ImportMapping(channels=("Another microphone",)))


def test_acceleration_and_unknown_units_do_not_become_sound_pressure(tmp_path):
    path = tmp_path / "acceleration.hdf"
    path.write_bytes(head_fixture(unit="m/s²", quantity="acceleration"))
    result = import_signal(path)
    assert result.channels[0].unit == "m/s²" and result.channels[0].physical_quantity == "acceleration"
    path.write_bytes(head_fixture(unit="unverified"))
    result = import_signal(path)
    assert result.channels[0].unit == result.channels[0].physical_quantity == "unknown"
    np.testing.assert_array_equal(result.samples, [[0.125, -0.5, 1.25]])


def test_source_metadata_and_samples_are_immutable(tmp_path):
    path = tmp_path / "immutable.hdf"
    path.write_bytes(head_fixture())
    result = import_signal(path)
    with pytest.raises(ValueError):
        result.samples[0, 0] = 99
    with pytest.raises(TypeError):
        result.metadata["head"]["fields"]["file"]["version"] = "5"


def test_signature_in_hdf5_user_block_does_not_override_hdf5_magic(tmp_path):
    path = tmp_path / "userblock.hdf"
    with h5py.File(path, "w", userblock_size=512) as output:
        dataset = output.create_dataset("samples", data=[0.25, -0.5])
        dataset.attrs.update({"sample_rate": 1000, "unit": "Pa"})
    with path.open("r+b") as output:
        output.write(b";\n; Copyright 1999 HEAD acoustics GmbH, Germany\n; HEAD acoustics datafile format\n")
    result = import_signal(path)
    assert result.metadata["format"] == "HDF5"


def test_header_comment_cannot_supply_structural_fields(tmp_path):
    blob = head_fixture(extra_fields={("channel", "implementation type"): None})
    injected = b"comment 0 :[Dataset]\r\nimplementation type: FLOAT32"
    extra = len(injected) - len(b"comment 0 :[Dataset]")
    blob = blob.replace(b"comment 0 :[Dataset]", injected).replace(b"\t" * extra, b"", 1)
    path = tmp_path / "comment_injection.hdf"
    path.write_bytes(blob)
    with pytest.raises(ImportError, match="implementation type"):
        import_signal(path)
