from __future__ import annotations

import numpy as np
import pytest

from autoacoustics.importers.registry import import_signal
from autoacoustics.model import ImportError, ImportMapping, MappingRequired


@pytest.mark.parametrize("suffix,delimiter", [("csv", ","), ("txt", "\t"), ("dat", ";")])
def test_delimiters_inferred_time_and_header_units_preserve_channel_order(tmp_path, suffix, delimiter):
    path = tmp_path / f"实际 表格.{suffix}"
    rows = [["time[s]", "Mic[V]", "Pressure[Pa]"], ["1.5", "0.25", "2"],
            ["1.501", "-0.5", "1"], ["1.502", "0.125", "-3"]]
    path.write_text("\n".join(delimiter.join(row) for row in rows), encoding="utf-8-sig")
    result = import_signal(path)
    np.testing.assert_array_equal(result.samples, [[0.25, -0.5, 0.125], [2, 1, -3]])
    assert result.sample_rate == pytest.approx(1000)
    assert [channel.name for channel in result.channels] == ["Mic", "Pressure"]
    assert [channel.unit for channel in result.channels] == ["V", "Pa"]
    assert result.metadata["time_origin_seconds"] == 1.5
    assert result.metadata["delimiter"] == delimiter


def test_explicit_mapping_column_reorder_rate_units_and_project_ready_snapshot(tmp_path):
    path = tmp_path / "no_metadata.csv"
    path.write_text("a,b\n2,0.25\n1,-0.5\n-3,0.125\n", encoding="utf-8")
    with pytest.raises(MappingRequired) as caught:
        import_signal(path)
    assert list(caught.value.options["columns"]) == ["a", "b"]
    mapping = ImportMapping(sample_rate=51200, data_columns=("b", "a"), channel_units=("V", "Pa"))
    result = import_signal(path, mapping)
    np.testing.assert_array_equal(result.samples, [[0.25, -0.5, 0.125], [2, 1, -3]])
    assert [channel.unit for channel in result.channels] == ["V", "Pa"]
    assert list(result.metadata["import_mapping"]["data_columns"]) == ["b", "a"]
    assert result.sample_rate == 51200


def test_plain_numeric_table_needs_explicit_rate_columns_and_unit_choice(tmp_path):
    path = tmp_path / "plain.dat"
    path.write_text("0.1 2\n0.2 3\n0.3 4\n", encoding="ascii")
    with pytest.raises(MappingRequired):
        import_signal(path)
    result = import_signal(path, ImportMapping(sample_rate=1000, unit="unknown", data_columns=(1,)))
    np.testing.assert_array_equal(result.samples, [[2, 3, 4]])
    assert result.channels[0].unit == "unknown"
    assert "unknown_source_unit" in result.quality


def test_known_comment_fields_sampling_rate_and_units(tmp_path):
    path = tmp_path / "comments.txt"
    path.write_text("# sample_rate: 48000\n# unit: Pa\npressure\n0.25\n-0.5\n", encoding="utf-8")
    result = import_signal(path)
    np.testing.assert_array_equal(result.samples, [[0.25, -0.5]])
    assert result.sample_rate == 48000 and result.channels[0].unit == "Pa"
    assert result.channels[0].scaled


@pytest.mark.parametrize("data", ["time[s],x[Pa]\n0,1\n0.001,2\n0.003,3\n",
                                  "time[s],x[Pa]\n0,1\n0,2\n0.001,3\n",
                                  "time[s],x[Pa]\n0,1\n0.001,\n0.002,3\n",
                                  "time[s],x[Pa]\n0,1\n0.001,nan\n0.002,3\n"])
def test_irregular_duplicate_time_missing_and_nonfinite_values_are_rejected(tmp_path, data):
    path = tmp_path / "bad.csv"
    path.write_text(data, encoding="utf-8")
    with pytest.raises(ImportError):
        import_signal(path)


def test_explicit_rate_conflicting_with_time_axis_rejected(tmp_path):
    path = tmp_path / "conflict.csv"
    path.write_text("time[s],x[Pa]\n0,1\n0.001,2\n0.002,3\n", encoding="utf-8")
    with pytest.raises(ImportError, match="采样率"):
        import_signal(path, ImportMapping(sample_rate=48000, time_column="time[s]", unit="Pa"))


def test_unlabelled_time_needs_time_unit_even_with_explicit_time_column(tmp_path):
    path = tmp_path / "milliseconds.csv"
    path.write_text("time,x[Pa]\n250,1\n251,2\n252,3\n", encoding="utf-8")
    for mapping in [None, ImportMapping(time_column="time", unit="Pa")]:
        with pytest.raises(MappingRequired) as caught:
            import_signal(path, mapping)
        assert caught.value.options["needs_time_unit"] is True
        assert list(caught.value.options["time_unit_choices"]) == ["s", "ms", "us"]
    result = import_signal(path, ImportMapping(time_column="time", time_unit="ms", unit="Pa"))
    assert result.sample_rate == pytest.approx(1000)
    assert result.metadata["time_origin_seconds"] == pytest.approx(0.25)
    assert result.metadata["time_unit"] == "ms"
    assert result.metadata["import_mapping"]["time_unit"] == "ms"


def test_comment_can_declare_time_unit_without_header_unit(tmp_path):
    path = tmp_path / "comment_units.txt"
    path.write_text("# time_unit: us\n# unit: Pa\ntime\tx\n0\t1\n20\t2\n40\t3\n", encoding="utf-8")
    result = import_signal(path)
    assert result.sample_rate == pytest.approx(50000)
