"""Delimited numeric data with explicit columns, units and uniform time."""
from __future__ import annotations

import csv
from pathlib import Path
import re

import numpy as np

from autoacoustics.model import ChannelInfo, ImportError, ImportMapping, MappingRequired
from .common import make_signal, positive_rate, resolve_units


def _read_text(path: Path):
    data = path.read_bytes()
    encodings = ("utf-16",) if data[:2] in {b"\xff\xfe", b"\xfe\xff"} else ("utf-8-sig", "gb18030")
    for encoding in encodings:
        try:
            return data.decode(encoding), encoding
        except UnicodeError:
            continue
    raise ImportError("表格文本编码无法读取。请另存为 UTF-8。")


def _column_index(value, names):
    if isinstance(value, bool):
        raise ImportError("列索引必须为整数或明确列名。")
    if isinstance(value, int) and 0 <= value < len(names):
        return value
    if isinstance(value, str) and names.count(value) == 1:
        return names.index(value)
    raise ImportError(f"所选列不存在或名称不唯一：{value}")


def _header_parts(name):
    match = re.fullmatch(r"\s*(.*?)\s*[\[(]([^\])]+)[\])]\s*", name)
    return (match.group(1).strip(), match.group(2).strip()) if match else (name.strip(), None)


def import_tabular(path: Path, mapping: ImportMapping | None = None):
    text, encoding = _read_text(path)
    declarations, lines = {}, []
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            match = re.fullmatch(r"\s*#\s*(sample_rate|fs|unit|units|time_unit)\s*[:=]\s*(.*?)\s*", line, re.I)
            if match:
                key, value = match.groups()
                if key.lower() in declarations:
                    raise ImportError("表格采样率或单位元数据重复。")
                declarations[key.lower()] = value
            continue
        lines.append(line)
    if not lines:
        raise ImportError("表格没有有效数值行。")
    delimiter = mapping.delimiter if mapping and mapping.delimiter else None
    if delimiter is None:
        sample = "\n".join(lines[:20])
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = "whitespace"
    if delimiter == "whitespace" or delimiter == " ":
        rows = [re.split(r"\s+", line.strip()) for line in lines]
    elif len(delimiter) == 1:
        rows = list(csv.reader(lines, delimiter=delimiter))
    else:
        raise ImportError("分隔符须为单个字符或 whitespace。")
    width = len(rows[0])
    if not width or any(len(row) != width for row in rows):
        raise ImportError("表格各行的列数不一致；请核对分隔符与缺失值。")
    try:
        [float(value) for value in rows[0]]
        has_header = False
    except ValueError:
        has_header = True
    names = [name.strip() for name in rows.pop(0)] if has_header else [f"Column {index + 1}" for index in range(width)]
    if len(set(names)) != len(names):
        raise MappingRequired("表格列名重复，请使用唯一列名。", {"columns": names})
    if not rows:
        raise ImportError("表格只有列名，没有数值行。")
    try:
        values = np.asarray([[float(value) for value in row] for row in rows], dtype=float)
    except ValueError as error:
        raise ImportError("表格包含缺失值或非数值内容。") from error
    if not np.isfinite(values).all():
        raise ImportError("表格包含 NaN 或 Infinity。")
    options = {"columns": names, "has_header": has_header}
    time_index = None
    if mapping and mapping.time_column is not None:
        time_index = _column_index(mapping.time_column, names)
    else:
        recognized = [index for index, name in enumerate(names)
                      if _header_parts(name)[0].lower() in {"time", "time_s", "t", "时间"}]
        if len(recognized) > 1:
            raise MappingRequired("表格有多个时间候选列，请明确选择。", options)
        if recognized:
            time_index = recognized[0]
    inferred_rate, time_origin, time_unit, source_time_unit = None, None, None, None
    if time_index is not None:
        _, source_time_unit = _header_parts(names[time_index])
        time_unit = (mapping.time_unit if mapping and mapping.time_unit is not None
                     else declarations.get("time_unit", source_time_unit))
        if time_unit is None:
            raise MappingRequired("时间列没有单位，请明确选择 s、ms 或 us 后再推算采样率。",
                                  {**options, "needs_time_unit": True, "time_column": names[time_index],
                                   "time_unit_choices": ["s", "ms", "us"]})
        if time_unit not in {"s", "ms", "us", "µs"}:
            raise ImportError("时间列单位须为 s、ms 或 us。")
        time_factor = {"s": 1.0, "ms": 0.001, "us": 1e-6, "µs": 1e-6}[time_unit]
        times = values[:, time_index] * time_factor
        steps = np.diff(times)
        if len(times) < 2 or np.any(steps <= 0):
            raise ImportError("时间列须至少包含两个严格递增的时间点。")
        delta = float(np.mean(steps))
        if not np.allclose(steps, delta, rtol=1e-5, atol=max(1e-12, abs(delta) * 1e-8)):
            raise ImportError("时间轴不均匀，不能静默重采样；请检查原始数据。")
        inferred_rate, time_origin = 1.0 / delta, float(times[0])
    source_rate = declarations.get("sample_rate", declarations.get("fs"))
    explicit_rate = mapping.sample_rate if mapping and mapping.sample_rate else source_rate
    rate = positive_rate(explicit_rate) if explicit_rate is not None else inferred_rate
    if rate is None:
        raise MappingRequired("文件没有采样率或可推算的时间列，请明确填写采样率。", {**options, "needs_sample_rate": True})
    if inferred_rate is not None and not np.isclose(rate, inferred_rate, rtol=1e-6, atol=1e-8):
        raise ImportError("映射采样率与文件时间轴推算的采样率不一致。")
    columns = ([_column_index(value, names) for value in mapping.data_columns]
               if mapping and mapping.data_columns else [index for index in range(width) if index != time_index])
    if not columns or len(set(columns)) != len(columns) or time_index in columns:
        raise ImportError("数据列须非空、不得重复，且不得包含时间列。")
    if not has_header and (not mapping or not mapping.data_columns):
        raise MappingRequired("无列名表格须明确选择数值列。", options)
    header_units = [_header_parts(names[index])[1] for index in columns]
    source_units = declarations.get("units", declarations.get("unit"))
    if source_units is not None and "units" in declarations:
        source_units = [unit.strip() for unit in source_units.split(",")]
        if len(source_units) == width:
            source_units = [source_units[index] for index in columns]
    if source_units is None and all(unit is not None for unit in header_units):
        source_units = header_units
    units = resolve_units(len(columns), mapping, source_units, options=options)
    channel_names = [_header_parts(names[index])[0] for index in columns]
    if mapping and mapping.channels:
        if len(mapping.channels) != len(columns):
            raise ImportError("通道名称数量与所选数据列不一致。")
        channel_names = list(mapping.channels)
    channels = tuple(ChannelInfo(name, unit, "sound_pressure" if unit == "Pa" else "unknown",
                                 scaled=unit in {"V", "Pa", "m/s²", "g"},
                                 metadata={"source_column": names[index], "source_unit": header_units[number]})
                     for number, (name, unit, index) in enumerate(zip(channel_names, units, columns)))
    return make_signal(path, values[:, columns].T, rate, channels, mapping=mapping,
                       quality=["unknown_source_unit"] if "unknown" in units else [],
                       metadata={"format": "numeric_table", "text_encoding": encoding, "delimiter": delimiter,
                                 "source_columns": names, "selected_columns": [names[index] for index in columns],
                                 "time_column": names[time_index] if time_index is not None else None,
                                 "time_unit": time_unit, "source_time_unit": source_time_unit,
                                 "time_origin_seconds": time_origin, "inferred_sample_rate": inferred_rate,
                                 "declared_metadata": declarations, "source_dtype": "decimal numeric text"})
