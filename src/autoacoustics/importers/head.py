"""Read the HEAD text-header layout evidenced by the supplied seat-motor corpus.

This is not the similarly named HDF5 format. Only version 4/release 6,
mono Intel FLOAT32 time data is accepted. Unsupported layouts fail before
sample decoding, rather than guessing an interleave or calibration formula.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from autoacoustics.model import ChannelInfo, ImportError, ImportMapping
from .common import make_signal


PROBE_BYTES = 16384
MAX_HEADER_BYTES = 1024 * 1024
MAX_DECODED_BYTES = 512 * 1024 * 1024
MAX_FRAMES = MAX_DECODED_BYTES // 8
MAX_FILE_BYTES = MAX_HEADER_BYTES + MAX_FRAMES * 4
_STRUCTURAL_FIELD = re.compile(r"([A-Za-z][A-Za-z0-9 #_-]*):\s*(.*?)\s*\Z")
_DATA_MARKER = re.compile(rb"data1 ([0-9]{1,12}) :\Z")
_CODE_PAGES = {"936": "cp936", "65001": "utf-8", "1252": "cp1252", "20127": "ascii"}
_ALLOWED_FIELDS = {
    "file": {"version", "release", "byte order", "kind", "start of data", "nbr of abscissa", "nbr of channel",
             "nbr of extra fields", "idx order", "ch order", "data org", "channel attributes", "scan mode"},
    "axis": {"abscissa definition", "name str", "abbreviation str", "physical quantity", "physical unit",
             "absc sort", "first value", "delta value", "nbr of scans", "distribution func"},
    "channel": {"channel definition", "name str", "abbreviation str", "title str", "physical channel nbr",
                "ch sort", "physical quantity", "physical unit", "emphasis", "calibration", "delta_headroom",
                "quantisation func", "composition", "implementation type", "graph style"},
}


def is_head_signature(prefix: bytes) -> bool:
    """Identify anchored HEAD declarations in the initial ASCII comment area."""
    if not prefix.startswith(b";"):
        return False
    comment_lines = []
    for line in prefix.splitlines():
        line = line.strip()
        if line and not line.startswith(b";"):
            break
        comment_lines.append(line)
    return any(re.fullmatch(rb"; Copyright[^\r\n]*HEAD acoustics[^\r\n]*", line)
               for line in comment_lines) and b"; HEAD acoustics datafile format" in comment_lines


def _error(message: str) -> ImportError:
    return ImportError(f"HEAD HDF：{message}")


def _integer(fields: dict[str, str], key: str, maximum: int, *, minimum: int = 1) -> int:
    value = fields.get(key, "")
    if not re.fullmatch(r"[0-9]{1,12}", value):
        raise _error(f"字段 {key} 缺失或不是有效整数。")
    number = int(value)
    if not minimum <= number <= maximum:
        raise _error(f"字段 {key} 超出支持范围 {minimum}–{maximum}。")
    return number


def _number(fields: dict[str, str], key: str, *, positive: bool = False) -> float:
    try:
        value = float(fields[key])
    except (KeyError, ValueError, OverflowError) as error:
        raise _error(f"字段 {key} 缺失或不是有效数字。") from error
    if not math.isfinite(value) or (positive and value <= 0):
        raise _error(f"字段 {key} 必须为有限{'正数' if positive else '数值'}。")
    return value


def _expect(fields: dict[str, str], expected: dict[str, str], section: str):
    for key, value in expected.items():
        if fields.get(key) != value:
            raise _error(f"不支持 {section} 的 {key}={fields.get(key)!r}；当前要求 {value!r}。")


def _read_header(path: Path):
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise _error(f"文件超过当前导入资源上限 {MAX_FILE_BYTES} 字节。")
    with path.open("rb") as source:
        prefix = source.read(PROBE_BYTES)
        if not is_head_signature(prefix):
            raise _error("缺少已识别的 HEAD 文本头签名。")
        # Read an explicitly declared offset, never a guessed default. It must
        # be in the bounded global section, before any axis/channel definition.
        control = re.split(rb"(?m)^abscissa definition:", prefix, maxsplit=1)[0]
        offsets = re.findall(rb"(?m)^start of data:[ \t]*([^\r\n]+)", control)
        if len(offsets) != 1:
            raise _error("start of data 缺失或重复；无法确定载荷起点。")
        try:
            offset_text = offsets[0].decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise _error("start of data 不是 ASCII 整数。") from error
        offset = _integer({"start of data": offset_text}, "start of data", MAX_HEADER_BYTES)
        if offset > size:
            raise _error("头部被截断，声明的数据起点超过文件长度。")
        source.seek(0)
        header = source.read(offset)
    marker = _DATA_MARKER.search(header)
    if marker is None:
        raise _error("data1 载荷长度声明缺失或未恰好结束于 start of data。")
    code_pages = re.findall(rb"(?m)^;#code page:[ \t]*([^\r\n]+)", header[:marker.start()])
    if len(code_pages) > 1:
        raise _error("code page 字段重复。")
    try:
        code_page = code_pages[0].decode("ascii").strip() if code_pages else "20127"
        encoding = _CODE_PAGES[code_page]
        text = header[:marker.start()].decode(encoding, errors="strict")
    except (KeyError, UnicodeDecodeError) as error:
        raise _error("文本编码未获支持或头部包含不符合声明编码的字节。") from error
    return text, offset, int(marker.group(1)), size, code_page, encoding


def _parse_sections(text: str):
    sections: dict[str, dict[str, str]] = {"file": {}, "axis": {}, "channel": {}}
    section = "file"
    comments = {name: {} for name in sections}
    metadata_lines = []
    in_metadata = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "; comment" or re.match(r"comment [0-9]+ :", line):
            in_metadata = True
        if in_metadata:
            if line:
                metadata_lines.append(line)
            continue
        if not line:
            continue
        if line.startswith(";"):
            match = _STRUCTURAL_FIELD.fullmatch(line[2:].strip()) if line.startswith(";#") else None
            if match:
                key, value = match.groups()
                if key in comments[section]:
                    raise _error(f"{section} 注释字段 {key} 重复。")
                comments[section][key] = value
            continue
        match = _STRUCTURAL_FIELD.fullmatch(line)
        if match is None:
            raise _error(f"{section} 含有无法识别的结构字段：{line[:100]!r}。")
        key, value = match.groups()
        if key == "abscissa definition":
            if section != "file":
                raise _error("abscissa definition 重复或顺序错误。")
            section = "axis"
        elif key == "channel definition":
            if section != "axis":
                raise _error("channel definition 重复或顺序错误；多通道布局尚未验证。")
            section = "channel"
        if key in sections[section]:
            raise _error(f"{section} 字段 {key} 重复。")
        if key not in _ALLOWED_FIELDS[section]:
            raise _error(f"{section} 含有未验证的结构字段 {key!r}。")
        sections[section][key] = value
    return sections, comments, "\n".join(metadata_lines)


def _sensor_metadata(metadata: str):
    # The real export embeds proprietary escape sequences in Quantity.Unit,
    # so parse only SensorInfo's opening tag. Never interpret external XML or
    # executable strings. The entire source value remains available as text.
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", metadata, flags=re.IGNORECASE):
        raise _error("XML 含有禁止的 DOCTYPE/ENTITY 声明。")
    matches = re.findall(r"(?m)^_SI_Sensor\s*=\s*string\s*\(\s*(.*?)\s*\)\s*$", metadata)
    if len(matches) > 1:
        raise _error("_SI_Sensor 元数据重复。")
    if not matches:
        return {}, None
    raw = matches[0]
    opening = re.match(r"<SensorInfo\b[^>]*>", raw)
    if opening is None:
        raise _error("XML SensorInfo 缺少可解析的起始标签。")
    tag = opening.group(0)
    if not tag.endswith("/>"):
        tag = tag[:-1] + "/>"
    try:
        sensor = dict(ET.fromstring(tag).attrib)
    except ET.ParseError as error:
        raise _error("XML SensorInfo 属性不完整或包含未解析实体。") from error
    return sensor, raw


def _unit_and_quantity(channel: dict[str, str], mapping: ImportMapping | None):
    original_unit = channel.get("physical unit", "").strip()
    quantity = channel.get("physical quantity", "").strip()
    units = {"Pa": ("Pa", 1.0), "V": ("V", 1.0), "mV": ("V", 0.001),
             "FS": ("FS", 1.0), "g": ("g", 1.0), "m/s²": ("m/s²", 1.0),
             "m/s^2": ("m/s²", 1.0)}
    unit, factor = units.get(original_unit, ("unknown", 1.0))
    if unit == "Pa" and quantity != "sound pressure":
        raise _error("Pa 单位与 physical quantity 不一致，不能作为声压导入。")
    if unit in {"g", "m/s²"} and quantity != "acceleration":
        raise _error("加速度单位与 physical quantity 不一致。")
    selected = None
    if mapping:
        if mapping.channel_units:
            if len(mapping.channel_units) != 1:
                raise _error("逐通道单位映射数量与单通道文件不一致。")
            selected = mapping.channel_units[0]
        elif mapping.unit != "unknown":
            selected = mapping.unit
    if selected is not None and selected != unit:
        raise _error("文件内单位不能用导入映射重新标记；请通过显式校准转换。")
    physical_quantity = ("sound_pressure" if unit == "Pa" else
                         "acceleration" if unit in {"g", "m/s²"} else "unknown")
    return unit, factor, original_unit, physical_quantity


def import_head(path: Path, mapping: ImportMapping | None = None):
    """Import one verified mono FLOAT32 HEAD record without sensor rescaling."""
    text, offset, declared_bytes, size, code_page, encoding = _read_header(path)
    sections, comments, metadata_text = _parse_sections(text)
    global_fields, axis, channel = (sections[name] for name in ("file", "axis", "channel"))
    _expect(global_fields, {"version": "4", "release": "6", "byte order": "Intel", "kind": "Time data",
                           "start of data": str(offset), "nbr of abscissa": "1", "nbr of channel": "1",
                           "nbr of extra fields": "0", "idx order": "1", "ch order": "1",
                           "data org": "a1b1 a2b2", "channel attributes": "private", "scan mode": "simultaneous"}, "file")
    _expect(axis, {"abscissa definition": "1", "physical quantity": "time", "physical unit": "s",
                   "absc sort": "calc", "distribution func": "linear"}, "axis")
    _expect(channel, {"channel definition": "1", "ch sort": "all data", "quantisation func": "linear",
                      "composition": "sample", "implementation type": "FLOAT32"}, "channel")
    if "emphasis" in channel:
        _expect(channel, {"emphasis": "False"}, "channel")
    if "delta_headroom" in channel and _number(channel, "delta_headroom") != 0:
        raise _error("非零 delta_headroom 的缩放含义尚未验证。")
    frames = _integer(axis, "nbr of scans", MAX_FRAMES)
    delta = _number(axis, "delta value", positive=True)
    origin = _number(axis, "first value")
    fs = 1.0 / delta
    if not math.isfinite(fs):
        raise _error("delta value 的倒数不是有限采样率。")
    nearest = round(fs)
    if nearest > 0 and abs(fs - nearest) <= 1e-9:
        fs = float(nearest)
    if mapping and mapping.sample_rate is not None and not math.isclose(mapping.sample_rate, fs, rel_tol=1e-10):
        raise _error("文件内时间轴不能用导入映射改采样率。")
    if declared_bytes != frames * 4 or size != offset + declared_bytes:
        raise _error("扫描数、data1 字节数和实际载荷长度不一致；文件截断或含未知尾部。")
    unit, unit_factor, source_unit, physical_quantity = _unit_and_quantity(channel, mapping)
    sensor, sensor_raw = _sensor_metadata(metadata_text)
    channel_metadata = {"source_unit": source_unit, "source_quantity": channel.get("physical quantity"),
                        "unit_conversion_factor": unit_factor, "sensor": sensor, "sensor_raw": sensor_raw,
                        "calibration_applied_by_importer": False, "head_fields": channel,
                        "head_comment_fields": comments["channel"]}
    if "calibration" in channel:
        channel_metadata["head_calibration"] = _number(channel, "calibration")
    name = comments["channel"].get("ext name str") or channel.get("name str") or "Channel 1"
    if mapping and mapping.channels and tuple(mapping.channels) != (name,):
        raise _error("所选通道与 HEAD 文件通道名不一致。")
    info = ChannelInfo(name, unit, physical_quantity, scaled=unit in {"V", "Pa", "g", "m/s²"},
                       metadata=channel_metadata)
    metadata = {"format": "HEAD_HDF", "calibration_applied_by_importer": False,
                "time_origin_seconds": origin, "source_quantity": channel.get("physical quantity"),
                "head": {"version": 4, "release": 6, "byte_order": "Intel", "implementation_type": "FLOAT32",
                         "data_offset": offset, "data_bytes": declared_bytes, "delta_seconds": delta,
                         "code_page": code_page, "text_encoding": encoding, "fields": sections,
                         "comment_fields": comments, "comment_metadata": metadata_text,
                         "support_evidence": "provided_corpus_mono_float32"}}
    quality = ["unknown_frontend_overload_status"]
    if unit == "unknown":
        quality.append("unknown_source_unit")
    samples = np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(1, frames))
    try:
        if not np.isfinite(samples).all():
            raise _error("载荷包含 NaN 或无穷值。")
        values = samples if unit_factor == 1.0 else samples.astype(np.float64) * unit_factor
        return make_signal(path, values, fs, (info,), metadata=metadata, mapping=mapping, quality=quality)
    finally:
        samples._mmap.close()
