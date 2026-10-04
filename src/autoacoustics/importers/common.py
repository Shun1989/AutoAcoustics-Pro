from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from autoacoustics.model import ChannelInfo, ImportError, ImportMapping, MappingRequired, SignalData, ValidationError, to_plain


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_signal(path: Path, samples, fs: float, channels: tuple[ChannelInfo, ...], *,
                metadata: dict | None = None, mapping: ImportMapping | None = None,
                quality=()) -> SignalData:
    metadata = dict(metadata or {})
    metadata["import_mapping"] = to_plain(mapping) if mapping else None
    try:
        return SignalData(np.asarray(samples), fs, channels, path.resolve(), file_hash(path), metadata, tuple(quality))
    except ValidationError as error:
        raise ImportError(f"导入信号无效：{error}") from error


def validate_mapping(mapping: ImportMapping | None):
    if mapping is None:
        return
    if mapping.sample_rate is not None and (
        isinstance(mapping.sample_rate, bool) or not np.isfinite(mapping.sample_rate) or mapping.sample_rate <= 0
    ):
        raise ImportError("导入映射的采样率必须为有限正数。")
    if mapping.channel_axis not in (None, 0, 1):
        raise ImportError("channel_axis 必须为 0 或 1，未知时留空。")


def plain_metadata(value):
    if isinstance(value, dict):
        return {str(key): plain_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [plain_metadata(item) for item in value]
    if isinstance(value, np.generic):
        return plain_metadata(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None or isinstance(value, (str, float, int, bool)):
        return value
    return str(value)


def canonical_unit(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    aliases = {"FS": "FS", "V": "V", "Pa": "Pa", "g": "g", "m/s²": "m/s²",
               "m/s^2": "m/s²", "m/s2": "m/s²", "unknown": "unknown"}
    return aliases.get(str(value).strip())


def resolve_units(count: int, mapping: ImportMapping | None, source_units=None, *, options=None):
    if mapping and mapping.channel_units:
        if len(mapping.channel_units) != count:
            raise ImportError("逐通道单位映射数量与所选通道不一致。")
        return tuple(mapping.channel_units)
    if mapping and mapping.unit != "unknown":
        return (mapping.unit,) * count
    if source_units is not None:
        values = list(source_units) if isinstance(source_units, (tuple, list, np.ndarray)) else [source_units]
        if len(values) == 1:
            values *= count
        units = tuple(canonical_unit(value) for value in values)
        if len(units) == count and all(unit is not None for unit in units):
            return units
    if mapping is not None:
        return ("unknown",) * count
    raise MappingRequired("文件缺少明确单位，请选择每个通道的单位；未知单位仅可做原始量查看。",
                          {**(options or {}), "needs_units": True, "unit_choices": ["unknown", "FS", "V", "Pa", "m/s²", "g"]})


def positive_rate(value):
    try:
        rate = float(value)
    except (TypeError, ValueError) as error:
        raise ImportError("采样率元数据无效。") from error
    if not np.isfinite(rate) or rate <= 0:
        raise ImportError("采样率必须为有限正数。")
    return rate
