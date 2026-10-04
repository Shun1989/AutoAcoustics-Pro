"""Strict reader for the supplied corpus's HEAD Pa reference exports only.

This module is a test reference, not a product importer. It does not infer
unknown HEAD encodings or convert calibration metadata into a scaling factor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np


class HeadReferenceError(ValueError):
    """Reference header or sample payload is unsupported or incomplete."""


@dataclass(frozen=True)
class HeadPaReference:
    samples: np.ndarray
    fs_hz: float
    frames: int
    unit: str
    channel_name: str
    delta_seconds: float
    calibration_metadata: float
    sensor_metadata: Mapping[str, str]
    scaling_state: str = "already_physical"
    data_offset: int = 65536
    encoding: str = "little-endian FLOAT32"


def _number(fields: dict[str, str], key: str, *, integer: bool = False):
    try:
        value = int(fields[key]) if integer else float(fields[key])
    except (KeyError, ValueError, OverflowError) as error:
        raise HeadReferenceError(f"Missing or invalid HEAD field: {key}") from error
    if not math.isfinite(value):
        raise HeadReferenceError(f"Nonfinite HEAD field: {key}")
    return value


def _expect(fields: dict[str, str], expected: dict[str, str], section: str):
    for key, value in expected.items():
        if fields.get(key) != value:
            raise HeadReferenceError(
                f"Unsupported {section} {key}: {fields.get(key)!r}; expected {value!r}"
            )


def read_head_pa_bytes(data: bytes) -> HeadPaReference:
    """Read one complete mono Intel FLOAT32/Pa export with the known layout."""
    if len(data) < 65536:
        raise HeadReferenceError("HEAD reference header is truncated")
    header = data[:65536]
    if not re.search(rb"(?m)^; Copyright[^\r\n]*HEAD acoustics", header):
        raise HeadReferenceError("HEAD copyright signature is missing")
    if b"; HEAD acoustics datafile format" not in header:
        raise HeadReferenceError("HEAD datafile signature is missing")
    # The corpus's header declares the exact payload length immediately before
    # sample data. No guessed offset, trailing bytes, or alternate organization.
    declaration = re.search(rb"data1\s+(\d+)\s+:\Z", header)
    if not declaration:
        raise HeadReferenceError("Known HEAD data1 payload declaration is missing")
    sections: dict[str, dict[str, str]] = {"file": {}, "axis": {}, "channel": {}}
    current = "file"
    for raw_line in header[:declaration.start()].splitlines():
        line = raw_line.decode("cp936", errors="strict").strip()
        if not line or line.startswith(";"):
            continue
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9 #_-]*):\s*(.*?)\s*", line)
        if not match:
            continue  # comment/XML metadata is not a structural field
        key, value = match.groups()
        if key == "abscissa definition":
            if current != "file":
                raise HeadReferenceError("Duplicate or misplaced abscissa definition")
            current = "axis"
        elif key == "channel definition":
            if current != "axis":
                raise HeadReferenceError("Duplicate or misplaced channel definition")
            current = "channel"
        if key in sections[current]:
            raise HeadReferenceError(f"Duplicate HEAD {current} field: {key}")
        sections[current][key] = value
    _expect(sections["file"], {
        "version": "4", "release": "6", "byte order": "Intel", "kind": "Time data",
        "start of data": "65536", "nbr of abscissa": "1", "nbr of channel": "1",
        "nbr of extra fields": "0", "idx order": "1", "ch order": "1",
        "data org": "a1b1 a2b2", "channel attributes": "private", "scan mode": "simultaneous",
    }, "file")
    _expect(sections["axis"], {
        "abscissa definition": "1", "physical quantity": "time", "physical unit": "s",
        "absc sort": "calc", "first value": "0", "distribution func": "linear",
    }, "axis")
    _expect(sections["channel"], {
        "channel definition": "1", "physical quantity": "sound pressure", "physical unit": "Pa",
        "ch sort": "all data", "quantisation func": "linear", "composition": "sample",
        "implementation type": "FLOAT32",
    }, "channel")
    frames = _number(sections["axis"], "nbr of scans", integer=True)
    delta = _number(sections["axis"], "delta value")
    calibration = _number(sections["channel"], "calibration")
    if frames <= 0 or delta <= 0:
        raise HeadReferenceError("HEAD scan count and sample interval must be positive")
    payload_bytes = frames * 4
    if int(declaration.group(1)) != payload_bytes or len(data) != 65536 + payload_bytes:
        raise HeadReferenceError("HEAD declared scan count and payload length differ")
    samples = np.frombuffer(data, dtype="<f4", count=frames, offset=65536).astype(np.float64)
    if not np.isfinite(samples).all():
        raise HeadReferenceError("HEAD reference contains nonfinite samples")
    samples = samples[np.newaxis, :]
    samples.flags.writeable = False
    fs = 1.0 / delta
    # Decimal representation of 1/48000 gives 48000.00000000008. Correct only
    # this floating representation tolerance, preserving the original delta.
    nearest = round(fs)
    if abs(fs - nearest) <= 1e-9:
        fs = float(nearest)
    sensor_match = re.search(rb"_SI_Sensor\s*=\s*string\s*\(\s*<SensorInfo\s+([^>]+)>", header)
    sensor = {}
    if sensor_match:
        sensor = {
            key.decode("ascii"): value.decode("cp936")
            for key, value in re.findall(rb'([A-Za-z][A-Za-z0-9]*)="([^"<>]*)"', sensor_match.group(1))
        }
    extended_name = re.search(rb"(?m)^;#ext name str:\s*([^\r\n]+)", header)
    name = (extended_name.group(1).decode("cp936").strip() if extended_name
            else sections["channel"].get("name str", ""))
    return HeadPaReference(
        samples=samples, fs_hz=fs, frames=frames, unit="Pa", channel_name=name,
        delta_seconds=delta, calibration_metadata=calibration,
        sensor_metadata=MappingProxyType(sensor),
    )


def read_head_pa(path: str | Path) -> HeadPaReference:
    return read_head_pa_bytes(Path(path).read_bytes())
