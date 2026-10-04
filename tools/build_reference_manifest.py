r"""Reproducible read-only corpus manifest and HEAD Pa reference regression.

Run from the project root:
    .venv\Scripts\python.exe tools/build_reference_manifest.py

Outputs evidence for input inventory and an existing regression corpus. It
does not exercise product import/calibration and is not an absolute calibration
or blind validation of an acoustic program.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path, PurePosixPath
import struct
import sys
import wave
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAINING_IDS = tuple(f"01-{number:02d}{direction}" for number in range(1, 5) for direction in ("CW", "CCW"))
EXPECTED_COUNTS = {"wav": 64, "png": 64, "hdf": 66}


class ManifestError(ValueError):
    """Corpus inventory, pairing, or reference calculation is invalid."""


def _load_head_reader():
    name = "autoacoustics_corpus_head_reference"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "tests/reference/head_pa_reader.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_wav_bytes(data: bytes) -> tuple[np.ndarray, dict]:
    """Independent corpus decoder: known mono 48 kHz 16-bit PCM only."""
    try:
        with wave.open(io.BytesIO(data), "rb") as stream:
            channels, bits, fs, frames = (stream.getnchannels(), stream.getsampwidth() * 8,
                                         stream.getframerate(), stream.getnframes())
            if channels != 1 or bits != 16 or fs != 48000 or stream.getcomptype() != "NONE":
                raise ManifestError("Corpus WAV must be mono 48 kHz 16-bit uncompressed PCM")
            payload = stream.readframes(frames)
            if frames <= 0 or len(payload) != frames * 2:
                raise ManifestError("Corpus WAV payload is empty or truncated")
    except (wave.Error, EOFError) as error:
        raise ManifestError(f"Invalid corpus WAV: {error}") from error
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0
    return samples, {"channels": channels, "frames": frames, "fs_hz": float(fs),
                     "unit": "FS", "encoding": "PCM_16", "duration_seconds": frames / fs}


def _validated_pair(pair):
    x, y = (np.asarray(value, dtype=np.float64).reshape(-1) for value in pair)
    if x.size == 0 or x.size != y.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ManifestError("Reference pair must have equal nonempty finite sample arrays")
    return x, y


def fit_reference_pairs(pairs: Mapping, training_ids: tuple[str, ...] = TRAINING_IDS) -> dict:
    """Fit y = k*x through zero using training only, preserving raw/DC content."""
    if not training_ids or len(set(training_ids)) != len(training_ids):
        raise ManifestError("Training IDs must be nonempty and unique")
    if any(item not in pairs for item in training_ids):
        raise ManifestError("A required training pair is missing")
    numerator, denominator = 0.0, 0.0
    for item in training_ids:
        x, y = _validated_pair(pairs[item])
        numerator += float(np.dot(x, y))
        denominator += float(np.dot(x, x))
    if denominator <= 0:
        raise ManifestError("Training WAV energy is zero")
    scale = numerator / denominator
    if not math.isfinite(scale) or scale <= 0:
        raise ManifestError("Pa/full-scale reference must be positive and finite")
    errors = {}
    for item in sorted(pairs):
        x, y = _validated_pair(pairs[item])
        calibrated = scale * x
        reference_rms = math.sqrt(float(np.mean(y * y)))
        if reference_rms == 0 or np.std(x) == 0 or np.std(y) == 0:
            raise ManifestError("A corpus pair has no usable reference energy or correlation")
        wav_rms = math.sqrt(float(np.mean(calibrated * calibrated)))
        rmse = math.sqrt(float(np.mean((calibrated - y) ** 2)))
        errors[item] = {
            "split": "training" if item in training_ids else "existing_holdout_regression",
            "frames": int(x.size), "reference_rms_pa": reference_rms,
            "calibrated_wav_rms_pa": wav_rms, "waveform_rmse_pa": rmse,
            "rms_difference_db": 20 * math.log10(wav_rms / reference_rms),
            "normalized_rmse_percent": 100 * rmse / reference_rms,
            "correlation": float(np.corrcoef(calibrated, y)[0, 1]),
        }
    held = [error for key, error in errors.items() if key not in training_ids]
    worst = {
        "absolute_rms_difference_db": max((abs(error["rms_difference_db"]) for error in held), default=None),
        "normalized_rmse_percent": max((error["normalized_rmse_percent"] for error in held), default=None),
        "minimum_correlation": min((error["correlation"] for error in held), default=None),
    }
    return {
        "method": "least_squares_through_zero_on_original_samples", "dc_removed": False,
        "pa_per_full_scale": scale, "training_ids": list(training_ids),
        "training_count": len(training_ids), "holdout_count": len(held),
        "independent_absolute_calibration": False, "blind_holdout": False,
        "provenance": "HEAD Pa reference conversion for this archive only",
        "limitation": "All 56 holdout files were inspected during planning; this is existing corpus regression. Product import, calibration and independent absolute calibration remain separate gates.",
        "holdout_thresholds": {"absolute_rms_difference_db": 0.05, "normalized_rmse_percent": 0.2,
                               "minimum_correlation": 0.99999},
        "holdout_worst": worst, "pairs": errors,
    }


class _ZipPairs(Mapping):
    """Decode a single pair on demand; don't retain all corpus arrays in RAM."""

    def __init__(self, archive: zipfile.ZipFile, files: dict, ids: list[str]):
        self.archive, self.files, self.ids = archive, files, ids

    def __getitem__(self, key):
        if key not in self.ids:
            raise KeyError(key)
        x, _ = read_wav_bytes(self.archive.read(self.files["wav"][key]))
        head = _load_head_reader().read_head_pa_bytes(self.archive.read(self.files["hdf"][key]))
        if x.size != head.frames or head.fs_hz != 48000.0:
            raise ManifestError(f"WAV/HEAD frame or sample-rate mismatch: {key}")
        return x, head.samples[0]

    def __iter__(self) -> Iterator[str]:
        return iter(self.ids)

    def __len__(self):
        return len(self.ids)


def build_manifest(path: str | Path, *, training_ids: tuple[str, ...] = TRAINING_IDS,
                   expected_counts: dict | None = EXPECTED_COUNTS) -> dict:
    path = Path(path)
    original_hash = sha256_file(path)
    files = {kind: {} for kind in EXPECTED_COUNTS}
    members = []
    try:
        with zipfile.ZipFile(path, "r") as archive:
            for info in sorted(archive.infolist(), key=lambda item: item.filename):
                if info.is_dir():
                    continue
                member = PurePosixPath(info.filename)
                kind = member.suffix.lstrip(".").lower()
                if kind not in files:
                    raise ManifestError(f"Unexpected archive member type: {info.filename}")
                if member.is_absolute() or ".." in member.parts:
                    raise ManifestError(f"Unsafe archive member path: {info.filename}")
                if member.stem in files[kind]:
                    raise ManifestError(f"Duplicate {kind} pairing basename: {member.stem}")
                files[kind][member.stem] = info.filename
                data = archive.read(info)
                entry = {"path": info.filename, "id": member.stem, "kind": kind,
                         "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                         "frames": None, "fs_hz": None, "unit": None}
                if kind == "wav":
                    _, metadata = read_wav_bytes(data)
                    entry.update(metadata)
                elif kind == "hdf":
                    head = _load_head_reader().read_head_pa_bytes(data)
                    entry.update({"channels": 1, "frames": head.frames, "fs_hz": head.fs_hz,
                                  "duration_seconds": head.frames / head.fs_hz, "unit": head.unit,
                                  "encoding": head.encoding, "data_offset": head.data_offset,
                                  "delta_seconds_original": head.delta_seconds,
                                  "channel_name": head.channel_name,
                                  "calibration_metadata_not_applied": head.calibration_metadata,
                                  "sensor_metadata_not_applied": dict(head.sensor_metadata),
                                  "scaling_state": head.scaling_state})
                else:
                    if len(data) < 29 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
                        raise ManifestError(f"Invalid PNG inventory signature: {info.filename}")
                    width, height = struct.unpack(">II", data[16:24])
                    if width <= 0 or height <= 0:
                        raise ManifestError(f"Invalid PNG dimensions: {info.filename}")
                    entry.update({"width": width, "height": height})
                members.append(entry)
            counts = {kind: len(names) for kind, names in files.items()}
            if expected_counts is not None and counts != expected_counts:
                raise ManifestError(f"Archive counts {counts} differ from required {expected_counts}")
            wav_ids = set(files["wav"])
            if not wav_ids or not wav_ids.issubset(files["hdf"]) or (files["png"] and set(files["png"]) != wav_ids):
                raise ManifestError("Every WAV needs a same-basename HDF and PNG set must match WAV set")
            paired = sorted(wav_ids)
            reference = fit_reference_pairs(_ZipPairs(archive, files, paired), training_ids)
    except (_load_head_reader().HeadReferenceError, zipfile.BadZipFile, OSError, UnicodeError) as error:
        raise ManifestError(f"Cannot inventory corpus: {error}") from error
    if sha256_file(path) != original_hash:
        raise ManifestError("Archive changed while the manifest was being generated")
    durations = [entry["duration_seconds"] for entry in members if entry["kind"] == "wav"]
    return {
        "schema_version": 1,
        "evidence_scope": "read-only inventory and known HEAD Pa export regression; not product acceptance",
        "archive": {"name": path.name, "size_bytes": path.stat().st_size, "sha256": original_hash},
        "counts": counts, "paired_ids": paired,
        "unpaired_hdf_ids": sorted(set(files["hdf"]) - wav_ids),
        "wav_duration_range_seconds": [min(durations), max(durations)],
        "reference_fit": reference, "members": members,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=ROOT / "压缩.zip", help="Read-only original corpus")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/validation/corpus_manifest.json")
    args = parser.parse_args(argv)
    result = build_manifest(args.zip)
    if args.output.resolve() == args.zip.resolve():
        raise ManifestError("Output must not overwrite the original archive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "pa_per_full_scale": result["reference_fit"]["pa_per_full_scale"],
                      "holdout_count": result["reference_fit"]["holdout_count"],
                      "holdout_worst": result["reference_fit"]["holdout_worst"]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
