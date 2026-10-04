"""Independent ZIP inventory and Pa export regression, not product acceptance."""

from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import wave
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_tool(relative: str, name: str):
    path = ROOT / relative
    if not path.is_file():
        raise AssertionError(f"Missing implementation: {relative}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def head_blob(values=(1.0, -2.0, 0.125), **overrides):
    fields = {
        "version": "4", "release": "6", "byte order": "Intel",
        "kind": "Time data", "start of data": "65536", "nbr of abscissa": "1",
        "nbr of channel": "1", "nbr of extra fields": "0", "idx order": "1",
        "ch order": "1", "data org": "a1b1 a2b2", "channel attributes": "private",
        "scan mode": "simultaneous", "abscissa definition": "1", "name str": "Time",
        "physical quantity": "time", "physical unit": "s", "absc sort": "calc",
        "first value": "0", "delta value": "2.08333333333333e-005",
        "nbr of scans": str(len(values)), "distribution func": "linear",
    }
    # Duplicate quantity/unit names are deliberately in separate sections.
    channel = {
        "channel definition": "1", "name str": "Ch. 8@SQuadriga",
        "physical quantity": "sound pressure", "physical unit": "Pa",
        "ch sort": "all data", "calibration": "83.084850197733005",
        "quantisation func": "linear", "composition": "sample",
        "implementation type": "FLOAT32",
    }
    for key, value in overrides.items():
        target = channel if key in channel else fields
        target[key] = str(value)
    lines = ["; Copyright 1999 HEAD acoustics GmbH, Germany",
             "; HEAD acoustics datafile format"]
    lines += [f"{key}: {value}" for key, value in fields.items()]
    lines += [f"{key}: {value}" for key, value in channel.items()]
    prefix = ("\r\n".join(lines) + "\r\n").encode("ascii")
    payload = np.asarray(values, dtype="<f4").tobytes()
    marker = f"data1 {len(payload)} :".encode("ascii")
    return prefix + b"\t" * (65536 - len(prefix) - len(marker)) + marker + payload


def wav_blob(samples):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return buffer.getvalue()


class HeadPaReaderTests(unittest.TestCase):
    def setUp(self):
        self.reader = load_tool("tests/reference/head_pa_reader.py", "head_pa_reference")

    def test_little_endian_pa_passes_through_without_calibration(self):
        result = self.reader.read_head_pa_bytes(head_blob())
        np.testing.assert_array_equal(result.samples, [[1.0, -2.0, 0.125]])
        self.assertEqual(result.fs_hz, 48000.0)
        self.assertEqual(result.frames, 3)
        self.assertEqual(result.unit, "Pa")
        self.assertEqual(result.calibration_metadata, 83.084850197733005)
        self.assertEqual(result.scaling_state, "already_physical")
        self.assertFalse(result.samples.flags.writeable)

    def test_unknown_encoding_unit_layout_and_version_are_rejected(self):
        for key, value in [("implementation type", "INT16"), ("byte order", "Motorola"),
                           ("physical unit", "V"), ("nbr of channel", "2"),
                           ("version", "5"), ("start of data", "8192"),
                           ("data org", "unknown"), ("absc sort", "explicit")]:
            with self.subTest(key=key):
                with self.assertRaises(self.reader.HeadReferenceError):
                    self.reader.read_head_pa_bytes(head_blob(**{key: value}))

    def test_truncation_count_mismatch_nonfinite_and_bad_magic_rejected(self):
        blob = head_blob()
        for data in [blob[:65530], blob[:-1], blob + b"\0", head_blob(**{"nbr of scans": "4"}),
                     head_blob((float("nan"),)), blob.replace(b"HEAD acoustics", b"OTHER acoustic")]:
            with self.subTest(length=len(data)):
                with self.assertRaises(self.reader.HeadReferenceError):
                    self.reader.read_head_pa_bytes(data)

    def test_duplicate_critical_header_field_rejected(self):
        original = head_blob()
        insertion = b"\r\nbyte order: Intel"
        prefix = original[:65536].replace(b"byte order: Intel", b"byte order: Intel" + insertion, 1)
        prefix = prefix.replace(b"\t" * len(insertion), b"", 1)
        blob = prefix + original[65536:]
        with self.assertRaisesRegex(self.reader.HeadReferenceError, "Duplicate HEAD file field"):
            self.reader.read_head_pa_bytes(blob)

    def test_invalid_scan_interval_and_calibration_metadata_rejected(self):
        for key, value in [("delta value", "0"), ("delta value", "nan"),
                           ("nbr of scans", "-3"), ("calibration", "nan")]:
            with self.subTest(key=key, value=value):
                with self.assertRaises(self.reader.HeadReferenceError):
                    self.reader.read_head_pa_bytes(head_blob(**{key: value}))


class ReferenceManifestTests(unittest.TestCase):
    def setUp(self):
        self.builder = load_tool("tools/build_reference_manifest.py", "corpus_manifest_tool")

    def test_fit_uses_training_only_and_records_per_pair_errors(self):
        x = np.asarray([0.125, -0.25, 0.375, -0.5])
        pairs = {
            "train": (x, 2.0 * x),
            "hold": (x, 2.02 * x),
        }
        result = self.builder.fit_reference_pairs(pairs, ("train",))
        self.assertEqual(result["pa_per_full_scale"], 2.0)
        self.assertEqual(result["training_ids"], ["train"])
        self.assertEqual(result["holdout_count"], 1)
        error = result["pairs"]["hold"]
        self.assertAlmostEqual(error["normalized_rmse_percent"], 100 * 0.02 / 2.02)
        self.assertAlmostEqual(error["rms_difference_db"], 20 * np.log10(2.0 / 2.02))
        self.assertAlmostEqual(error["correlation"], 1.0)
        self.assertFalse(result["independent_absolute_calibration"])
        self.assertFalse(result["blind_holdout"])

    def test_zip_inventory_preserves_original_hash_and_checks_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.zip"
            wav = wav_blob([4096, -8192, 12288, -16384])
            head = head_blob((0.25, -0.5, 0.75, -1.0))
            # PNG inventory only needs its signature and IHDR width/height.
            png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 640, 480) + b"\x08\x02\0\0\0"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("WAV/one.wav", wav)
                archive.writestr("HDF/one.hdf", head)
                archive.writestr("PNG/one.png", png)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            result = self.builder.build_manifest(path, training_ids=("one",), expected_counts=None)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            self.assertEqual(result["archive"]["sha256"], before)
            self.assertEqual(result["counts"], {"wav": 1, "png": 1, "hdf": 1})
            self.assertEqual(result["paired_ids"], ["one"])
            self.assertEqual(result["reference_fit"]["pa_per_full_scale"], 2.0)
            entries = {entry["path"]: entry for entry in result["members"]}
            self.assertEqual(entries["WAV/one.wav"]["frames"], 4)
            self.assertEqual(entries["HDF/one.hdf"]["unit"], "Pa")
            self.assertEqual(entries["PNG/one.png"]["width"], 640)
            self.assertEqual(entries["PNG/one.png"]["sha256"], hashlib.sha256(png).hexdigest())
            with self.assertRaises(self.builder.ManifestError):
                self.builder.build_manifest(path, training_ids=("one",))

    def test_alignment_mismatch_and_duplicate_basename_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("WAV/one.wav", wav_blob([4096, -8192]))
                archive.writestr("HDF/one.hdf", head_blob((0.25, -0.5, 0.75)))
            with self.assertRaises(self.builder.ManifestError):
                self.builder.build_manifest(path, training_ids=("one",), expected_counts=None)
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr("OTHER/one.wav", wav_blob([4096, -8192]))
            with self.assertRaises(self.builder.ManifestError):
                self.builder.build_manifest(path, training_ids=("one",), expected_counts=None)

    def test_reference_wav_unknown_codec_or_truncation_rejected(self):
        complete = wav_blob([4096, -8192])
        for blob in [complete[:-1], complete.replace(b"\x80\xbb\x00\x00", b"\x44\xac\x00\x00", 1)]:
            with self.subTest(length=len(blob)):
                with self.assertRaises(self.builder.ManifestError):
                    self.builder.read_wav_bytes(blob)

    def test_zero_training_energy_or_invalid_pair_cannot_create_a_profile(self):
        for pairs in [{"one": (np.zeros(3), np.ones(3))},
                      {"one": (np.ones(3), np.ones(4))},
                      {"one": (np.ones(3), np.full(3, np.nan))}]:
            with self.subTest(pair=pairs):
                with self.assertRaises(self.builder.ManifestError):
                    self.builder.fit_reference_pairs(pairs, ("one",))

    def test_real_corpus_has_64_aligned_triplets_and_56_regression_pairs(self):
        path = ROOT / "压缩.zip"
        self.assertTrue(path.is_file(), "Real corpus is required for this acceptance tool test")
        result = self.builder.build_manifest(path)
        self.assertEqual(result["counts"], {"wav": 64, "png": 64, "hdf": 66})
        self.assertEqual(len(result["paired_ids"]), 64)
        self.assertEqual(len(result["unpaired_hdf_ids"]), 2)
        self.assertEqual(result["reference_fit"]["holdout_count"], 56)
        self.assertLess(result["reference_fit"]["holdout_worst"]["absolute_rms_difference_db"], 0.05)
        self.assertLess(result["reference_fit"]["holdout_worst"]["normalized_rmse_percent"], 0.2)
        self.assertGreater(result["reference_fit"]["holdout_worst"]["minimum_correlation"], 0.99999)


if __name__ == "__main__":
    unittest.main()
