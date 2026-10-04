"""Durable chunk journal, atomic HDF5 completion, and explicit partial recovery."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from uuid import uuid4

import h5py
import numpy as np

from autoacoustics import __version__
from autoacoustics.importers.common import file_hash
from autoacoustics.importers.registry import import_signal
from autoacoustics.model import ChannelInfo, SignalData, UNITS, to_plain
from .device import AcquisitionError, AcquisitionStatus, RecordingSession


SCHEMA_VERSION = 1


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_manifest(path, data):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(to_plain(data), output, ensure_ascii=False, allow_nan=False, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_manifest(path):
    path = Path(path).resolve()
    try:
        if path.stat().st_size > 8 * 1024 * 1024:
            raise AcquisitionError("DAQ 会话清单超过资源上限。")
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
                or data.get("schema_version") != SCHEMA_VERSION
                or data.get("status") not in {s.value for s in AcquisitionStatus}):
            raise AcquisitionError("DAQ 会话版本或状态无效。")
        if (not isinstance(data.get("session_id"), str) or not data["session_id"]
                or data.get("backend") not in {"NI_DAQMX", "REPLAY", "HEAD_FILE_HANDOFF", "SOUND_CARD"}
                or type(data.get("is_simulated")) is not bool
                or data.get("calibration_applied_to_raw") is not False):
            raise AcquisitionError("JSON 不是具有明确来源和原始缩放状态的 DAQ 会话清单。")
        for name in ("frames_read", "frames_written"):
            if type(data.get(name)) is not int or data[name] < 0:
                raise AcquisitionError("DAQ 会话帧计数须为非负整数。")
        if (any(not isinstance(data.get(name), list) for name in ("flags", "errors", "events"))
                or not isinstance(data.get("measurement_context"), dict)
                or any(not isinstance(event, dict) for event in data["events"])):
            raise AcquisitionError("DAQ 会话质量、事件或工况字段无效。")
        if data["status"] == "complete":
            _validate_complete_manifest(data)
        return path, data
    except (OSError, ValueError, AttributeError) as error:
        raise AcquisitionError(f"无法读取 DAQ 会话清单：{error}") from error


def _validate_complete_manifest(data):
    digest = data.get("data_sha256")
    channels = data.get("channels")
    rate = data.get("actual_sample_rate")
    if (not isinstance(data.get("data_file"), str) or not data["data_file"]
            or not isinstance(digest, str) or len(digest) != 64
            or not set(digest.lower()) <= set("0123456789abcdef")
            or not isinstance(channels, list) or not channels
            or any(not isinstance(channel, dict) or not isinstance(channel.get("name"), str)
                   or not channel["name"] or channel.get("unit") not in UNITS for channel in channels)):
        raise AcquisitionError("完整 DAQ 会话缺少合法的数据文件、SHA-256 或通道描述。")
    if (isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate)
            or rate <= 0 or data["frames_written"] <= 0
            or data["frames_read"] != data["frames_written"] or data["errors"]):
        raise AcquisitionError("完整 DAQ 会话的采样率、采样计数或错误状态不一致。")
    expected = data.get("frames_expected")
    if expected is not None and (type(expected) is not int or expected != data["frames_written"]):
        raise AcquisitionError("完整 DAQ 会话未达到记录的目标帧数。")


def safe_session_file(manifest_path, name):
    base = Path(manifest_path).resolve().parent
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise AcquisitionError("DAQ 文件引用须位于会话目录。")
    target = (base / name).resolve()
    if target.parent != base:
        raise AcquisitionError("DAQ 文件引用越出会话目录。")
    return target


def session_from_manifest(manifest_path, data):
    name = data.get("data_file") or data.get("recovery_file") or data.get("raw_file")
    return RecordingSession(data["session_id"], AcquisitionStatus(data["status"]), data["backend"],
        bool(data.get("is_simulated", False)), Path(manifest_path),
        safe_session_file(manifest_path, name) if name else None,
        data.get("requested_sample_rate"), data.get("actual_sample_rate"), data.get("frames_expected"),
        data.get("frames_read", 0), data.get("frames_written", 0),
        tuple(data.get("flags", [])), tuple(data.get("errors", [])))


class SessionWriter:
    """One raw append stream. A journal record commits exactly one fsynced block.

    Interrupted appends leave a recoverable prefix. HDF5 is created only while
    finalizing, so a crash does not require opening a dirty live HDF5 writer.
    """
    def __init__(self, directory, config, configured):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise AcquisitionError("录制目录必须为空，不能覆盖已有测量。")
        expected_bytes = (config.target_frames or config.block_frames) * len(config.channels) * 8 * 2
        if shutil.disk_usage(self.directory).free < config.minimum_free_bytes + expected_bytes:
            raise AcquisitionError("磁盘可用空间不足以保存原数据与终结文件。", flag="disk_space_low")
        self.config, self.configured = config, configured
        self.frames_written = self.byte_offset = 0
        self.manifest_path = self.directory / "session.json"
        self.raw_path = self.directory / "samples.raw.partial"
        self.journal_path = self.directory / "chunks.jsonl"
        self.data_path = self.directory / "samples.h5"
        self.manifest = {"schema_version": SCHEMA_VERSION, "session_id": uuid4().hex,
            "status": "recording", "backend": configured.device.backend,
            "is_simulated": configured.device.is_simulated, "software_version": __version__,
            "device": to_plain(configured.device), "configuration": to_plain(config),
            "configured": to_plain(configured), "channels": to_plain(configured.channels),
            "requested_sample_rate": configured.requested_sample_rate,
            "actual_sample_rate": configured.actual_sample_rate, "time_origin_seconds": 0.0,
            "started_at_utc": utc_now(), "ended_at_utc": None, "raw_file": self.raw_path.name,
            "journal_file": self.journal_path.name, "data_file": None, "frames_expected": config.target_frames,
            "frames_read": 0, "frames_written": 0, "flags": [], "errors": [], "events": [],
            "measurement_context": to_plain(config.measurement_context), "controlled_recording": True,
            "calibration_applied_to_raw": False, "sample_layout": "frames_by_channels_little_float64"}
        write_manifest(self.manifest_path, self.manifest)
        self.raw = self.raw_path.open("xb")
        self.journal = self.journal_path.open("x", encoding="utf-8")

    def append(self, block):
        if block.first_sample_index != self.frames_written or block.samples.shape[0] != len(self.config.channels):
            raise AcquisitionError("落盘块序号或通道数不连续。", flag="sample_gap")
        if block.frames > self.config.block_frames:
            raise AcquisitionError("样本块超过声明的块大小。", flag="invalid_sample_block")
        payload = block.samples.T.astype("<f8", copy=False).tobytes(order="C")
        if shutil.disk_usage(self.directory).free < self.config.minimum_free_bytes + len(payload):
            raise AcquisitionError("录制期间磁盘保留空间不足。", flag="disk_space_low")
        written = self.raw.write(payload)
        if written != len(payload):
            raise OSError("Short recording write")
        self.raw.flush()
        os.fsync(self.raw.fileno())
        record = {"first_sample_index": block.first_sample_index, "frames": block.frames,
                  "byte_offset": self.byte_offset, "byte_count": len(payload),
                  "sha256": hashlib.sha256(payload).hexdigest(), "timestamp_seconds": block.timestamp_seconds}
        self.journal.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        self.journal.flush()
        os.fsync(self.journal.fileno())
        self.frames_written += block.frames
        self.byte_offset += len(payload)

    def abort(self):
        for stream in (self.raw, self.journal):
            if not stream.closed:
                stream.close()

    def finalize(self, *, complete, frames_expected, frames_read, flags=(), errors=(), events=()):
        self.abort()
        self.manifest.update({"status": "finalizing", "frames_expected": frames_expected,
            "frames_read": frames_read, "frames_written": self.frames_written,
            "flags": list(flags), "errors": list(errors), "events": to_plain(events), "ended_at_utc": utc_now()})
        write_manifest(self.manifest_path, self.manifest)
        if complete:
            try:
                records, journal_flags = _verified_records(self.manifest_path, self.manifest)
                if journal_flags or sum(record["frames"] for record in records) != self.frames_written:
                    raise AcquisitionError("终结时录制块校验或帧数不一致。", flag="recording_integrity_failed")
                _build_hdf5(self.manifest_path, self.manifest, records, self.data_path, complete=True)
                self.manifest.update({"data_file": self.data_path.name, "data_sha256": file_hash(self.data_path),
                                      "journal_sha256": file_hash(self.journal_path), "status": "complete"})
                write_manifest(self.manifest_path, self.manifest)
                self.raw_path.unlink()  # finalized HDF5 retains the exact original recorded values
            except Exception as error:
                self.manifest["flags"].append(getattr(error, "flag", None) or "disk_finalize_failed")
                self.manifest["errors"].append(str(error))
                self.manifest["status"] = "failed"
                write_manifest(self.manifest_path, self.manifest)
        else:
            self.manifest["status"] = "failed"
            write_manifest(self.manifest_path, self.manifest)
        return session_from_manifest(self.manifest_path, self.manifest)


def _verified_records(manifest_path, manifest):
    raw_path = safe_session_file(manifest_path, manifest["raw_file"])
    journal_path = safe_session_file(manifest_path, manifest["journal_file"])
    channels = len(manifest["channels"])
    max_block = manifest["configuration"]["block_frames"]
    records, flags, expected, offset = [], [], 0, 0
    with raw_path.open("rb") as raw, journal_path.open("rb") as ledger:
        for line in ledger:
            if len(line) > 2048:
                raise AcquisitionError("录制块清单超出资源上限。")
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                flags.append("journal_tail_incomplete")
                break
            count = record.get("frames")
            if (not isinstance(count, int) or not 0 < count <= max_block
                    or record.get("first_sample_index") != expected or record.get("byte_offset") != offset
                    or record.get("byte_count") != count * channels * 8):
                raise AcquisitionError("录制块索引校验失败。", flag="recording_integrity_failed")
            payload = raw.read(record["byte_count"])
            if len(payload) != record["byte_count"] or hashlib.sha256(payload).hexdigest() != record.get("sha256"):
                raise AcquisitionError("录制块内容哈希校验失败。", flag="recording_integrity_failed")
            records.append(record)
            expected += count
            offset += record["byte_count"]
        if raw.read(1):
            flags.append("uncommitted_raw_tail")
    return records, flags


def _build_hdf5(manifest_path, manifest, records, destination, *, complete):
    frames = sum(record["frames"] for record in records)
    if not frames:
        raise AcquisitionError("没有可恢复的已提交样本。")
    count = len(manifest["channels"])
    raw_path = safe_session_file(manifest_path, manifest["raw_file"])
    temporary = Path(destination).with_name(f".{Path(destination).name}.{uuid4().hex}.partial")
    try:
        with h5py.File(temporary, "x") as output, raw_path.open("rb") as raw:
            samples = output.create_dataset("samples", shape=(count, frames), dtype="<f8",
                chunks=(count, min(frames, manifest["configuration"]["block_frames"])))
            names = [channel["name"] for channel in manifest["channels"]]
            units = [channel["unit"] for channel in manifest["channels"]]
            string_type = h5py.string_dtype("utf-8")
            samples.attrs.create("channel_names", names, dtype=string_type)
            samples.attrs.create("units", units, dtype=string_type)
            samples.attrs.update({"sample_rate": manifest["actual_sample_rate"], "channel_axis": 0,
                "time_origin_seconds": manifest.get("time_origin_seconds", 0.0),
                "acquisition_complete": complete, "calibration_applied_to_raw": False,
                "acquisition_session_id": manifest["session_id"]})
            for record in records:
                raw.seek(record["byte_offset"])
                payload = raw.read(record["byte_count"])
                if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                    raise AcquisitionError("终结期间录制块校验失败。")
                values = np.frombuffer(payload, dtype="<f8").reshape(record["frames"], count).T
                start = record["first_sample_index"]
                samples[:, start:start+record["frames"]] = values
            output.flush()
        with temporary.open("r+b") as source:
            os.fsync(source.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def recover_session(manifest_path):
    path, manifest = read_manifest(manifest_path)
    if manifest["status"] == "complete":
        raise AcquisitionError("完整会话无需部分恢复。")
    try:
        records, flags = _verified_records(path, manifest)
        destination = path.parent / "recovered.h5"
        _build_hdf5(path, manifest, records, destination, complete=False)
        manifest.update({"status": "failed", "recovery_file": destination.name,
            "recovery_sha256": file_hash(destination), "frames_written": sum(r["frames"] for r in records),
            "flags": list(dict.fromkeys([*manifest.get("flags", []), "incomplete_recording", *flags])),
            "recovered_at_utc": utc_now()})
        write_manifest(path, manifest)
        return session_from_manifest(path, manifest)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise AcquisitionError(f"恢复录制失败：{error}") from error


def open_acquisition_session(manifest_path, *, allow_partial=False):
    initial_hash = file_hash(Path(manifest_path))
    path, manifest = read_manifest(manifest_path)
    complete = manifest["status"] == "complete"
    if not complete and not allow_partial:
        raise AcquisitionError("录制未通过完整性检查；只能明确选择恢复查看，不能作为完整测量。")
    name, expected_hash = (manifest.get("data_file"), manifest.get("data_sha256")) if complete else (
        manifest.get("recovery_file"), manifest.get("recovery_sha256"))
    if not name or not expected_hash:
        raise AcquisitionError("会话没有可打开的已验证数据文件；请先恢复部分录制。")
    data_path = safe_session_file(path, name)
    if data_path.suffix.lower() == ".json":
        raise AcquisitionError("DAQ 原始数据不能引用 JSON 清单，禁止递归会话引用。")
    if not isinstance(expected_hash, str) or file_hash(data_path) != expected_hash.lower():
        raise AcquisitionError("DAQ 数据文件哈希不一致，拒绝打开。")
    signal = import_signal(data_path)
    if signal.frames != manifest["frames_written"] or signal.sample_rate != manifest["actual_sample_rate"]:
        raise AcquisitionError("DAQ 文件帧数或实际采样率与会话清单不一致。")
    recorded_channels = manifest.get("channels", [])
    if len(recorded_channels) != signal.channel_count:
        raise AcquisitionError("DAQ 通道数与会话清单不一致。")
    channels = []
    for source, config in zip(signal.channels, recorded_channels):
        if source.unit != config["unit"] or source.name != config["name"]:
            raise AcquisitionError("DAQ 通道顺序、名称或单位与会话清单不一致。")
        channels.append(replace(source, metadata={**source.metadata, "acquisition_channel": config}))
    quality = (*signal.quality, *manifest.get("flags", []), *(() if complete else ("incomplete_recording",)))
    if file_hash(path) != initial_hash:
        raise AcquisitionError("DAQ 会话清单在导入过程中发生变化，请重新导入稳定会话。")
    return SignalData(signal.samples, signal.sample_rate, tuple(channels), path, initial_hash,
        {**signal.metadata, "acquisition_session": manifest, "acquisition_manifest_path": str(path),
         "acquisition_manifest_sha256": initial_hash, "acquisition_data_path": str(data_path),
         "acquisition_data_sha256": signal.source_hash,
         "time_origin_seconds": manifest.get("time_origin_seconds", 0.0)}, tuple(dict.fromkeys(quality)))
