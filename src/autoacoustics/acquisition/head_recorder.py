"""HEAD C2 stable-file handoff; C3 stays unavailable without a confirmed SDK."""
from __future__ import annotations

import math
import os
from pathlib import Path
import time
from uuid import uuid4

from autoacoustics import __version__
from autoacoustics.importers.common import file_hash
from autoacoustics.importers.registry import import_signal
from autoacoustics.model import to_plain
from .device import AcquisitionError, DiscoveryResult
from .storage import session_from_manifest, utc_now, write_manifest


def _copy_bytes(source, destination):
    with Path(source).open("rb") as incoming, Path(destination).open("xb") as output:
        while chunk := incoming.read(1024 * 1024):
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())


def copy_completed_recording(source, directory, *, device_stopped, device_info=None, operator="",
                             measurement_context=None, events=(), stability_checks=3, stability_interval=0.2,
                             is_simulated=False):
    """Copy one user-selected stopped recording; never scan drives or control HEAD.

    Stable stat observations are a fallback in addition to explicit operator
    confirmation and complete format parsing, not a claim about a device API.
    """
    if not device_stopped:
        raise AcquisitionError("请先在设备或官方 Recorder 中停止录制，再确认文件交接。")
    if (not isinstance(stability_checks, int) or not 2 <= stability_checks <= 10
            or not math.isfinite(stability_interval) or not 0 <= stability_interval <= 1):
        raise AcquisitionError("文件稳定性检查次数或间隔无效。")
    source = Path(source).resolve()
    if not source.is_file():
        raise AcquisitionError("所选设备录制文件不存在或不可读取。")
    initial = source.stat()
    evidence = []
    for index in range(stability_checks):
        if index:
            time.sleep(stability_interval)
        stat = source.stat()
        evidence.append({"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        if stat.st_size != initial.st_size or stat.st_mtime_ns != initial.st_mtime_ns:
            raise AcquisitionError("设备文件尚未稳定，可能仍在写入或复制；请等待完成后重试。")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise AcquisitionError("交接目录必须为空，不能覆盖已有测量。")
    destination = directory / source.name
    temporary = directory / f".{source.stem}.{uuid4().hex}.incoming{source.suffix}"
    manifest_path = directory / "session.json"
    manifest = {"schema_version": 1, "session_id": uuid4().hex, "status": "recording",
        "backend": "HEAD_FILE_HANDOFF", "is_simulated": bool(is_simulated), "software_version": __version__,
        "controlled_recording": False, "supports_realtime_samples": False,
        "device": to_plain(device_info or {"model": "unknown", "firmware": "unknown"}),
        "operator": operator, "operator_confirmed_device_stopped": True, "started_at_utc": utc_now(),
        "ended_at_utc": None, "source_file": str(source), "source_size_bytes": initial.st_size,
        "source_mtime_ns": initial.st_mtime_ns, "file_stability_observations": evidence,
        "data_file": None, "frames_expected": None, "frames_read": 0, "frames_written": 0,
        "flags": [], "errors": [], "events": to_plain(events),
        "measurement_context": to_plain(measurement_context or {}), "calibration_applied_to_raw": False}
    write_manifest(manifest_path, manifest)
    try:
        original_hash = file_hash(source)
        _copy_bytes(source, temporary)
        if file_hash(temporary) != original_hash or file_hash(source) != original_hash:
            raise AcquisitionError("设备文件交接期间发生变化，哈希校验失败。")
        stat = source.stat()
        if stat.st_size != initial.st_size or stat.st_mtime_ns != initial.st_mtime_ns:
            raise AcquisitionError("设备文件交接期间发生变化。")
        try:
            signal = import_signal(temporary)
        except Exception as error:
            raise AcquisitionError(f"设备文件未通过完整格式与载荷检查：{error}") from error
        manifest.update({"status": "finalizing", "actual_sample_rate": signal.sample_rate,
            "requested_sample_rate": None, "time_origin_seconds": signal.time_origin,
            "frames_expected": signal.frames, "frames_read": signal.frames, "frames_written": signal.frames,
            "channels": to_plain(signal.channels), "source_sha256": original_hash, "files_received": 1,
            "source_format": signal.metadata.get("format"), "source_metadata": to_plain(signal.metadata)})
        write_manifest(manifest_path, manifest)
        # Windows rename is atomic and refuses an existing destination, and
        # also works on FAT/exFAT work disks that cannot create hard links.
        if os.name == "nt":
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
            temporary.unlink()
        manifest.update({"status": "complete", "data_file": destination.name,
                         "data_sha256": original_hash, "ended_at_utc": utc_now()})
        write_manifest(manifest_path, manifest)
        return session_from_manifest(manifest_path, manifest)
    except Exception as error:
        manifest.update({"status": "failed", "flags": ["file_handoff_failed"], "errors": [str(error)],
                         "ended_at_utc": utc_now(), "partial_file": temporary.name if temporary.exists() else None})
        write_manifest(manifest_path, manifest)
        if isinstance(error, AcquisitionError):
            raise
        raise AcquisitionError(f"HEAD 设备文件交接失败：{error}") from error


class HeadManagedRecorder:
    supports_realtime_samples = False
    unavailable_reason = ("HEAD 官方受控录制不可用：尚未取得并验证此 SQuadriga III 的固件/选件、"
                          "ArtemiS Recorder/ASX 04 安装许可、版本对应的官方 SDK/API 文档与控制示例。"
                          "当前仅提供停止后的设备文件交接；没有实时原始样本接口证据。")

    def discover(self):
        return DiscoveryResult(False, reason=self.unavailable_reason)

    def configure(self, config=None):
        raise AcquisitionError(self.unavailable_reason, flag="official_head_api_unavailable")

    def start(self):
        raise AcquisitionError(self.unavailable_reason, flag="official_head_api_unavailable")

    def stop(self):
        raise AcquisitionError(self.unavailable_reason, flag="official_head_api_unavailable")

    def completed_files(self):
        raise AcquisitionError(self.unavailable_reason, flag="official_head_api_unavailable")
