"""Audio-stream selection and native-rate floating decode through FFmpeg."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from autoacoustics.model import ChannelInfo, ImportError, ImportMapping, MappingRequired
from .common import make_signal


def _tool(name: str) -> Path:
    try:
        from autoacoustics.resources import media_tool
    except ModuleNotFoundError:
        if getattr(sys, "frozen", False):
            raise ImportError(f"发布包缺少内置 {name} 资源。请修复安装。")
        location = shutil.which(name)
        if not location:
            raise ImportError(f"开发环境缺少 {name}；发布包必须内置此依赖。")
        return Path(location)
    return media_tool(name)


def _run(command: list[str], *, timeout: int):
    try:
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
                                 creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    except FileNotFoundError as error:
        raise ImportError("媒体解码工具不存在。请修复安装。") from error
    except subprocess.TimeoutExpired as error:
        raise ImportError("媒体探测或解码超时；未生成有效导入结果。") from error
    if process.returncode:
        detail = process.stderr.decode("utf-8", errors="replace").strip()[-2000:]
        raise ImportError(f"媒体解码失败：{detail}")
    return process.stdout


def _number(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def probe_media(path: Path) -> dict:
    data = _run([str(_tool("ffprobe")), "-v", "error", "-show_streams", "-show_format",
                 "-of", "json", str(path)], timeout=30)
    try:
        return json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeError) as error:
        raise ImportError("媒体探测返回了无效的格式信息。") from error


def import_media(path: Path, mapping: ImportMapping | None = None):
    probe = probe_media(path)
    streams = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "audio"]
    if not streams:
        raise ImportError("文件没有音轨，无法进行声学分析。")
    options = {"audio_streams": [{"index": stream["index"], "codec": stream.get("codec_name"),
                                  "sample_rate": stream.get("sample_rate"), "channels": stream.get("channels"),
                                  "title": stream.get("tags", {}).get("title", "")}
                                 for stream in streams]}
    requested = mapping.audio_stream_index if mapping else None
    if requested is None:
        if len(streams) > 1:
            raise MappingRequired("文件含多条音轨，请明确选择 ffprobe 的 stream index。", options)
        stream = streams[0]
    else:
        matches = [stream for stream in streams if stream["index"] == requested]
        if not matches or isinstance(requested, bool):
            raise ImportError("所选音轨 index 不存在或不是音频。")
        stream = matches[0]
    fs, channel_count = _number(stream.get("sample_rate")), stream.get("channels")
    if fs is None or fs <= 0 or not isinstance(channel_count, int) or channel_count <= 0:
        raise ImportError("音轨没有可靠的采样率或通道数。")
    index = stream["index"]
    if mapping and mapping.unit not in {"unknown", "FS"}:
        raise ImportError("媒体解码保留 FS 数字量；请在校准步骤配置物理量换算。")
    payload = _run([str(_tool("ffmpeg")), "-hide_banner", "-loglevel", "error", "-nostdin", "-xerror",
                    "-i", str(path), "-map", f"0:{index}", "-vn", "-sn", "-dn",
                    "-c:a", "pcm_f64le", "-f", "f64le", "pipe:1"], timeout=600)
    if not payload or len(payload) % (8 * channel_count):
        raise ImportError("媒体解码输出为空或不含完整音频帧。")
    samples = np.frombuffer(payload, dtype="<f8").reshape(-1, channel_count).T
    codec = stream.get("codec_name", "unknown")
    lossy = not (codec.startswith("pcm_") or codec in {"flac", "alac", "wavpack"})
    padding = {"initial_padding": stream.get("initial_padding"),
               "trailing_padding": stream.get("trailing_padding"), "first_packet_side_data": None,
               "availability": "not_reported"}
    try:
        raw = _run([str(_tool("ffprobe")), "-v", "error", "-select_streams", str(index),
                    "-read_intervals", "%+#1", "-show_packets", "-of", "json", str(path)], timeout=30)
        packets = json.loads(raw.decode("utf-8")).get("packets", [])
        if packets:
            padding["first_packet_side_data"] = packets[0].get("side_data_list")
        if any(padding[key] is not None for key in ("initial_padding", "trailing_padding", "first_packet_side_data")):
            padding["availability"] = "reported_by_ffprobe"
    except (ImportError, ValueError, UnicodeError):
        padding["availability"] = "padding_probe_unavailable"
    channels = tuple(ChannelInfo(f"Channel {number + 1}", "FS", "digital_amplitude", full_scale=1.0,
                                 metadata={"codec": codec, "source_audio_stream_index": index})
                     for number in range(channel_count))
    quality = ["lossy_source"] if lossy else []
    if np.any(np.abs(samples) > 1.0):
        quality.append("decoded_amplitude_exceeds_nominal_fs")
    return make_signal(path, samples, fs, channels, mapping=mapping, quality=quality,
                       metadata={"format": probe.get("format", {}).get("format_name", path.suffix.lower()),
                                 "codec": codec, "lossy": lossy, "audio_stream_index": index,
                                 "source_stream": stream,
                                 "time_origin_seconds": _number(stream.get("start_time")),
                                 "container_time_origin_seconds": _number(probe.get("format", {}).get("start_time")),
                                 "decoded_time_basis": "sample_index_from_decoded_frame_zero",
                                 "encoder_padding": padding, "normalization": "FFmpeg native-rate pcm_f64le digital samples"})
