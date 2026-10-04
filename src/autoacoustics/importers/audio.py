"""PCM/float WAV and FLAC without mixing, resampling or automatic gain."""
from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import soundfile as sf

from autoacoustics.model import ChannelInfo, ImportError, ImportMapping
from .common import make_signal


def _check_riff_bounds(path: Path):
    # libsndfile tolerates a truncated data chunk by shortening it. Reject that
    # condition before decoding so a damaged recording is never silently whole.
    size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) < 12 or header[:4] not in (b"RIFF", b"RIFX", b"RF64") or header[8:12] != b"WAVE":
            raise ImportError("WAV 文件头损坏或不是 WAV 容器。")
        rf64_data_size, rf64_frames, rf64_sizes = None, None, {}
        if header[:4] == b"RF64":
            ds_header = stream.read(8)
            if len(ds_header) != 8 or ds_header[:4] != b"ds64":
                raise ImportError("RF64 缺少必需的 ds64 长度块。")
            ds_size = struct.unpack("<I", ds_header[4:])[0]
            if ds_size < 28 or stream.tell() + ds_size > size:
                raise ImportError("RF64 ds64 长度块已截断。")
            block = stream.read(ds_size)
            riff_size, rf64_data_size, rf64_frames, table_count = struct.unpack("<QQQI", block[:28])
            if ds_size < 28 + 12 * table_count:
                raise ImportError("RF64 ds64 子块长度表已截断。")
            for index in range(table_count):
                entry = block[28 + index * 12:40 + index * 12]
                rf64_sizes.setdefault(entry[:4], []).append(struct.unpack("<Q", entry[4:])[0])
            stream.seek(stream.tell() + (ds_size & 1))
            riff_end = riff_size + 8
        else:
            riff_end = struct.unpack(">I" if header[:4] == b"RIFX" else "<I", header[4:8])[0] + 8
        endian = ">" if header[:4] == b"RIFX" else "<"
        if riff_end > size:
            raise ImportError("WAV 文件已截断：RIFF 声明的长度超过文件实际长度。")
        while stream.tell() < riff_end:
            chunk = stream.read(8)
            if len(chunk) != 8:
                raise ImportError("WAV 子块文件头已截断。")
            length = struct.unpack(endian + "I", chunk[4:])[0]
            if length == 0xFFFFFFFF:
                if chunk[:4] == b"data" and rf64_data_size is not None:
                    length = rf64_data_size
                elif rf64_sizes.get(chunk[:4]):
                    length = rf64_sizes[chunk[:4]].pop(0)
                else:
                    raise ImportError("RF64 子块缺少明确的 64-bit 长度。")
            end = stream.tell() + length
            if end > riff_end or end > size:
                raise ImportError("WAV 数据子块已截断。")
            stream.seek(end + (length & 1))
        return rf64_frames


def import_audio(path: Path, mapping: ImportMapping | None = None):
    rf64_frames = None
    if path.suffix.lower() == ".wav":
        rf64_frames = _check_riff_bounds(path)
    try:
        with sf.SoundFile(path, "r") as stream:
            fs, subtype, format_name, declared_frames = (stream.samplerate, stream.subtype, stream.format, len(stream))
            samples = stream.read(dtype="float64", always_2d=True).T
    except (sf.LibsndfileError, RuntimeError, OSError) as error:
        raise ImportError(f"无法读取无损音频：{error}") from error
    if samples.shape[1] != declared_frames:
        raise ImportError("音频实际帧数与文件声明不一致。")
    if rf64_frames and declared_frames != rf64_frames:
        raise ImportError("RF64 实际帧数与 ds64 声明不一致。")
    pcm_bits = {"PCM_U8": 8, "PCM_S8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32}.get(subtype)
    floating = subtype in {"FLOAT", "DOUBLE"}
    lossy = pcm_bits is None and not floating
    quality = []
    if lossy:
        quality.append("lossy_source")
    if pcm_bits is not None and samples.size and (
        np.any(samples <= -1.0) or np.any(samples >= 1.0 - 2.0 ** (1 - pcm_bits))
    ):
        quality.append("possible_clipping")
    if floating:
        quality.append("floating_full_scale_unknown")
    # A sound file represents a digital amplitude. A physical unit declaration
    # requires a scientific file or calibration, rather than relabelling PCM.
    if mapping and mapping.unit not in {"unknown", "FS"}:
        raise ImportError("音频导入保留 FS 数字量；请在校准步骤配置物理量换算。")
    channels = tuple(ChannelInfo(f"Channel {index + 1}", "FS", "digital_amplitude", False,
                                 1.0 if pcm_bits else None,
                                 metadata={"source_subtype": subtype, "bit_depth": pcm_bits})
                     for index in range(samples.shape[0]))
    return make_signal(path, samples, fs, channels, mapping=mapping, quality=quality,
                       metadata={"format": format_name, "subtype": subtype, "bit_depth": pcm_bits,
                                 "lossy": lossy, "source_dtype": "float" if floating else "integer",
                                 "normalization": "libsndfile PCM / 2^(bits-1)" if pcm_bits else "native floating values",
                                 "time_origin_seconds": 0.0})
