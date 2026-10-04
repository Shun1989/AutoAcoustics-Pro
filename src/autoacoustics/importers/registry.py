from __future__ import annotations

from pathlib import Path

from autoacoustics.model import ImportError, ImportMapping
from .common import file_hash, validate_mapping


SUPPORTED_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".mp4", ".avi", ".mkv",
                      ".csv", ".txt", ".dat", ".mat", ".h5", ".hdf5", ".hdf", ".tdms", ".json")


def import_signal(path: Path, mapping: ImportMapping | None = None):
    path = Path(path)
    validate_mapping(mapping)
    if not path.is_file():
        raise ImportError(f"文件不存在或不可读取：{path}")
    suffix = path.suffix.lower()
    if suffix == ".zip":
        raise ImportError("请先解压 ZIP，再导入其中的音频文件。")
    try:
        with path.open("rb") as stream:
            prefix = stream.read(16384)
        from .head import MAX_FILE_BYTES, is_head_signature
        import h5py
        is_generic_hdf5 = h5py.is_hdf5(path)
        is_head = is_head_signature(prefix) and not is_generic_hdf5
        if is_head and path.stat().st_size > MAX_FILE_BYTES:
            raise ImportError(f"HEAD HDF：文件超过当前导入资源上限 {MAX_FILE_BYTES} 字节。")
        if not is_head and not is_generic_hdf5 and suffix not in SUPPORTED_SUFFIXES:
            raise ImportError(f"暂不支持此文件格式：{suffix or '(无扩展名)'}")
        original_hash = file_hash(path)
        if suffix == ".json" and not is_head and not is_generic_hdf5:
            if mapping is not None:
                raise ImportError("DAQ 会话的通道、采样率和单位已记录，不能使用额外导入映射重标。")
            from ..acquisition.device import AcquisitionError
            from ..acquisition.storage import open_acquisition_session
            try:
                result = open_acquisition_session(path)
            except (AcquisitionError, KeyError, TypeError, ValueError) as error:
                raise ImportError(f"DAQ 会话导入失败：{error}") from error
        elif is_head:
            from .head import import_head
            result = import_head(path, mapping)
        elif is_generic_hdf5:
            from .scientific import import_scientific
            result = import_scientific(path, mapping)
        elif suffix in {".wav", ".flac"}:
            from .audio import import_audio
            result = import_audio(path, mapping)
        elif suffix in {".mp3", ".m4a", ".aac", ".ogg", ".mp4", ".avi", ".mkv"}:
            from .media import import_media
            result = import_media(path, mapping)
        elif suffix in {".csv", ".txt", ".dat"}:
            from .tabular import import_tabular
            result = import_tabular(path, mapping)
        else:
            from .scientific import import_scientific
            result = import_scientific(path, mapping)
        if result.source_hash != original_hash or file_hash(path) != original_hash:
            raise ImportError("源文件在导入过程中发生变化，请重新导入稳定的文件。")
        return result
    except ImportError:
        raise
    except OSError as error:
        raise ImportError(f"读取文件失败：{error}") from error
