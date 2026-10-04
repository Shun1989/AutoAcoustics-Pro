"""Locate bundled resources; a frozen distribution never depends on PATH codecs."""
import shutil
import sys
from pathlib import Path

from .model import ImportError


def resource_root() -> Path:
    return Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[2]))


def media_tool(name: str) -> Path:
    if name not in {'ffmpeg', 'ffprobe'}:
        raise ImportError('未知媒体工具。')
    bundled = resource_root() / 'bin' / (name + '.exe')
    if bundled.is_file():
        return bundled
    if not getattr(sys, 'frozen', False):
        executable = shutil.which(name)
        if executable:
            return Path(executable)
    raise ImportError(f'交付包缺少 {name}；请使用包含完整依赖的发布目录。')
