"""Desktop bootstrap and distribution diagnostics."""
import argparse
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow

from .resources import media_tool


def self_check(app, destination: Path):
    import numpy as np
    import scipy
    import soundfile
    from mosqito.sq_metrics import loudness_zwtv
    from .analysis._loudness_engine import compute_loudness

    started = time.perf_counter()
    x = np.sqrt(2) * np.sin(2 * np.pi * 1000 * np.arange(48000) / 48000)
    total, specific, bark, times = loudness_zwtv(x, 48000)
    compiled = compute_loudness(x, 48000)
    result = {'schema_version': 1, 'frozen': bool(getattr(sys, 'frozen', False)),
              'platform': platform.platform(), 'python': platform.python_version(),
              'versions': {name: version(name) for name in ('PyQt6', 'mosqito', 'scipy', 'numpy', 'soundfile')},
              'loudness_samples': len(total), 'loudness_finite': bool(np.isfinite(total).all()),
              'loudness_max_sone': float(np.max(total)), 'elapsed_s': time.perf_counter() - started}
    result['optimized_backend'] = compiled['metadata']
    result['optimized_backend_total_max_error'] = float(np.max(np.abs(compiled['total'] - total)))
    result['optimized_backend_specific_max_error'] = float(np.max(np.abs(compiled['specific'] - specific)))
    for name in ('ffmpeg', 'ffprobe'):
        tool = media_tool(name)
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        output = subprocess.run([str(tool), '-version'], capture_output=True, text=True,
                                check=True, creationflags=flags)
        result[name] = {'path': str(tool), 'version': output.stdout.splitlines()[0]}
    window = QMainWindow()
    window.setWindowTitle('AutoAcoustics Pro — 构建验证')
    window.setCentralWidget(QLabel('桌面与声学依赖验证通过'))
    window.resize(680, 320)
    window.show()
    app.processEvents()
    destination.parent.mkdir(parents=True, exist_ok=True)
    image_path = destination.with_suffix('.png')
    if not window.grab().save(str(image_path)):
        raise RuntimeError('Qt 窗口截图保存失败。')
    result['qt_window_image'] = str(image_path)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    window.close()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-check', type=Path)
    parser.add_argument('--verify-bundle', type=Path)
    parser.add_argument('--verify-knowledge', type=Path)
    parser.add_argument('--pipeline-probe', type=Path)
    parser.add_argument('--project', type=Path, help='Open a saved measurement project on startup.')
    parser.add_argument('--verify-live-nvh',type=Path)
    parser.add_argument('--microphone-seconds',type=float,default=0.)
    args = parser.parse_args(argv)
    app = QApplication(sys.argv[:1])
    app.setApplicationName('AutoAcoustics Pro')
    app.setOrganizationName('AutoAcoustics')
    try:
        return _run(app, args)
    finally:
        from .qt_lifecycle import dispose_gui
        dispose_gui(app)


def _run(app, args):
    if args.verify_live_nvh:
        from .live_validation import verify_live_nvh
        return verify_live_nvh(app,args.verify_live_nvh,args.microphone_seconds)
    if args.self_check:
        return self_check(app, args.self_check)
    if args.verify_bundle:
        from .validation import verify_bundle
        return verify_bundle(app,args.verify_bundle)
    if args.verify_knowledge:
        from .knowledge_validation import verify_knowledge
        return verify_knowledge(app, args.verify_knowledge)
    if args.pipeline_probe:
        from .resource_probe import run_probe
        run_probe(args.pipeline_probe)
        return 0
    from .ui.main_window import MainWindow
    window = MainWindow()
    if args.project:
        window.open_project_path(args.project)
    else:
        window.open_soundcard()
    window.show()
    return app.exec()
