import multiprocessing
import sys
from pathlib import Path

if not getattr(sys, 'frozen', False):
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        from autoacoustics.app import main
        raise SystemExit(main())
    except Exception:
        import os
        import traceback
        import tempfile
        log_root = Path(os.environ.get('LOCALAPPDATA', tempfile.gettempdir())) / 'AutoAcousticsPro' / 'logs'
        if '--self-check' in sys.argv:
            log_root = Path(sys.argv[sys.argv.index('--self-check')+1]).resolve().parent
        elif '--verify-bundle' in sys.argv:
            log_root = Path(sys.argv[sys.argv.index('--verify-bundle')+1]).resolve().parent
        elif '--verify-knowledge' in sys.argv:
            log_root = Path(sys.argv[sys.argv.index('--verify-knowledge')+1]).resolve().parent
        elif '--pipeline-probe' in sys.argv:
            log_root = Path(sys.argv[sys.argv.index('--pipeline-probe')+1]).resolve().parent
        elif not getattr(sys, 'frozen', False):
            log_root = Path(__file__).resolve().parent / 'output/logs'
        log_root.mkdir(parents=True, exist_ok=True)
        log_path = log_root / 'startup.log'
        log_path.write_text(traceback.format_exc(), encoding='utf-8')
        if sys.stderr is not None:
            sys.stderr.write(traceback.format_exc())
        if not any(flag in sys.argv for flag in ('--self-check','--verify-bundle','--verify-knowledge','--pipeline-probe')):
            if sys.platform == 'win32':
                import ctypes
                ctypes.windll.user32.MessageBoxW(None, f'程序启动失败。详细信息已保存：\n{log_path}',
                                               'AutoAcoustics Pro', 0x10)
        raise SystemExit(1)
