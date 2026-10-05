"""Record the actual frozen process exit, independently of its internal report."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', '--executable', dest='executable', type=Path,
                        default=ROOT/'dist/AutoAcousticsProLive031/AutoAcousticsProLive031.exe')
    parser.add_argument('--output', '--destination', dest='destination', type=Path, required=True)
    parser.add_argument('--mode', '--workflow', dest='workflow', choices=('live', 'bundle'), default='live')
    parser.add_argument('--microphone-seconds', type=float, default=0.)
    options = parser.parse_args()
    executable = options.executable.resolve()
    destination = options.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    before = digest(executable)
    if options.workflow == 'bundle':
        request = {'fixtures_directory':str(ROOT/'tests/fixtures/imports'),
                   'corpus_directory':str(ROOT/'data/corpus'),
                   'profile':str(ROOT/'profiles/head-pa-reference.json'),
                   'output_directory':str(destination),'run_real_batch':True,'run_head':True}
        request_path = destination/'request.json'
        request_path.write_text(json.dumps(request,ensure_ascii=False,indent=2),encoding='utf-8')
        command = [str(executable),'--verify-bundle',str(request_path)]
        acceptance_receipt = destination/'bundle_validation.json'
    else:
        command = [str(executable),'--verify-live-nvh',str(destination),
                   '--microphone-seconds',str(options.microphone_seconds)]
        acceptance_receipt = destination/'live_nvh_validation.json'
    environment = dict(os.environ)
    for key in ('PYTHONHOME', 'PYTHONPATH', 'QT_QPA_PLATFORM', 'QT_PLUGIN_PATH'):
        environment.pop(key, None)
    # Do not accidentally resolve codec/Qt DLLs from the developer's PATH.
    system_root = Path(environment.get('SystemRoot', 'C:/Windows'))
    environment['PATH'] = os.pathsep.join(map(str, [system_root/'System32', system_root]))
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    previous_error_mode = None
    if os.name == 'nt':
        import ctypes
        # Only diagnostic children suppress fault dialogs; the exit remains a failure.
        previous_error_mode = ctypes.windll.kernel32.SetErrorMode(3)
    try:
        with (destination/'process.log').open('wb') as log:
            try:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         env=environment, timeout=600,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = -1
    finally:
        if previous_error_mode is not None:
            ctypes.windll.kernel32.SetErrorMode(previous_error_mode)
    receipt = {'schema_version': 1, 'started_at_utc': started_at,
               'command': command, 'exe_sha256': before,
               'exe_unchanged': before == digest(executable),
               'exit_code': code, 'elapsed_seconds': time.monotonic()-started,
               'PATH':environment['PATH'],'PYTHONHOME':False,'PYTHONPATH':False,'native_Qt_platform':True,
               'acceptance_receipt':str(acceptance_receipt),'diagnostic_Windows_fault_dialogs':False}
    (destination/'invocation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2),
                                             encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0 if code == 0 and receipt['exe_unchanged'] and acceptance_receipt.is_file() else 2


if __name__ == '__main__':
    raise SystemExit(main())
