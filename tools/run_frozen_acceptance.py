"""Run an explicit native acceptance entry and bind its exit to the EXE bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe',type=Path,default=ROOT/'dist/AutoAcousticsProLive/AutoAcousticsProLive.exe')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('live','bundle'),required=True)
    parser.add_argument('--microphone-seconds',type=float,default=0.)
    args = parser.parse_args()
    executable = args.exe.resolve()
    output = args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    environment = os.environ.copy()
    environment['PATH'] = r'C:\Windows\System32;C:\Windows'
    for key in ('PYTHONHOME','PYTHONPATH','QT_QPA_PLATFORM','QT_PLUGIN_PATH'):
        environment.pop(key,None)
    if args.mode == 'bundle':
        request = {'fixtures_directory':str(ROOT/'tests/fixtures/imports'),
                   'corpus_directory':str(ROOT/'data/corpus'),
                   'profile':str(ROOT/'profiles/head-pa-reference.json'),
                   'output_directory':str(output),'run_real_batch':True,'run_head':True}
        request_path = output/'request.json'
        request_path.write_text(json.dumps(request,ensure_ascii=False,indent=2),encoding='utf-8')
        command = [str(executable),'--verify-bundle',str(request_path)]
        receipt = output/'bundle_validation.json'
    else:
        command = [str(executable),'--verify-live-nvh',str(output),'--microphone-seconds',str(args.microphone_seconds)]
        receipt = output/'live_nvh_validation.json'
    started = time.monotonic()
    # Diagnostic children report native crashes by exit code instead of
    # repeatedly opening Windows fault boxes on the user's desktop.
    previous_error_mode = None
    if os.name == 'nt':
        import ctypes
        previous_error_mode = ctypes.windll.kernel32.SetErrorMode(3)
    try:
        completed = subprocess.run(command,cwd=ROOT,env=environment,timeout=600)
    finally:
        if previous_error_mode is not None:
            ctypes.windll.kernel32.SetErrorMode(previous_error_mode)
    unchanged = hashlib.sha256(executable.read_bytes()).hexdigest() == digest
    record = {'schema_version':1,'exe_sha256':digest,'exe_unchanged':unchanged,
        'exit_code':completed.returncode,'elapsed_seconds':time.monotonic()-started,
        'PATH':environment['PATH'],'PYTHONHOME':False,'PYTHONPATH':False,'native_Qt_platform':True,
        'command':command,'acceptance_receipt':str(receipt),
        'diagnostic_Windows_fault_dialogs':False}
    (output/'invocation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False),flush=True)
    if completed.returncode or not unchanged or not receipt.is_file():raise SystemExit(2)


if __name__ == '__main__':main()
