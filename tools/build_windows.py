"""Build from the pinned project environment; codecs are part of the bundle."""
import sys
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    source_paths=sorted((ROOT/'src').rglob('*.py'))+[ROOT/'launcher.py',ROOT/'requirements.lock']
    source_hashes={path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in source_paths}
    from autoacoustics.resources import media_tool
    codec_paths = {name: media_tool(name) for name in ('ffmpeg', 'ffprobe')}
    # Codex's tools add third-party ICU/CRT DLLs to PATH. Qt needs the Windows
    # ICU API; collecting Poppler's different icuuc.dll breaks frozen QtCore.
    system_root = Path(os.environ.get('SystemRoot', os.environ.get('SYSTEMROOT', 'C:/Windows')))
    os.environ['PATH'] = os.pathsep.join(map(str, [Path(sys.executable).parent,
        Path(sys.base_prefix), system_root / 'System32', system_root]))
    from PyInstaller.__main__ import run
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--name', help='Alternate bundle name, allowing an existing application to remain open.')
    parser.add_argument('--snapshot', type=Path, default=ROOT/'output/build_source_snapshot.json')
    options = parser.parse_args()
    name = options.name or ('AutoAcousticsProbe' if options.probe else 'AutoAcousticsPro')
    if not re.fullmatch(r'AutoAcoustics[A-Za-z0-9_]+',name):
        parser.error('--name must start with AutoAcoustics and contain only ASCII letters, numbers or underscores.')
    args = ['--noconfirm', '--clean', '--onedir', '--console' if options.probe or options.diagnostic else '--windowed', '--name', name,
            '--paths', str(ROOT / 'src'), '--collect-all', 'mosqito',
            '--hidden-import', 'numpy', '--hidden-import', 'scipy.signal',
            '--hidden-import', 'soundfile', '--collect-all', 'sounddevice', '--copy-metadata', 'PyQt6',
            '--collect-all', 'llvmlite', '--copy-metadata', 'numba',
            '--collect-submodules', 'nidaqmx', '--recursive-copy-metadata', 'nidaqmx',
            '--copy-metadata', 'mosqito', '--copy-metadata', 'scipy',
            '--copy-metadata', 'numpy', '--copy-metadata', 'soundfile']
    for codec_name in ('ffmpeg', 'ffprobe'):
        args += ['--add-binary', str(codec_paths[codec_name]) + ';bin']
    if options.probe:
        args += ['--exclude-module', 'autoacoustics.ui']
    for folder in ('profiles', 'docs/validation', 'licenses'):
        if (ROOT / folder).exists():
            args += ['--add-data', str(ROOT / folder) + ';' + folder]
    args += [str(ROOT / 'launcher.py')]
    run(args)
    if any(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected
           for path,expected in source_hashes.items()):
        raise RuntimeError('Product source changed during the frozen build.')
    snapshot={'schema_version':1,'source_unchanged_during_build':True,'product_source_sha256':source_hashes,
              'exe_sha256':hashlib.sha256((ROOT/'dist'/name/(name+'.exe')).read_bytes()).hexdigest()}
    options.snapshot.parent.mkdir(parents=True,exist_ok=True)
    options.snapshot.write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
    # Release inventories describe the finished bundle and are generated last.
    # Do not ship stale inventories collected from the previous build as data.
    for filename in ('release_manifest.json', 'release_zip.json'):
        (ROOT / 'dist' / name / '_internal' / 'docs' / 'validation' / filename).unlink(missing_ok=True)
    from collect_distribution_notices import main as collect_notices
    # Inventory the final binaries, rather than shipping a previous build's hashes.
    if collect_notices(['--destination',str(ROOT/'dist'/name),
                       '--bundle',str(ROOT/'dist'/name),'--strict']):
        raise RuntimeError('发行目录的第三方依赖库存不完整。')
    readme=(ROOT/'README.md').read_text(encoding='utf-8')
    (ROOT/'dist'/name/'README.md').write_text(readme.replace('(docs/validation/',
        '(_internal/docs/validation/'),encoding='utf-8')


if __name__ == '__main__':
    main()
