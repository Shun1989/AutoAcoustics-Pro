"""Package and verify every byte of the already validated desktop directory."""
import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(stream):
    result = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, default=ROOT / 'dist/AutoAcousticsPro')
    parser.add_argument('--destination', type=Path,
                        default=ROOT / 'AutoAcousticsPro-0.1.0-Windows.zip')
    args = parser.parse_args()
    bundle, destination = args.bundle.resolve(), args.destination.resolve()
    manifest_path = bundle / 'release_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = {item['path']: item for item in manifest['files']}
    with manifest_path.open('rb') as stream:
        manifest_sha = digest(stream)
    expected['release_manifest.json'] = {
        'bytes': manifest_path.stat().st_size, 'sha256': manifest_sha}
    members = {p.relative_to(bundle).as_posix(): p
               for p in bundle.rglob('*') if p.is_file()}
    if set(members) != set(expected):
        raise RuntimeError('Bundle files changed after the release manifest was recorded.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.part')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1) as archive:
            for name, path in sorted(members.items()):
                archive.write(path, f'{bundle.name}/{name}')
        with zipfile.ZipFile(temporary) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != {
                    f'{bundle.name}/{name}' for name in expected}:
                raise RuntimeError('Portable archive has missing, unexpected or duplicate entries.')
            for name, item in expected.items():
                full_name = f'{bundle.name}/{name}'
                if archive.getinfo(full_name).file_size != item['bytes']:
                    raise RuntimeError(f'Archive size mismatch: {name}')
                # Reading to EOF also checks each entry's ZIP CRC.
                with archive.open(full_name) as stream:
                    if digest(stream) != item['sha256']:
                        raise RuntimeError(f'Archive hash mismatch: {name}')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    with destination.open('rb') as stream:
        archive_sha = digest(stream)
    proof = {
        'schema_version': 1, 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'path': str(destination), 'size_bytes': destination.stat().st_size,
        'sha256': archive_sha, 'exe_sha256': manifest['exe_sha256'],
        'release_manifest_sha256': manifest_sha,
        'entries': len(expected), 'all_entry_sizes_hashes_and_crc_passed': True,
        'scope': 'Archive integrity; not a second-machine or first-time-user test.'}
    (ROOT / 'docs/validation/release_zip.json').write_text(
        json.dumps(proof, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(proof, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
