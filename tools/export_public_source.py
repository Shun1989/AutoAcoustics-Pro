"""Export an explicit public source set, without local Git history or datasets.

Run from the development workspace before committing in public/AutoAcoustics-Pro.
In the exported repository, --verify checks the committed byte manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'PUBLIC_SOURCE_MANIFEST.json'
ROOT_FILES = ('LICENSE', 'NOTICE', 'CONTRIBUTING.md', 'pyproject.toml',
              'requirements.lock', 'launcher.py', 'REWRITE_PRODUCT_SPEC.md')
SECRET_PATTERNS = {
    'github_token': r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{35,})\b',
    'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'provider_key': r'\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{30,}\b',
    'personal_windows_path': r'(?i)C:[/\\]+Users[/\\]+guoyu\b',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan(paths):
    findings = []
    for path in paths:
        if path.suffix.lower() not in {'.py', '.md', '.json', '.toml', '.yml', '.txt', ''}:
            continue
        try:
            body = path.read_text(encoding='utf-8')
        except UnicodeError:
            continue
        for name, pattern in SECRET_PATTERNS.items():
            if re.search(pattern, body):
                findings.append({'file': path.name, 'rule': name})
    if findings:
        # Report locations/rules only, never the matched credential text.
        raise ValueError('Public export review required: ' + json.dumps(findings))


def selected_files():
    selected = {name: ROOT / name for name in ROOT_FILES}
    for directory in ('src', 'tools', 'tests'):
        for path in (ROOT / directory).rglob('*.py'):
            relative = path.relative_to(ROOT).as_posix()
            if '__pycache__' in path.parts or relative.startswith('tests/reference/upstream/'):
                continue
            selected[relative] = path
    # Only the generator's known synthetic outputs, never arbitrary files added
    # to the fixture directory by later experiments.
    fixture_dir = ROOT / 'tests/fixtures/imports'
    fixture = json.loads((fixture_dir / 'fixture_manifest.json').read_text(encoding='utf-8'))
    for name in ['fixture_manifest.json', *(case['path'] for case in fixture['cases'])]:
        if Path(name).name != name:
            raise ValueError('Fixture manifest path must be a basename.')
        selected['tests/fixtures/imports/' + name] = fixture_dir / name
    for path in (ROOT / 'licenses/source').glob('*.txt'):
        selected[path.relative_to(ROOT).as_posix()] = path
    selected['tests/reference/upstream/LICENSE.mosqito.txt'] = ROOT / 'tests/reference/upstream/LICENSE.mosqito.txt'
    for name in ('README.md', 'ARCHITECTURE.md', 'THIRD_PARTY_NOTICES.md', 'VALIDATION.md', 'ci.yml'):
        selected['docs/open_source/' + name] = ROOT / 'docs/open_source' / name
    selected['README.md'] = ROOT / 'docs/open_source/README.md'
    selected['THIRD_PARTY_NOTICES.md'] = ROOT / 'docs/open_source/THIRD_PARTY_NOTICES.md'
    selected['.github/workflows/windows.yml'] = ROOT / 'docs/open_source/ci.yml'
    return selected


def verify(root):
    manifest = json.loads((root / MANIFEST).read_text(encoding='utf-8'))
    for record in manifest['files']:
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts or '.git' in relative.parts:
            raise ValueError('Invalid public manifest path.')
        path = root / relative
        if not path.is_file() or path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError('Public source hash mismatch: ' + record['path'])
    scan([root / record['path'] for record in manifest['files']])
    return {'passed': True, 'files': len(manifest['files']),
            'bytes': sum(item['bytes'] for item in manifest['files']),
            'manifest_sha256': digest(root / MANIFEST)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT / 'public/AutoAcoustics-Pro')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(ROOT), indent=2))
        return
    destination = args.destination.resolve()
    allowed_root = (ROOT / 'public').resolve()
    if not destination.is_relative_to(allowed_root) or destination == allowed_root:
        raise ValueError('Public export destination must be a child directory of the workspace public folder.')
    selected = selected_files()
    scan(list(selected.values()))
    destination.mkdir(parents=True, exist_ok=True)
    for relative, source in sorted(selected.items()):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (destination / '.gitattributes').write_text('* -text\n', encoding='utf-8')
    (destination / '.gitignore').write_text(
        '.venv/\n__pycache__/\n*.pyc\n*.egg-info/\n.pytest_cache/\nbuild/\ndist/\n'
        'output/\nartifacts/\nreviews/\nrecordings/\ndata/\nprofiles/\npublic/\n'
        '.env\n.env.*\n*.sqlite\n*.sqlite3\n*.dmp\n*.log\n*.spec\n*.zip\n'
        'docs/knowledge_reviews/\ndocs/validation/\n'
        'tests/reference/upstream/*\n!tests/reference/upstream/LICENSE.mosqito.txt\n',
        encoding='utf-8')
    relative_names = sorted([*selected, '.gitattributes', '.gitignore'])
    manifest = {'schema_version': 1, 'version': '0.3.0', 'license': 'GPL-3.0-only',
                'scope': 'Public source snapshot; original local Git history is not exported.',
                'excluded': ['real audio', 'private calibration profiles', 'customer reports',
                             'engineering PDF/text/page images', 'knowledge databases',
                             'external reference datasets', 'vendor binaries and SDKs',
                             'local Git history'],
                'files': [{'path': name, 'bytes': (destination / name).stat().st_size,
                           'sha256': digest(destination / name)} for name in relative_names]}
    (destination / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(verify(destination), indent=2))


if __name__ == '__main__':
    main()
