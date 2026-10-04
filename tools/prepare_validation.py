"""Safely materialize the checked local corpus for executable/UI verification."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def prepare():
    manifest=json.loads((ROOT/'docs/validation/corpus_manifest.json').read_text(encoding='utf-8'))
    archive_path=ROOT/'压缩.zip'
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest()==manifest['archive']['sha256']
    target=(ROOT/'data/corpus').resolve();target.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in manifest['members']:
            path=(target/member['path']).resolve()
            if not path.is_relative_to(target):raise ValueError('Corpus path escapes its target')
            content=archive.read(member['path'])
            assert hashlib.sha256(content).hexdigest()==member['sha256']
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(content)
    request={'fixtures_directory':str(ROOT/'tests/fixtures/imports'),
             'corpus_directory':str(target),'profile':str(ROOT/'profiles/head-pa-reference.json'),
             'output_directory':str(ROOT/'output/bundle-validation'),'run_real_batch':True}
    destination=ROOT/'output/bundle_request.json';destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(request,ensure_ascii=False,indent=2),encoding='utf-8')
    print(destination)

if __name__=='__main__':prepare()
