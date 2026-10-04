"""Record an explicitly completed page review, bound to source/text/image hashes.

This command must be run only after the named pages were actually read/viewed.
It checks preserved bytes; it cannot certify understanding or perfect extraction.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('document')
    parser.add_argument('first', type=int)
    parser.add_argument('last', type=int)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--reviewer', required=True)
    parser.add_argument('--notes', required=True)
    parser.add_argument('--text', action='store_true')
    parser.add_argument('--visual', action='store_true')
    args = parser.parse_args()
    if not (args.text or args.visual):raise ValueError('No completed review declared.')
    manifest = json.loads((ROOT / 'output/knowledge_corpus/source_manifest.json').read_text(encoding='utf-8'))
    doc = next(d for d in manifest['documents'] if d['relative_path'] == args.document)
    pages = [p for p in doc['pages'] if args.first <= p['page'] <= args.last]
    if len(pages) != args.last - args.first + 1:raise ValueError('Invalid page range.')
    existing = json.loads(args.receipt.read_text(encoding='utf-8')) if args.receipt.exists() else {'schema_version':1,'reviews':[]}
    records = {(r['source_sha256'],r['page']):r for r in existing['reviews']}
    for page in pages:
        for kind in ('text', 'render') if args.visual else ('text',):
            digest = hashlib.sha256(Path(page[kind+'_path']).read_bytes()).hexdigest()
            if digest != page[kind+'_sha256']:raise ValueError('Preserved file changed during review.')
        record = dict(source_sha256=doc['sha256'],page=page['page'],text_sha256=page['text_sha256'],
            render_sha256=page.get('render_sha256',''),text_reviewed=args.text,visual_reviewed=args.visual,
            reviewer=args.reviewer,reviewed_at=datetime.now(timezone.utc).isoformat(),notes=args.notes)
        records[(doc['sha256'],page['page'])] = record
    existing['reviews'] = sorted(records.values(),key=lambda r:(r['source_sha256'],r['page']))
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(existing,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'receipt':str(args.receipt),'records':len(records)},ensure_ascii=False))


if __name__ == '__main__':main()
