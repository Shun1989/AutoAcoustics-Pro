"""Print source-preserved pages for human/agent review; does not mark them read."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('document')
    parser.add_argument('first', type=int)
    parser.add_argument('last', type=int)
    parser.add_argument('--images', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'output/knowledge_corpus/source_manifest.json').read_text(encoding='utf-8'))
    document = next(d for d in manifest['documents'] if d['relative_path'] == args.document)
    pages = [p for p in document['pages'] if args.first <= p['page'] <= args.last]
    if len(pages) != args.last - args.first + 1:
        raise ValueError('Requested pages are outside the document.')
    if args.images:
        print(json.dumps([{'page':p['page'],'path':p['render_path']} for p in pages],ensure_ascii=False))
    else:
        for page in pages:
            print(f"=== {document['relative_path']} 第 {page['page']} 页 ===")
            print(Path(page['text_path']).read_text(encoding='utf-8'), end='\n')


if __name__ == '__main__':
    main()
