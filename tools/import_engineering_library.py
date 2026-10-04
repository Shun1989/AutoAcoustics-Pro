"""Merge attributed review receipts and import the preserved local corpus."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--database',type=Path)
    parser.add_argument('--merge-only',action='store_true')
    args=parser.parse_args()
    manifest_path=ROOT/'output/knowledge_corpus/source_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    known={(d['sha256'],p['page']):p for d in manifest['documents'] for p in d['pages']}
    merged={};files=[]
    for path in sorted((manifest_path.parent/'reviews').rglob('*.json')):
        receipt=json.loads(path.read_text(encoding='utf-8'))
        if receipt.get('schema_version')!=1 or not isinstance(receipt.get('reviews'),list):continue
        files.append(str(path))
        for record in receipt['reviews']:
            key=(record['source_sha256'],record['page'])
            if key not in known:raise ValueError('Review points outside corpus: '+str(key))
            page=known[key]
            if record['text_sha256']!=page['text_sha256'] or (record['visual_reviewed'] and record['render_sha256']!=page['render_sha256']):
                raise ValueError('Review hashes changed: '+str(key))
            if key in merged and merged[key]!=record:raise ValueError('Conflicting overlapping review ownership: '+str(key))
            merged[key]=record
    review_path=manifest_path.parent/'merged_page_reviews.json'
    review_path.write_text(json.dumps({'schema_version':1,'reviews':list(merged.values()),'input_receipts':files},ensure_ascii=False,indent=2),encoding='utf-8')
    manifest['reviews_path']=str(review_path)
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    coverage=[]
    for document in manifest['documents']:
        reviewed=[merged.get((document['sha256'],p['page']),{}) for p in document['pages']]
        coverage.append({'document':document['relative_path'],'pages':document['page_count'],
            'text_reviewed_pages':sum(r.get('text_reviewed',False) for r in reviewed),
            'visual_reviewed_pages':sum(r.get('visual_reviewed',False) for r in reviewed),
            'unreviewed_pages':[p['page'] for p,r in zip(document['pages'],reviewed) if not r.get('text_reviewed',False) or (document['media_type']=='pdf' and not r.get('visual_reviewed',False))]})
    proof={'schema_version':1,'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'review_receipt_sha256':hashlib.sha256(review_path.read_bytes()).hexdigest(),
        'documents':coverage,'source_file_count':len(manifest['documents']),
        'reviewed_records':len(merged),'actual_review_is_not_zero_error_certification':True}
    if not args.merge_only:
        from autoacoustics.knowledge.corpus import import_corpus
        from autoacoustics.knowledge.models import KnowledgeEntry
        from autoacoustics.knowledge.search import SearchEngine
        from autoacoustics.knowledge.store import KnowledgeStore
        if args.database is None:
            from PyQt6.QtCore import QStandardPaths
            from PyQt6.QtWidgets import QApplication
            app=QApplication([]);app.setApplicationName('AutoAcoustics Pro');app.setOrganizationName('AutoAcoustics')
            args.database=Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation))/'knowledge.sqlite'
        with KnowledgeStore(args.database) as store:
            result=import_corpus(store,manifest_path)
            proof['database']=str(args.database.resolve());proof['import']=asdict(result)
            interpretations=[]
            for path in sorted((ROOT/'docs/knowledge_reviews').glob('*.md')):
                entry=KnowledgeEntry(path.stem+' · 工程解读',path.read_text(encoding='utf-8'),
                    tags=('工程解读','非原文','需核对原页'),source_path=path,
                    source_version=hashlib.sha256(path.read_bytes()).hexdigest())
                saved=store.add_source(entry);interpretations.append({'path':str(path),'status':saved.status})
            proof['interpretations']=interpretations
            engine=SearchEngine(store)
            questions=['频谱泄漏与窗函数','声压与声功率有什么区别','齿轮啮合频率与边带','角度重采样与阶次','响度与声压级']
            proof['retrieval']=[{'question':q,'hits':[{'title':e.title,'source':str(e.source_path),
                'page':e.metadata.get('page'),'coverage':e.metadata.get('coverage_status','工程解读'),
                'excerpt':e.snippet[:240] if hasattr(e,'snippet') else e.text[:240]} for e in engine.search_local(q)]} for q in questions]
    (manifest_path.parent/'library_import.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(json.dumps({'reviews':len(merged),'database':proof.get('database'),'import':proof.get('import')},ensure_ascii=False,default=str),flush=True)


if __name__=='__main__':main()
