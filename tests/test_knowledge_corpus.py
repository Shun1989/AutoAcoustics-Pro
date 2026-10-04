"""Page-level provenance, integrity, idempotence and asynchronous UI handoff."""
import hashlib
import importlib
import json
import os
import threading
import time
import struct
import zlib
from pathlib import Path

import pytest

from autoacoustics.knowledge.models import KnowledgeEntry, KnowledgeError
from autoacoustics.knowledge.store import KnowledgeStore
from autoacoustics.knowledge.search import SearchEngine


def png_chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))


PNG = (b'\x89PNG\r\n\x1a\n' + png_chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
       + png_chunk(b'IDAT', zlib.compress(b'\x00\xff\xff\xff')) + png_chunk(b'IEND', b''))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fixture_manifest(tmp_path, texts=('齿轮啮合频率由齿数与转速确定。\n  原文间隔 Ω。\n', '加窗校正适用于本页的频谱定义。', '')):
    root = tmp_path / '工程资料'
    root.mkdir(exist_ok=True)
    pdf = root / '齿轮资料.pdf'
    pdf.write_bytes(b'%PDF-1.4\n% controlled provenance fixture\n%%EOF\n')
    pages = []
    for page, text in enumerate(texts, 1):
        text_path = root / f'page-{page}.txt'
        render_path = root / f'page-{page}.png'
        text_path.write_text(text, encoding='utf-8', newline='')
        render_path.write_bytes(PNG)
        pages.append(dict(page=page, text_path=text_path.name, render_path=render_path.name,
                          text_sha256=sha(text_path.read_bytes()), render_sha256=sha(PNG),
                          flags=['image_only'] if not text.strip() else [], review_status='not_reviewed'))
    manifest = root / 'source_manifest.json'
    data = dict(schema_version=1, source_root=str(root), documents=[dict(
        source_path=str(pdf), relative_path=pdf.name, sha256=sha(pdf.read_bytes()),
        aliases=['重复资料.pdf'], page_count=len(pages), pages=pages)])
    manifest.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    return manifest, data


def save_manifest(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')


def importer():
    return importlib.import_module('autoacoustics.knowledge.corpus').import_corpus


def test_pdf_pages_preserve_raw_text_aliases_and_do_not_overwrite_each_other(tmp_path):
    manifest, data = fixture_manifest(tmp_path)
    progress = []
    with KnowledgeStore(tmp_path / 'knowledge.sqlite') as store:
        result = importer()(store, manifest, progress=lambda done, total, message: progress.append((done, total, message)))
        records = store.list_sources()
        assert result.total_pages == 3 and result.added == 3 and result.duplicates == 0
        assert len(records) == 3 and {r.entry.source_locator for r in records} == {'page:1', 'page:2', 'page:3'}
        first = next(r for r in records if r.entry.source_locator == 'page:1')
        assert first.entry.body == (manifest.parent / 'page-1.txt').read_text(encoding='utf-8')
        assert first.entry.source_version == data['documents'][0]['sha256']
        assert first.entry.title == '齿轮资料.pdf · 第1页'
        assert first.entry.metadata['aliases'] == ('重复资料.pdf',)
        image = next(r for r in records if r.entry.source_locator == 'page:3')
        assert image.entry.body == '' and image.entry.metadata['visual_reviewed'] is False
        assert '文字待OCR' in image.entry.metadata['coverage_status']
        assert result.visual_reviewed_pages == 0 and result.image_only_pages == 1
        assert progress and progress[-1][0] == progress[-1][1]
        repeat = importer()(store, manifest)
        assert repeat.duplicates == 3 and repeat.added == 0 and len(store.list_sources()) == 3
        evidence = SearchEngine(store).search_local('齿轮啮合频率')
        assert evidence and evidence[0].metadata['page'] == 1
        assert evidence[0].metadata['render_path'] == str((manifest.parent / 'page-1.png').resolve())
        assert evidence[0].metadata['source_locator'] == 'page:1'
        assert evidence[0].snippet in first.entry.body


def test_same_source_new_version_preserves_prior_page_versions_and_restart(tmp_path):
    manifest, data = fixture_manifest(tmp_path, ('原版齿轮说明。', '原版噪声。'))
    db = tmp_path / 'knowledge.sqlite'
    with KnowledgeStore(db) as store:
        importer()(store, manifest)
        ids = {r.entry.source_locator: r.source_id for r in store.list_sources()}
        source = Path(data['documents'][0]['source_path'])
        source.write_bytes(source.read_bytes() + b'% revision 2')
        data['documents'][0]['sha256'] = sha(source.read_bytes())
        (manifest.parent / 'page-1.txt').write_text('新版齿轮说明。', encoding='utf-8')
        data['documents'][0]['pages'][0]['text_sha256'] = sha((manifest.parent / 'page-1.txt').read_bytes())
        save_manifest(manifest, data)
        result = importer()(store, manifest)
        assert result.updated == 2
    with KnowledgeStore(db) as store:
        assert {r.entry.source_locator: r.source_id for r in store.list_sources()} == ids
        assert store.get_source(ids['page:1'], 1).entry.body == '原版齿轮说明。'
        assert store.get_source(ids['page:1']).entry.body == '新版齿轮说明。'


@pytest.mark.parametrize('damage', ['missing_page', 'duplicate_page', 'text_hash', 'render_hash', 'source_hash', 'missing_text', 'video'])
def test_invalid_corpus_is_rejected_before_any_database_write(tmp_path, damage):
    manifest, data = fixture_manifest(tmp_path, ('齿轮。', '噪声。'))
    doc = data['documents'][0]
    if damage == 'missing_page': doc['pages'].pop()
    elif damage == 'duplicate_page': doc['pages'][1]['page'] = 1
    elif damage == 'text_hash': (manifest.parent / 'page-2.txt').write_text('更改', encoding='utf-8')
    elif damage == 'render_hash': (manifest.parent / 'page-2.png').write_bytes(b'not the fixed render')
    elif damage == 'source_hash': Path(doc['source_path']).write_bytes(b'changed source')
    elif damage == 'missing_text': (manifest.parent / 'page-2.txt').unlink()
    elif damage == 'video':
        video = manifest.parent / '非工程视频.mp4'; video.write_bytes(b'video')
        doc.update(source_path=str(video), sha256=sha(b'video'))
    save_manifest(manifest, data)
    with KnowledgeStore(tmp_path / 'knowledge.sqlite') as store:
        with pytest.raises(KnowledgeError): importer()(store, manifest)
        assert store.list_sources() == ()


def test_render_not_supplied_is_explicitly_incomplete_not_fully_reviewed(tmp_path):
    manifest, data = fixture_manifest(tmp_path, ('可检索原文齿轮。',))
    data['documents'][0]['pages'][0].update(render_path=None, render_sha256=None)
    save_manifest(manifest, data)
    with KnowledgeStore(tmp_path / 'knowledge.sqlite') as store:
        result = importer()(store, manifest)
        assert result.complete is False and result.render_missing_pages == 1
        assert result.visual_reviewed_pages == 0
        assert '原页图缺失' in store.list_sources()[0].entry.metadata['coverage_status']


def test_cancel_during_page_writes_rolls_back_all_new_pages(tmp_path):
    manifest, _ = fixture_manifest(tmp_path, ('齿轮。', '噪声。'))
    def cancel(done, total, message):
        if '导入' in message and done == 1:
            raise KnowledgeError('导入已取消')
    with KnowledgeStore(tmp_path / 'knowledge.sqlite') as store:
        with pytest.raises(KnowledgeError, match='取消'): importer()(store, manifest, progress=cancel)
        assert store.list_sources() == ()


def test_legacy_blank_locator_remains_compatible_and_distinct_page_keys(tmp_path):
    with KnowledgeStore(tmp_path / 'knowledge.sqlite') as store:
        old = store.add_source(KnowledgeEntry('资料', '齿轮说明。', source_path=tmp_path / 'same.pdf'))
        page = store.add_source(KnowledgeEntry('第1页', '齿轮正文。', source_path=tmp_path / 'same.pdf', source_locator='page:1'))
        assert old.source_id != page.source_id
        assert store.add_source(KnowledgeEntry('资料', '齿轮说明。', source_path=tmp_path / 'same.pdf')).status == 'duplicate'


def test_pending_image_page_is_stored_without_fake_body_or_searchable_evidence(tmp_path):
    manifest, data = fixture_manifest(tmp_path, ('',))
    page = data['documents'][0]['pages'][0]
    page.update(render_path=str(manifest.parent / 'not-rendered.png'), render_sha256=None, render_status='pending')
    save_manifest(manifest, data)
    with KnowledgeStore(tmp_path / 'knowledge.sqlite') as store:
        result = importer()(store, manifest)
        entry = store.list_sources()[0].entry
        assert entry.body == '' and result.image_only_pages == 1 and result.complete is False
        assert entry.metadata['review_status'] == 'not_reviewed'
        assert entry.metadata['visual_reviewed'] is False
        assert not SearchEngine(store).search_local('齿轮资料')


def test_original_txt_keeps_crlf_and_is_not_reported_as_missing_pdf_render(tmp_path):
    manifest, data = fixture_manifest(tmp_path, ('临时提取。',))
    source = manifest.parent / 'NVH知识.txt'
    original = '原始知识：齿轮啮合。\r\n\r\n \t保留换行和空格。\r\n'
    source.write_bytes(original.encode('utf-8'))
    doc = data['documents'][0]
    doc.update(source_path=str(source), sha256=sha(source.read_bytes()), media_type='txt')
    doc['pages'][0].update(text_path=str(source), text_sha256=sha(source.read_bytes()), render_path=None,
                           render_sha256=None, render_status='not_applicable_text')
    save_manifest(manifest, data)
    with KnowledgeStore(tmp_path / 'knowledge.sqlite') as store:
        result = importer()(store, manifest)
        assert store.list_sources()[0].entry.body == original
        assert result.render_missing_pages == 0 and result.complete is True
        assert result.visual_reviewed_pages == 0


def test_watermark_only_body_is_retained_but_not_used_as_engineering_evidence(tmp_path):
    manifest,data=fixture_manifest(tmp_path,('www.bzfxw.com\n',))
    data['documents'][0]['pages'][0]['flags']=['watermark_only_or_missing_body','extracted_text_may_be_incomplete']
    save_manifest(manifest,data)
    with KnowledgeStore(tmp_path/'knowledge.sqlite') as store:
        result=importer()(store,manifest)
        entry=store.list_sources()[0].entry
        assert entry.body=='www.bzfxw.com\n'
        assert '正文待OCR' in entry.metadata['coverage_status']
        assert result.incomplete_text_pages==1 and result.text_reviewed_pages==0
        assert not SearchEngine(store).search_local('齿轮资料')


def test_page_image_changed_after_import_is_not_shown_as_fixed_evidence(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6.QtWidgets import QApplication
    from autoacoustics.ui import knowledge_dialog as ui
    manifest, _ = fixture_manifest(tmp_path, ('齿轮啮合。',))
    app = QApplication.instance() or QApplication([])
    dialog = ui.KnowledgeDialog(tmp_path / 'knowledge.sqlite')
    try:
        importer()(dialog.store, manifest)
        entry = dialog.store.list_sources()[0].entry
        Path(entry.metadata['render_path']).write_bytes(PNG + b'changed')
        assert dialog.show_page_image(entry.metadata) is None
        assert '哈希' in dialog.status.text()
    finally:
        dialog.close(); app.processEvents()


def test_dialog_imports_on_its_own_connection_and_opens_exact_page_image(tmp_path, monkeypatch):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6.QtWidgets import QApplication, QPushButton
    from autoacoustics.ui import knowledge_dialog as ui
    corpus = importlib.import_module('autoacoustics.knowledge.corpus')
    settings_values={}
    class LocalSettings:
        def __init__(self,*args):pass
        def value(self,key,default=None):return settings_values.get(key,default)
        def setValue(self,key,value):settings_values[key]=value
    monkeypatch.setattr(ui,'QSettings',LocalSettings)
    manifest, _ = fixture_manifest(tmp_path, ('齿轮啮合原文。',))
    app = QApplication.instance() or QApplication([])
    main_thread = threading.get_ident()
    observed = []
    original = corpus.import_corpus
    def tracked(store, path, progress=None):
        observed.append((threading.get_ident(), store))
        return original(store, path, progress)
    monkeypatch.setattr(ui, 'import_corpus', tracked)
    dialog = ui.KnowledgeDialog(tmp_path / 'knowledge.sqlite')
    try:
        assert dialog.findChild(QPushButton, 'knowledgeImportCorpus') is not None
        dialog.import_corpus_file(manifest)
        deadline = time.monotonic() + 5
        while dialog.corpus_busy and time.monotonic() < deadline:
            app.processEvents(); time.sleep(.005)
        app.processEvents()
        assert not dialog.corpus_busy and dialog.sources.count() == 1
        assert observed and observed[0][0] != main_thread and observed[0][1] is not dialog.store
        dialog.question.setPlainText('齿轮啮合'); dialog.prepare_preview()
        evidence = dialog.preview.evidence[0]
        assert '未核验' in dialog.evidence_list.item(0).text()
        popup = dialog.open_evidence(evidence.evidence_id)
        assert popup.findChild(QPushButton, 'knowledgeEvidencePageImage') is not None
        image = dialog.show_page_image(evidence.metadata)
        assert image is not None and image.windowTitle().endswith('第1页')
        assert '未核验' in popup.findChild(ui.QTextBrowser).toPlainText()
    finally:
        dialog.close(); app.processEvents()


def review_fixture(manifest, data):
    page=data['documents'][0]['pages'][0]
    review=dict(source_sha256=data['documents'][0]['sha256'], page=1,
                text_sha256=page['text_sha256'], render_sha256=page['render_sha256'],
                text_reviewed=True, visual_reviewed=True, reviewer='explicit local fixture reviewer',
                reviewed_at='2026-10-04T05:00:00+00:00', notes='页级审阅记录；不是标准认证。')
    receipt=manifest.parent/'page_reviews.json'
    receipt.write_text(json.dumps(dict(schema_version=1,reviews=[review]),ensure_ascii=False),encoding='utf-8')
    data['reviews_path']=receipt.name
    save_manifest(manifest,data)
    return receipt,review


def test_only_independent_matching_review_receipt_changes_page_review_state(tmp_path):
    manifest,data=fixture_manifest(tmp_path,('齿轮原文。',))
    # Extractor status alone is never a reading attestation.
    data['documents'][0]['pages'][0]['review_status']='reviewed'
    save_manifest(manifest,data)
    with KnowledgeStore(tmp_path/'knowledge.sqlite') as store:
        before=importer()(store,manifest)
        assert before.visual_reviewed_pages==0
        receipt,review=review_fixture(manifest,data)
        result=importer()(store,manifest)
        entry=store.list_sources()[0].entry
        assert result.visual_reviewed_pages==1 and result.text_reviewed_pages==1
        assert entry.metadata['visual_reviewed'] is True and entry.metadata['text_reviewed'] is True
        assert entry.metadata['review_reviewer']==review['reviewer']
        assert entry.metadata['review_receipt_sha256']==sha(receipt.read_bytes())
        assert '未核验' not in entry.metadata['coverage_status']
        assert '不是标准认证' in entry.metadata['review_notes']


@pytest.mark.parametrize('damage',['source_hash','text_hash','render_hash','missing_reviewer','unknown_page','untyped_true'])
def test_review_receipt_with_unmatched_or_unattributed_claim_is_rejected(tmp_path,damage):
    manifest,data=fixture_manifest(tmp_path,('齿轮原文。',))
    receipt,review=review_fixture(manifest,data)
    if damage=='source_hash':review['source_sha256']='0'*64
    elif damage=='text_hash':review['text_sha256']='0'*64
    elif damage=='render_hash':review['render_sha256']='0'*64
    elif damage=='missing_reviewer':review['reviewer']=''
    elif damage=='unknown_page':review['page']=2
    elif damage=='untyped_true':review['visual_reviewed']='true'
    receipt.write_text(json.dumps(dict(schema_version=1,reviews=[review])),encoding='utf-8')
    with KnowledgeStore(tmp_path/'knowledge.sqlite') as store:
        with pytest.raises(KnowledgeError):importer()(store,manifest)
        assert store.list_sources()==()
