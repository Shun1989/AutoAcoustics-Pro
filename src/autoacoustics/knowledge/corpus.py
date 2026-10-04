"""Import locally extracted engineering pages without inventing image content.

The manifest is a provenance receipt, not evidence that a page was understood.
All declared source/text/render hashes and continuous page ranges are checked
before an atomic source import. Original UTF-8 text is stored without rewriting.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

from .models import KnowledgeEntry, KnowledgeError


@dataclass(frozen=True)
class CorpusImportResult:
    manifest_path: Path
    documents: int
    total_pages: int
    added: int
    updated: int
    duplicates: int
    image_only_pages: int
    render_missing_pages: int
    visual_reviewed_pages: int
    complete: bool
    text_reviewed_pages: int = 0
    manifest_sha256: str = ''
    incomplete_text_pages: int = 0


def _hash_file(path):
    digest = hashlib.sha256()
    try:
        with path.open('rb') as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise KnowledgeError(f'工程资料文件不可读取：{path}') from error
    return digest.hexdigest()


def _expected_hash(value, label):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-fA-F]{64}', value):
        raise KnowledgeError(f'{label} 缺少有效 SHA256。')
    return value.lower()


def _path(value, base, label):
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeError(f'{label} 缺少文件路径。')
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _checked_file(value, expected, base, label, cache):
    path = _path(value, base, label)
    wanted = _expected_hash(expected, label)
    if path not in cache:
        cache[path] = _hash_file(path)
    if cache[path] != wanted:
        raise KnowledgeError(f'{label} SHA256 不符，未导入：{path}')
    return path


def _load_reviews(document, base):
    """A separate attributed receipt, never the extractor's review_status.

    Hash binding verifies what was claimed reviewed; it cannot certify the
    reviewer's understanding or establish an acoustic standard's authority.
    """
    if not document.get('reviews_path'):return {}, '', ''
    path = _path(document['reviews_path'], base, '页级审阅记录')
    try:
        raw = path.read_bytes()
        receipt = json.loads(raw.decode('utf-8-sig'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise KnowledgeError('独立页级审阅记录无法读取。') from error
    if not isinstance(receipt, dict) or receipt.get('schema_version') != 1 or not isinstance(receipt.get('reviews'), list):
        raise KnowledgeError('独立审阅记录须为 schema_version=1 / reviews 数组。')
    output = {}
    for review in receipt['reviews']:
        if not isinstance(review, dict):raise KnowledgeError('页级审阅记录须为对象。')
        number = review.get('page')
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:raise KnowledgeError('审阅记录缺少有效页码。')
        key = (_expected_hash(review.get('source_sha256'), '审阅来源'), number)
        if key in output:raise KnowledgeError('同一来源页存在重复审阅记录，请明确合并。')
        for field in ('text_reviewed', 'visual_reviewed'):
            if not isinstance(review.get(field), bool):raise KnowledgeError('审阅状态须为明确布尔值，不接受字符串或数字。')
        if not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip():raise KnowledgeError('审阅记录缺少审阅者。')
        try:
            date = datetime.fromisoformat(review['reviewed_at'])
            if date.tzinfo is None:raise ValueError('missing timezone')
        except (KeyError, TypeError, ValueError) as error:raise KnowledgeError('审阅记录缺少带时区的 ISO8601 日期。') from error
        _expected_hash(review.get('text_sha256'), '审阅原文')
        if review['visual_reviewed']:_expected_hash(review.get('render_sha256'), '审阅原图')
        if not isinstance(review.get('notes', ''), str):raise KnowledgeError('审阅备注须为文字。')
        output[key] = review
    return output, str(path), hashlib.sha256(raw).hexdigest()


def import_corpus(store, manifest, progress=None):
    """Return verified coverage/counts. ``progress(done,total,message)`` may cancel.

    Relative extracted paths are resolved beside the manifest. Original source
    paths may be absolute or relative to source_root. A render omitted from the
    manifest is explicitly incomplete; a declared missing/corrupt file fails.
    Only PDF/TXT/Markdown documents are accepted, never arbitrary video files.
    """
    manifest = Path(manifest).resolve()
    try:
        raw = manifest.read_bytes()
        document = json.loads(raw.decode('utf-8-sig'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise KnowledgeError('工程资料清单无法读取；请选择有效 source_manifest.json。') from error
    if not isinstance(document, dict) or document.get('schema_version') != 1:
        raise KnowledgeError('工程资料清单须使用 schema_version=1。')
    docs = document.get('documents')
    if not isinstance(docs, list) or not docs:
        raise KnowledgeError('工程资料清单没有 documents。')
    source_root = _path(document.get('source_root', '.'), manifest.parent, '源目录')
    reviews, review_path, review_hash = _load_reviews(document, manifest.parent)
    used_reviews = set()
    entries = []
    cache = {}
    seen_sources = set()
    total = 0
    for doc in docs:
        if not isinstance(doc, dict):raise KnowledgeError('工程资料文档条目须为对象。')
        count = doc.get('page_count')
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise KnowledgeError('文档 page_count 须为正整数。')
        pages = doc.get('pages')
        if not isinstance(pages, list) or len(pages) != count:
            raise KnowledgeError('文档页数与清单不符；缺页不能标为完整资料。')
        numbers = [page.get('page') if isinstance(page, dict) else None for page in pages]
        if any(isinstance(n, bool) or not isinstance(n, int) for n in numbers) or sorted(numbers) != list(range(1, count + 1)):
            raise KnowledgeError('文档页码须完整覆盖 1..page_count，不得重复或缺页。')
        total += count
    done = 0
    for doc in docs:
        source = _checked_file(doc.get('source_path'), doc.get('sha256'), source_root, '原资料', cache)
        if source.suffix.lower() not in ('.pdf', '.txt', '.md', '.markdown'):
            raise KnowledgeError('工程语料仅接受明确提取的 PDF/TXT/Markdown；视频须另行核验，不能自动导入。')
        source_hash = _expected_hash(doc['sha256'], '原资料')
        if source_hash in seen_sources:
            raise KnowledgeError('同内容资料重复列入 documents；重复文件须使用 aliases，避免重复检索权重。')
        seen_sources.add(source_hash)
        aliases = doc.get('aliases', [])
        if not isinstance(aliases, list) or any(not isinstance(a, str) for a in aliases):
            raise KnowledgeError('资料 aliases 须为文字数组。')
        for page in sorted(doc['pages'], key=lambda item: item['page']):
            number = page['page']
            text_path = _checked_file(page.get('text_path'), page.get('text_sha256'), manifest.parent, f'第{number}页原文', cache)
            try:
                text = text_path.read_bytes().decode('utf-8')
            except (OSError, UnicodeError) as error:
                raise KnowledgeError(f'第{number}页提取文字须为 UTF-8，未导入。') from error
            # Reading and validation use the exact same text bytes.
            if hashlib.sha256(text.encode('utf-8')).hexdigest() != _expected_hash(page.get('text_sha256'), '提取文字'):
                raise KnowledgeError('校验期间提取文字发生变化，请重新导入。')
            render_path = None
            pending_render = page.get('render_status') in ('pending', 'render_failed')
            text_document = source.suffix.lower() != '.pdf'
            if page.get('render_path') and not (pending_render and not page.get('render_sha256')):
                render_path = _checked_file(page['render_path'], page.get('render_sha256'), manifest.parent, f'第{number}页原图', cache)
            elif page.get('render_sha256'):
                raise KnowledgeError('原页图仅给出 hash，未提供路径。')
            flags = page.get('flags', [])
            if not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags):
                raise KnowledgeError('页面 flags 须为文字数组。')
            if not text.strip() and render_path is None and text_document:
                raise KnowledgeError('文本资料没有正文，不能作为已覆盖资料导入。')
            review_status = page.get('review_status', 'not_reviewed')
            if not isinstance(review_status, str):raise KnowledgeError('页面 review_status 须为文字。')
            review = reviews.get((source_hash, number))
            if review:
                if _expected_hash(review['text_sha256'], '审阅原文') != _expected_hash(page['text_sha256'], '页面原文'):
                    raise KnowledgeError(f'{source.name} 第{number}页审阅原文哈希不符。')
                if review['visual_reviewed'] and (render_path is None or _expected_hash(review['render_sha256'], '审阅原图') != _expected_hash(page.get('render_sha256'), '页面原图')):
                    raise KnowledgeError(f'{source.name} 第{number}页审阅图哈希不符或原页图缺失。')
                used_reviews.add((source_hash, number))
            # A manifest status is preserved verbatim; no automatic assertion of
            # human review is inferred from extraction/rendering or a filename.
            coverage = [] if text_document else ['原页图已按独立记录审阅' if review and review['visual_reviewed'] else '原页图未核验']
            coverage.append('文字已按独立记录审阅' if review and review['text_reviewed'] else '提取文字未核验')
            if not text.strip():coverage.append('无可检索提取文字；原页待OCR' if review and review['visual_reviewed'] else '文字待OCR')
            if 'watermark_only_or_missing_body' in flags:coverage.append('仅水印或正文缺失；正文待OCR')
            elif 'extracted_text_may_be_incomplete' in flags:coverage.append('提取正文可能不完整；原页图文待核验')
            if any('glyph' in flag or 'unicode' in flag or 'control' in flag for flag in flags):coverage.append('提取字形 / 字符待核验')
            if render_path is None and not text_document:coverage.append('原页图缺失')
            if text_document:coverage.append('原始文本；不适用页图核验')
            metadata = dict(page=number, page_count=doc['page_count'], aliases=aliases,
                            media_type=doc.get('media_type', 'pdf' if not text_document else 'txt'),
                            relative_path=doc.get('relative_path', source.name),
                            text_path=str(text_path), text_sha256=page['text_sha256'].lower(),
                            render_path=str(render_path) if render_path else '',
                            render_sha256=page.get('render_sha256') or '', flags=flags,
                            extraction_status=page.get('extraction_status', 'extracted'),
                            render_status=page.get('render_status', 'rendered' if render_path else 'missing'),
                            review_status=review_status, visual_reviewed=bool(review and review['visual_reviewed']),
                            text_reviewed=bool(review and review['text_reviewed']),
                            coverage_status='；'.join(coverage), corpus_manifest=str(manifest))
            if review:
                metadata.update(review_reviewer=review['reviewer'], review_date=review['reviewed_at'],
                                review_notes=review.get('notes', ''), review_receipt=review_path, review_receipt_sha256=review_hash)
            # The manifest hash is an import receipt, not page content identity:
            # changing an unrelated page must not create versions for this page.
            entry = KnowledgeEntry(source.name + f' · 第{number}页', text,
                                   tags=('工程原始资料', '页原文'), source_path=source,
                                   source_version=source_hash, source_locator=f'page:{number}', metadata=metadata)
            entries.append(entry)
            done += 1
            if progress:progress(done, total, f'校验 {source.name} 第{number}页')
    if set(reviews) != used_reviews:raise KnowledgeError('独立审阅记录引用的来源 / 页码不在本语料内，未导入。')
    counts = {'added': 0, 'updated': 0, 'duplicate': 0}
    with store.source_transaction():
        for index, entry in enumerate(entries, 1):
            saved = store.add_source(entry)
            counts[saved.status] += 1
            if progress:progress(index, total, f'导入 {entry.title}')
    missing = sum(not entry.metadata['render_path'] and entry.metadata['media_type']=='pdf' for entry in entries)
    return CorpusImportResult(manifest, len(docs), len(entries), counts['added'], counts['updated'], counts['duplicate'],
                              sum(not entry.body.strip() for entry in entries), missing,
                              sum(entry.metadata['visual_reviewed'] for entry in entries), missing == 0,
                              sum(entry.metadata['text_reviewed'] for entry in entries), hashlib.sha256(raw).hexdigest(),
                              sum(not entry.body.strip() or 'watermark_only_or_missing_body' in entry.metadata['flags'] or 'extracted_text_may_be_incomplete' in entry.metadata['flags'] for entry in entries))
