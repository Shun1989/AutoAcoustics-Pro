"""Build a resumable engineering corpus without equating extraction with reading.

Original source bytes and their SHA-256 are authoritative. PDF extraction does
not OCR raster pages or promise layout/Unicode fidelity. Every PDF page receives
exact extracted text, a PNG render, metadata/checksums and pending-review flags.
Only the three explicitly selected engineering/UI videos are processed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import unicodedata

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader
from pypdf.generic import ContentStream


SCHEMA_VERSION = 1
VIDEO_NAMES = {'AutoAcoustics Pro.mp4', '科普阻抗失配，Opus5.5直出.mp4', '智能制造.mp4'}
LOCK = threading.Lock()


def sha256(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.writing')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def log(message):
    with LOCK:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] {message}', flush=True)


def flag_text(text):
    replacements = text.count('\ufffd')
    private = sum(unicodedata.category(char) == 'Co' for char in text)
    controls = sum(unicodedata.category(char) == 'Cc' and char not in '\n\r\t\f' for char in text)
    suspicious = sorted(set(char for char in text if char == '\ufffd' or
                            unicodedata.category(char) == 'Co' or
                            (unicodedata.category(char) == 'Cc' and char not in '\n\r\t\f')))
    flags = []
    if not text.strip():
        flags.append('no_extractable_text_requires_visual_review_or_ocr')
    if replacements:
        flags.append('unicode_replacement_characters')
    if private:
        flags.append('private_use_glyphs_require_source_visual_check')
    if controls:
        flags.append('unexpected_control_characters')
    return {'flags': flags, 'replacement_character_count': replacements,
            'private_use_character_count': private, 'unexpected_control_character_count': controls,
            'suspicious_codepoints': [f'U+{ord(char):04X}' for char in suspicious],
            'character_count': len(text), 'non_whitespace_character_count': sum(not c.isspace() for c in text)}


def flag_body_quality(text, graphics):
    """Do not mistake a searchable URL watermark for the scanned page body.

    The concrete ABA/PGT manual (塑料齿轮设计手册), first131 pages, has an
    extractable www.bzfxw.com watermark while its engineering body is raster.
    This flag is an explicit heuristic, not an OCR/transcription/read attestation.
    A genuinely short title page is not labelled watermark-only without a URL.
    """
    compact = re.sub(r'\s+', '', text)
    without_urls = re.sub(r'(?:https?://)?(?:www\.)?bzfxw\.com', '', compact, flags=re.I)
    # Other explicit www/HTTP URLs are removed one original line at a time;
    # never consume text from a following line as part of a domain name.
    generic = []
    for line in text.splitlines():
        line = re.sub(r'(?:https?://|www\.)[A-Za-z0-9._/-]+', '', line)
        generic.append(line)
    had_url = without_urls != compact or ''.join(generic) != ''.join(text.splitlines())
    if without_urls == compact:
        without_urls = re.sub(r'\s+', '', '\n'.join(generic))
    # Standalone printed page numbers are metadata, not usable body text.
    without_page_numbers = re.sub(r'(?m)^\s*\d+\s*$', '', '\n'.join(generic))
    meaningful = re.sub(r'\s+', '', without_page_numbers)
    if 'bzfxw.com' in compact.lower():
        meaningful = without_urls
    meaningful = re.sub(r'^\d+$', '', meaningful)
    image_count = graphics.get('image_object_count', 0)+graphics.get('inline_image_commands', 0)
    flags = []
    if had_url and image_count > 0 and len(meaningful) < 20:
        flags.extend(('watermark_only_or_missing_body', 'extracted_text_may_be_incomplete',
                      'watermark_only_or_missing_body_requires_visual_review_or_ocr'))
    elif text.strip() and image_count > 0 and len(meaningful) < 20:
        flags.append('low_text_yield_with_raster_images_requires_visual_check')
    return {'meaningful_text_character_count_excluding_urls_page_numbers': len(meaningful),
            'body_text_quality_flags': flags,
            'body_text_quality_method': 'Heuristic removes explicit URL watermarks and standalone page numbers, preserves raw extraction; short image/title pages require review rather than assumed OCR failure.'}


def _object_id(value):
    return str(getattr(value, 'idnum', id(value)))+':'+str(getattr(value, 'generation', 0))


def page_graphics(page, reader):
    """Count PDF drawing commands, not inferred semantic figures or their content."""
    result = {'image_object_count': 0, 'image_draw_commands': 0,
              'inline_image_commands': 0, 'vector_paint_commands': 0,
              'vector_path_construction_commands': 0, 'form_draw_commands': 0,
              'font_count': 0, 'fonts_without_to_unicode': [], 'type0_fonts_without_to_unicode': []}
    images, fonts = set(), set()
    paint = {b'S', b's', b'f', b'F', b'f*', b'B', b'B*', b'b', b'b*'}
    paths = {b'm', b'l', b'c', b'v', b'y', b'h', b're'}

    def walk(contents, resources, ancestry):
        if contents is None:
            return
        resources = resources.get_object() if hasattr(resources, 'get_object') else resources
        resources = resources or {}
        font_objects = resources.get('/Font') or {}
        font_objects = font_objects.get_object() if hasattr(font_objects, 'get_object') else font_objects
        for name, reference in font_objects.items():
            identifier = _object_id(reference)
            if identifier in fonts:
                continue
            fonts.add(identifier)
            font = reference.get_object()
            if '/ToUnicode' not in font:
                label = f'{name}: {font.get("/BaseFont", "unknown")}'
                result['fonts_without_to_unicode'].append(label)
                if font.get('/Subtype') == '/Type0':
                    result['type0_fonts_without_to_unicode'].append(label)
        objects = resources.get('/XObject') or {}
        objects = objects.get_object() if hasattr(objects, 'get_object') else objects
        stream = contents if isinstance(contents, ContentStream) else ContentStream(contents, reader)
        for operands, operator in stream.operations:
            if operator in paint:
                result['vector_paint_commands'] += 1
            if operator in paths:
                result['vector_path_construction_commands'] += 1
            if operator == b'INLINE IMAGE':
                result['inline_image_commands'] += 1
            if operator != b'Do' or not operands or operands[0] not in objects:
                continue
            reference = objects[operands[0]]
            obj = reference.get_object()
            identifier = _object_id(reference)
            if obj.get('/Subtype') == '/Image':
                images.add(identifier)
                result['image_draw_commands'] += 1
            elif obj.get('/Subtype') == '/Form':
                result['form_draw_commands'] += 1
                if identifier not in ancestry:
                    walk(obj, obj.get('/Resources', resources), ancestry | {identifier})
    walk(page.get_contents(), page.get('/Resources'), set())
    result['image_object_count'] = len(images)
    result['font_count'] = len(fonts)
    return result


def document_paths(output, digest):
    directory = output/'documents'/digest
    return directory, directory/'pages.json', directory/'full_text.txt'


def inventory(source_root, output):
    groups = {}
    sources = []
    for source in sorted(source_root.iterdir(), key=lambda p: p.name.casefold()):
        if not source.is_file():
            continue
        if source.suffix.lower() != '.pdf' and source.name != 'NVH知识.txt' and source.name not in VIDEO_NAMES:
            continue
        digest = sha256(source)
        record = {'source_path': str(source.resolve()), 'relative_path': source.name,
                  'sha256': digest, 'size_bytes': source.stat().st_size,
                  'media_type': source.suffix.lower().lstrip('.')}
        sources.append(record)
        if source.suffix.lower() == '.pdf':
            groups.setdefault(digest, []).append(source)
    documents = []
    for digest, aliases in groups.items():
        source = aliases[0]
        directory, page_index, full_text = document_paths(output, digest)
        document = {'source_path': str(source.resolve()), 'relative_path': source.name, 'sha256': digest,
                    'aliases': [str(path.resolve()) for path in aliases],
                    'media_type': 'pdf', 'status': 'pending', 'review_status': 'not_read',
                    'page_index_path': str(page_index), 'full_text_path': str(full_text), 'pages': []}
        try:
            reader = PdfReader(source)
            document['page_count'] = len(reader.pages)
            document['encrypted'] = reader.is_encrypted
            for number in range(1, len(reader.pages)+1):
                document['pages'].append({'page': number,
                    'text_path': str(directory/'text'/f'page-{number:04d}.txt'),
                    'render_path': str(directory/'renders'/f'page-{number:04d}.png'),
                    'metadata_path': str(directory/'metadata'/f'page-{number:04d}.json'),
                    'review_status': 'not_read', 'extraction_status': 'pending', 'render_status': 'pending'})
        except Exception as error:
            document.update(status='source_open_failed', page_count=0, error=str(error))
        documents.append(document)
    excluded = sorted(p.name for p in source_root.iterdir() if p.is_file() and p.suffix.lower() == '.mp4' and p.name not in VIDEO_NAMES)
    return {'schema_version': SCHEMA_VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
            'source_root': str(source_root.resolve()), 'output_root': str(output.resolve()),
            'source_authority': 'Original files and source SHA-256; extracted text/renders are derived, not a reading attestation.',
            'documents': documents, 'sources': sources, 'videos': [], 'excluded_videos': excluded,
            'coverage': {'pdf_files_including_aliases': sum(len(d['aliases']) for d in documents),
                         'unique_pdfs': len(documents), 'unique_pdf_pages': sum(d['page_count'] for d in documents),
                         'pages_including_aliases': sum(d['page_count']*len(d['aliases']) for d in documents)},
            'limitations': ['Text extraction does not inspect or interpret figures, tables, formulae or raster-only text.',
                           'PDF punctuation/whitespace are retained exactly as returned by pypdf; extraction order may differ from the page.',
                           'Every page remains pending visual/text review. No OCR accuracy or video speech transcription is asserted.']}


def extract_document(document):
    source = Path(document['source_path'])
    if document['status'] == 'source_open_failed':
        return
    reader = PdfReader(source)
    texts = []
    started = time.monotonic()
    for row, page in zip(document['pages'], reader.pages):
        meta_path = Path(row['metadata_path'])
        cached = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else None
        text_path = Path(row['text_path'])
        if cached and cached.get('source_sha256') == document['sha256'] and cached.get('extraction_status') == 'extracted' and text_path.exists() and sha256(text_path) == cached.get('text_sha256'):
            row.update(cached)
            text = text_path.read_bytes().decode('utf-8')
        else:
            row.update(source_sha256=document['sha256'], review_status='pending_visual_and_text_review')
            try:
                text = page.extract_text() or ''
                row.update(flag_text(text))
                row['extraction_status'] = 'extracted'
            except Exception as error:
                text = ''
                row.update(flag_text(text), extraction_status='extraction_failed', extraction_error=str(error))
            try:
                row.update(page_graphics(page, reader))
                if row['type0_fonts_without_to_unicode']:
                    row['flags'].append('type0_font_without_unicode_mapping_requires_visual_check')
            except Exception as error:
                row['graphics_metadata_error'] = str(error)
                row['flags'].append('graphics_inventory_incomplete')
            row.update(flag_body_quality(text, row))
            row['flags'].extend(row['body_text_quality_flags'])
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_bytes(text.encode('utf-8'))
            row.update(text_sha256=sha256(text_path), text_method='pypdf.PdfReader.extract_text',
                       page_width_points=float(page.mediabox.width), page_height_points=float(page.mediabox.height),
                       page_rotation=int(page.get('/Rotate', 0)))
            write_json(meta_path, row)
        texts.append(text)
        if row['page'] % 25 == 0:
            log(f'text {document["relative_path"]}: {row["page"]}/{document["page_count"]}')
    full = Path(document['full_text_path'])
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes('\f'.join(texts).encode('utf-8'))
    document.update(status='extracted', review_status='pending_visual_and_text_review',
                    full_text_sha256=sha256(full), full_text_page_separator='U+000C inserted between pages',
                    extraction_seconds=round(time.monotonic()-started, 3))
    write_json(document['page_index_path'], {'schema_version': SCHEMA_VERSION, **document})
    log(f'extracted {document["relative_path"]}: {document["page_count"]} pages in {document["extraction_seconds"]}s')


def extract_text_document(source, output):
    path = Path(source['source_path'])
    raw = path.read_bytes()
    if raw.startswith(b'\xef\xbb\xbf'):
        encoding, text = 'utf-8-sig', raw.decode('utf-8-sig')
    else:
        try:
            encoding, text = 'utf-8', raw.decode('utf-8')
        except UnicodeDecodeError:
            encoding, text = 'gb18030', raw.decode('gb18030')
    directory, page_index, full = document_paths(output, source['sha256'])
    raw_path = directory/'original-bytes.txt'
    directory.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw)
    full.write_bytes(text.encode('utf-8'))
    row = {'page': 1, 'text_path': str(full), 'text_sha256': sha256(full),
           'render_path': None, 'render_sha256': None, 'render_status': 'not_applicable_text',
           'extraction_status': 'extracted', 'review_status': 'pending_text_review', **flag_text(text)}
    document = {**source, 'aliases': [source['source_path']], 'page_count': 1, 'encoding': encoding,
                'encoding_detection': 'BOM when present; strict UTF-8 decode, otherwise strict GB18030 decode.',
                'original_bytes_path': str(raw_path), 'original_bytes_sha256': sha256(raw_path),
                'full_text_path': str(full), 'full_text_sha256': sha256(full),
                'page_index_path': str(page_index), 'pages': [row], 'status': 'extracted',
                'review_status': 'pending_text_review', 'line_endings_and_punctuation_preserved': True}
    write_json(page_index, {'schema_version': SCHEMA_VERSION, **document})
    return document


def _font(size=18):
    for path in (Path('C:/Windows/Fonts/msyh.ttc'), Path('C:/Windows/Fonts/arial.ttf')):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def contact_sheets(items, directory, prefix='contact', label='page'):
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    font = _font()
    for offset in range(0, len(items), 12):
        chunk = items[offset:offset+12]
        sheet = Image.new('RGB', (990, 1464), 'white')
        draw = ImageDraw.Draw(sheet)
        for slot, item in enumerate(chunk):
            image = Image.open(item['path']).convert('RGB')
            image.thumbnail((310, 335))
            x, y = 10+(slot % 3)*330, 35+(slot//3)*366
            sheet.paste(image, (x+(310-image.width)//2, y))
            draw.text((x, y-28), f'{label} {item["number"]}', font=font, fill='black')
        destination = directory/f'{prefix}-{offset//12+1:03d}.jpg'
        sheet.save(destination, quality=88)
        records.append({'path': str(destination), 'sha256': sha256(destination),
                        'pages' if label == 'page' else 'frames': [item['number'] for item in chunk]})
    return records


def _process(args):
    return subprocess.run([str(a) for a in args], capture_output=True,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)


def render_document(document, renderer, dpi, checkpoint):
    if document.get('media_type') != 'pdf' or not document['pages']:
        return
    started = time.monotonic()
    directory = Path(document['page_index_path']).parent
    render_temp = directory/'render_tmp'
    render_temp.mkdir(parents=True, exist_ok=True)
    pending = []
    for row in document['pages']:
        image_path = Path(row['render_path'])
        valid = (image_path.exists() and row.get('render_dpi') == dpi and
                 row.get('render_status') == 'rendered' and sha256(image_path) == row.get('render_sha256'))
        if not valid:
            pending.append(row)
    for offset in range(0, len(pending), 20):
        chunk = pending[offset:offset+20]
        first, last = chunk[0]['page'], chunk[-1]['page']
        # A bounded contiguous range also recovers partially completed batches.
        process = _process([renderer, '-f', first, '-l', last, '-r', dpi, '-cropbox', '-png',
                            document['source_path'], render_temp/'page'])
        warning = process.stderr.decode('utf-8', errors='replace')[-12000:]
        candidates = {}
        for candidate in render_temp.glob('page-*.png'):
            try:
                number = int(candidate.stem.split('-')[-1])
            except ValueError:
                continue
            candidates[number] = candidate
        for row in chunk:
            candidate = candidates.get(row['page'])
            if candidate is None:
                row.update(render_status='render_failed', render_error=f'Poppler exit {process.returncode}: {warning}')
            else:
                try:
                    with Image.open(candidate) as image:
                        dimensions = [image.width, image.height]
                        image.verify()
                    target = Path(row['render_path'])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(candidate, target)
                    with LOCK:
                        row.update(render_status='rendered', render_dpi=dpi, render_sha256=sha256(target),
                                   render_width_pixels=dimensions[0], render_height_pixels=dimensions[1],
                                   render_method='Poppler pdftoppm -cropbox -png', review_status='pending_visual_and_text_review')
                        row.pop('render_error', None)
                except Exception as error:
                    row.update(render_status='render_failed', render_error=str(error))
            if warning:
                row['render_warnings'] = warning
            write_json(row['metadata_path'], row)
        write_json(document['page_index_path'], {'schema_version': SCHEMA_VERSION, **document})
        checkpoint()
        log(f'render {document["relative_path"]}: {sum(p.get("render_status")=="rendered" for p in document["pages"])}/{document["page_count"]}; {time.monotonic()-started:.1f}s')
    good = [{'path': row['render_path'], 'number': row['page']} for row in document['pages'] if row.get('render_status') == 'rendered']
    contacts = contact_sheets(good, directory/'contact_sheets')
    unchanged = sha256(document['source_path']) == document['sha256']
    with LOCK:
        document['contact_sheets'] = contacts
        document['render_seconds'] = round(time.monotonic()-started, 3)
        document['status'] = 'extracted_and_rendered' if len(good) == document['page_count'] else 'incomplete_render'
        document['source_unchanged'] = unchanged
    write_json(document['page_index_path'], {'schema_version': SCHEMA_VERSION, **document})
    checkpoint()


def process_video(source, output, ffprobe, ffmpeg, baseline):
    directory = output/'videos'/source['sha256']
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory/'ffprobe.json'
    process = _process([ffprobe, '-v', 'error', '-show_format', '-show_streams', '-of', 'json', source['source_path']])
    if process.returncode:
        return {**source, 'status': 'probe_failed', 'error': process.stderr.decode('utf-8', errors='replace')}
    metadata = json.loads(process.stdout)
    write_json(metadata_path, metadata)
    record = {**source, 'ffprobe_path': str(metadata_path), 'ffprobe_sha256': sha256(metadata_path),
              'duration_seconds': float(metadata.get('format', {}).get('duration', 0)),
              'status': 'probed', 'review_status': 'pending_frame_and_audio_review', 'frames': [], 'audio_tracks': [],
              'subtitle_tracks': [stream for stream in metadata['streams'] if stream['codec_type'] == 'subtitle']}
    if Path(source['source_path']).name == 'AutoAcoustics Pro.mp4' and (baseline/'manifest.json').exists():
        previous = json.loads((baseline/'manifest.json').read_text(encoding='utf-8'))
        if previous.get('sha256') == source['sha256']:
            for index, frame in enumerate(sorted(baseline.glob('frame-*.png')), start=1):
                record['frames'].append({'number': index, 'path': str(frame.resolve()), 'sha256': sha256(frame),
                                         'approximate_time_seconds': (index-1)*3+1.5})
            record.update(frame_sampling='Existing original UI baseline, one frame/3 s; timestamps approximate filter sample centres.',
                          existing_baseline_path=str((baseline/'manifest.json').resolve()),
                          existing_baseline_sha256=sha256(baseline/'manifest.json'))
            record['contact_sheets'] = [{'path': str(path.resolve()), 'sha256': sha256(path)} for path in sorted(baseline.glob('contact-*.jpg'))]
    if not record['frames']:
        frame_directory = directory/'frames'
        frame_directory.mkdir(exist_ok=True)
        process = _process([ffmpeg, '-y', '-v', 'error', '-i', source['source_path'], '-vf', 'fps=1/3',
                            '-vsync', '0', frame_directory/'frame-%06d.png'])
        if process.returncode:
            record['frame_error'] = process.stderr.decode('utf-8', errors='replace')
        for index, frame in enumerate(sorted(frame_directory.glob('frame-*.png')), start=1):
            record['frames'].append({'number': index, 'path': str(frame), 'sha256': sha256(frame),
                                     'approximate_time_seconds': (index-1)*3+1.5})
        record['frame_sampling'] = 'One frame/3 s; timestamps approximate filter sample centres, not a frame-complete reading.'
        record['contact_sheets'] = contact_sheets(record['frames'], directory/'contact_sheets', label='frame')
    audio = [stream for stream in metadata['streams'] if stream['codec_type'] == 'audio']
    for index, stream in enumerate(audio):
        destination = directory/f'audio-{index:02d}-16k-mono.wav'
        process = _process([ffmpeg, '-y', '-v', 'error', '-i', source['source_path'], '-map', f'0:a:{index}',
                            '-vn', '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', destination])
        track = {'original_stream': stream, 'speech_presence': 'not_assessed',
                 'transcription_status': 'pending_speech_review_or_transcription',
                 'subtitles_present': bool(record['subtitle_tracks'])}
        if process.returncode:
            track['extraction_error'] = process.stderr.decode('utf-8', errors='replace')
        else:
            track.update(path=str(destination), sha256=sha256(destination), derived_sample_rate=16000, derived_channels=1)
        record['audio_tracks'].append(track)
    if audio and not record['subtitle_tracks']:
        record['audio_note'] = 'Audio exists, no embedded subtitle stream. Speech has not been assessed/transcribed; no spoken content is claimed read.'
    elif not audio:
        record['audio_note'] = 'No audio stream reported by ffprobe.'
    record['source_unchanged'] = sha256(source['source_path']) == source['sha256']
    write_json(directory/'video.json', {'schema_version': SCHEMA_VERSION, **record})
    log(f'video {source["relative_path"]}: {len(record["frames"])} sampled frames, {len(audio)} audio, {len(record["subtitle_tracks"])} subtitle streams')
    return record


def verify_corpus(manifest_path):
    """Re-read all hashes/PNG checksums; never infer content-review completion."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    errors, source_count, text_count, image_count, frame_count = [], 0, 0, 0, 0
    aggregate = hashlib.sha256()
    def verify_hash(path, expected, kind):
        try:
            actual = sha256(path)
            if actual != expected:
                errors.append({'kind': kind, 'path': str(path), 'error': 'SHA-256 mismatch'})
                return False
            return True
        except Exception as error:
            errors.append({'kind': kind, 'path': str(path), 'error': str(error)})
            return False
    for source in manifest['sources']:
        source_count += verify_hash(source['source_path'], source['sha256'], 'original_source')
    for document in manifest['documents']:
        if [row['page'] for row in document['pages']] != list(range(1, document['page_count']+1)):
            errors.append({'kind': 'page_coverage', 'path': document['source_path'], 'error': 'Pages are not contiguous 1..N'})
        for row in document['pages']:
            if verify_hash(row['text_path'], row.get('text_sha256'), 'page_text'):
                text_count += 1
            aggregate.update(f'{document["sha256"]}:{row["page"]}:{row.get("text_sha256")}:{row.get("render_sha256")}\n'.encode())
            if document['media_type'] == 'pdf':
                path = row.get('render_path')
                if row.get('render_status') != 'rendered' or not path:
                    errors.append({'kind': 'page_render', 'path': str(path), 'error': 'Page is not rendered'})
                    continue
                valid = verify_hash(path, row.get('render_sha256'), 'page_png')
                try:
                    with Image.open(path) as image:
                        dimensions = [image.width, image.height]
                        image.verify()
                    if dimensions != [row['render_width_pixels'], row['render_height_pixels']]:
                        errors.append({'kind': 'page_png', 'path': path, 'error': 'Pixel dimensions mismatch'})
                        valid = False
                    image_count += valid
                except Exception as error:
                    errors.append({'kind': 'page_png', 'path': path, 'error': str(error)})
        verify_hash(document['full_text_path'], document['full_text_sha256'], 'full_text')
        if document['media_type'] == 'txt':
            verify_hash(document['original_bytes_path'], document['original_bytes_sha256'], 'original_text_bytes')
        for contact in document.get('contact_sheets', []):
            verify_hash(contact['path'], contact['sha256'], 'contact_sheet')
    for video in manifest['videos']:
        verify_hash(video['ffprobe_path'], video['ffprobe_sha256'], 'video_probe')
        for frame in video.get('frames', []):
            frame_count += verify_hash(frame['path'], frame['sha256'], 'video_frame')
        for track in video.get('audio_tracks', []):
            if 'path' in track:
                verify_hash(track['path'], track['sha256'], 'derived_audio')
        for contact in video.get('contact_sheets', []):
            verify_hash(contact['path'], contact['sha256'], 'video_contact_sheet')
    pages = [row for d in manifest['documents'] if d['media_type'] == 'pdf' for row in d['pages']]
    audit = {'schema_version': SCHEMA_VERSION, 'audited_at': datetime.now(timezone.utc).isoformat(),
             'manifest_path': str(manifest_path), 'manifest_sha256': sha256(manifest_path),
             'builder_path': str(Path(__file__).resolve()), 'builder_sha256': sha256(__file__),
             'passed': not errors, 'errors': errors, 'original_sources_verified': source_count,
             'page_text_files_verified_including_txt': text_count, 'pdf_png_files_crc_and_sha_verified': image_count,
             'video_frames_sha_verified': frame_count, 'ordered_page_checksums_sha256': aggregate.hexdigest(),
             'no_extractable_text_pages': sum('no_extractable_text_requires_visual_review_or_ocr' in p.get('flags', []) for p in pages),
             'watermark_only_or_missing_body_pages': sum('watermark_only_or_missing_body_requires_visual_review_or_ocr' in p.get('flags', []) for p in pages),
             'low_text_yield_with_raster_image_pages': sum('low_text_yield_with_raster_images_requires_visual_check' in p.get('flags', []) for p in pages),
             'glyph_mapping_flag_pages': sum(any(flag in p.get('flags', []) for flag in (
                 'unicode_replacement_characters', 'private_use_glyphs_require_source_visual_check',
                 'type0_font_without_unicode_mapping_requires_visual_check', 'unexpected_control_characters')) for p in pages),
             'render_batch_warning_page_records': sum('render_warnings' in p for p in pages),
             'video_status': [{'name': v['relative_path'], 'frames': len(v.get('frames', [])),
                               'audio_tracks': len(v.get('audio_tracks', [])), 'subtitle_tracks': len(v.get('subtitle_tracks', [])),
                               'speech_presence': [a.get('speech_presence') for a in v.get('audio_tracks', [])],
                               'transcription_status': [a.get('transcription_status') for a in v.get('audio_tracks', [])]} for v in manifest['videos']],
             'review_attestation': 'This is file/extraction/render integrity verification only, not content reading or figure interpretation.'}
    write_json(manifest_path.parent/'corpus_audit.json', audit)
    log('audit '+json.dumps({name: audit[name] for name in ('passed', 'original_sources_verified',
        'page_text_files_verified_including_txt', 'pdf_png_files_crc_and_sha_verified',
        'video_frames_sha_verified', 'no_extractable_text_pages', 'glyph_mapping_flag_pages')}, ensure_ascii=False))
    return audit


def refresh_body_quality(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for document in manifest['documents']:
        if document['media_type'] != 'pdf':
            continue
        for row in document['pages']:
            text = Path(row['text_path']).read_bytes().decode('utf-8')
            old = set(row.get('body_text_quality_flags', []))
            row['flags'] = [flag for flag in row.get('flags', []) if flag not in old]
            row.update(flag_body_quality(text, row))
            row['flags'].extend(row['body_text_quality_flags'])
            write_json(row['metadata_path'], row)
        write_json(document['page_index_path'], {'schema_version': SCHEMA_VERSION, **document})
    manifest['quality_refreshed_at'] = datetime.now(timezone.utc).isoformat()
    manifest['coverage']['watermark_only_or_missing_body_pages'] = sum(
        'watermark_only_or_missing_body_requires_visual_review_or_ocr' in p.get('flags', [])
        for d in manifest['documents'] if d['media_type'] == 'pdf' for p in d['pages'])
    write_json(manifest_path, manifest)
    log('body quality flags refreshed without modifying text, PNG or original files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True, help='Directory containing your own engineering material.')
    parser.add_argument('--output-dir', type=Path, default=Path('output/knowledge_corpus'))
    parser.add_argument('--stage', choices=('extract', 'render', 'all'), default='all')
    parser.add_argument('--dpi', type=int, default=120)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--renderer', type=Path, default=Path(shutil.which('pdftoppm') or 'pdftoppm'))
    parser.add_argument('--ffprobe', type=Path, default=Path(shutil.which('ffprobe') or 'ffprobe'))
    parser.add_argument('--ffmpeg', type=Path, default=Path(shutil.which('ffmpeg') or 'ffmpeg'))
    parser.add_argument('--video-baseline', type=Path, default=Path('artifacts/ui_revision/video_baseline'))
    parser.add_argument('--verify-only', action='store_true', help='Verify existing source_manifest and every recorded output without rebuilding.')
    parser.add_argument('--refresh-quality', action='store_true', help='Refresh scanned/watermark body-quality flags without re-rendering or changing extracted text.')
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir/'source_manifest.json'
    if args.refresh_quality:
        refresh_body_quality(manifest_path)
        audit = verify_corpus(manifest_path)
        raise SystemExit(0 if audit['passed'] else 1)
    if args.verify_only:
        audit = verify_corpus(manifest_path)
        raise SystemExit(0 if audit['passed'] else 1)
    manifest = inventory(args.input_dir.resolve(), args.output_dir)
    manifest.update(render_dpi=args.dpi, renderer=str(args.renderer), extraction_method='pypdf; no OCR')
    def checkpoint():
        with LOCK:
            manifest['coverage'].update(
                extracted_pdf_pages=sum(p.get('extraction_status') == 'extracted' for d in manifest['documents'] if d['media_type'] == 'pdf' for p in d['pages']),
                rendered_pdf_pages=sum(p.get('render_status') == 'rendered' for d in manifest['documents'] if d['media_type'] == 'pdf' for p in d['pages']),
                visually_reviewed_pdf_pages=0, content_read_pdf_pages=0)
            write_json(manifest_path, manifest)
    checkpoint()
    log('inventory '+json.dumps(manifest['coverage'], ensure_ascii=False))
    # Extraction always runs/reuses verified per-page cache, so --stage render
    # can resume from an interrupted invocation without trusting stale state.
    for document in sorted(manifest['documents'], key=lambda d: d['page_count']):
        extract_document(document)
        checkpoint()
    for source in manifest['sources']:
        if source['media_type'] == 'txt':
            manifest['documents'].append(extract_text_document(source, args.output_dir))
    checkpoint()
    if args.stage != 'extract':
        with ThreadPoolExecutor(max_workers=max(1, min(4, args.workers))) as pool:
            futures = [pool.submit(render_document, document, args.renderer, args.dpi, checkpoint)
                       for document in sorted(manifest['documents'], key=lambda d: d['page_count']) if document['media_type'] == 'pdf']
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    log(f'render worker failed: {type(error).__name__}: {error}')
    for source in manifest['sources']:
        if source['media_type'] == 'mp4':
            try:
                manifest['videos'].append(process_video(source, args.output_dir, args.ffprobe, args.ffmpeg, args.video_baseline))
            except Exception as error:
                manifest['videos'].append({**source, 'status': 'extraction_failed', 'error': str(error)})
            checkpoint()
    for source in manifest['sources']:
        source['source_unchanged'] = sha256(source['source_path']) == source['sha256']
    manifest['completed_at'] = datetime.now(timezone.utc).isoformat()
    checkpoint()
    log('finished '+json.dumps(manifest['coverage'], ensure_ascii=False))


if __name__ == '__main__':
    main()
