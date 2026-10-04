"""Two-process release check using isolated synthetic sources and loopback HTTP.

This diagnostic never contacts a real provider or accesses an existing key.
Its request contains schema_version, mode=create|reopen, workspace_dir,
optional report_path and optional check_windows_credentials.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import ssl
import sys
import threading
import time
from uuid import uuid4

from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtWidgets import QPushButton, QTextBrowser

from .knowledge.credentials import CredentialError, CredentialVault
from .knowledge.provider import ProviderClient, load_provider, save_provider
from .knowledge.store import KnowledgeStore
from .ui.knowledge_dialog import KnowledgeDialog


TITLE = '软件验收蜗杆润滑'
BODY_V1 = '软件诊断夹具：蜗杆润滑牌号为 LUB-V1，内容仅供软件协议验证。'
BODY_V2 = '软件诊断夹具：蜗杆润滑牌号为 LUB-V2，内容仅供软件协议验证。'
ACTUAL_MODEL = 'release-protocol-fixture'


class DiagnosticFailure(RuntimeError):
    pass


def _check(condition, name, checks):
    if not condition:
        raise DiagnosticFailure(name)
    checks[name] = True


def _wait(app, predicate, name, timeout=5):
    end = time.monotonic() + timeout
    while not predicate() and time.monotonic() < end:
        app.processEvents()
        time.sleep(.005)
    app.processEvents()
    if not predicate():
        raise DiagnosticFailure(name)


def _click(owner, name):
    widget = owner.findChild(QPushButton, name)
    if widget is None:
        raise DiagnosticFailure('missing_' + name)
    widget.click()


def _open(app, path):
    dialog = KnowledgeDialog(path)
    dialog.show()
    app.processEvents()
    return dialog


def _close(app, dialog):
    dialog.close()
    app.processEvents()


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def _endpoint(secret):
    """Record only authorization equality, never the Authorization value."""
    records = []
    control = {'delay': 0.}
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                authenticated = self.headers.get('Authorization') == 'Bearer ' + secret
                records.append({'path': self.path, 'authenticated': authenticated, 'payload': payload})
                if control['delay']:
                    release.wait(control['delay'])
                try:
                    content = json.loads(payload['messages'][-1]['content'])
                    identity = content['evidence'][0]['evidence_id']
                except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                    identity = 'protocol_connection_probe'
                body = json.dumps({'model': ACTUAL_MODEL, 'choices': [
                    {'message': {'content': '软件协议验证回答 [' + identity + ']'}}]}).encode('utf-8')
                self.send_response(200 if authenticated else 401)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (OSError, ValueError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=.02), daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', records, control
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def _import(app, dialog, path, mapping=None):
    preview = dialog.import_file(path)
    if preview is None:
        raise DiagnosticFailure('import_dialog_opened')
    if mapping:
        for field, column in mapping.items():
            widget = preview.mapping[field]
            index = widget.findData(column)
            if index < 0:
                raise DiagnosticFailure('csv_mapping_column_exists')
            widget.setCurrentIndex(index)
        _click(preview, 'knowledgeImportMap')
    app.processEvents()
    if preview.preview is None:
        raise DiagnosticFailure('import_preview_available')
    preview.apply_button.click()
    app.processEvents()
    statuses = [preview.table.item(row, 2).text() for row in range(preview.table.rowCount())]
    preview.close()
    return statuses


def _windows_credentials(request, dialog, secret, messages):
    result = {'passed': False, 'status': 'not_requested'}
    if not request.get('check_windows_credentials', False):
        return result
    vault = CredentialVault()
    reference = None
    original = load_provider(dialog.store)
    try:
        reference = vault.store(secret, name='release-diagnostic-' + uuid4().hex)
        if vault.get(reference) != secret:
            raise DiagnosticFailure('windows_credential_roundtrip')
        config = replace(original, credential_ref=reference)
        save_provider(dialog.store, config)
        reply = ProviderClient(load_provider(dialog.store)).complete(messages)
        if reply.status != 'ok' or reply.model != ACTUAL_MODEL:
            raise DiagnosticFailure('windows_credential_provider_chain')
        vault.delete(reference)
        if vault.status(reference) != 'missing':
            raise DiagnosticFailure('windows_credential_deleted')
        result = {'passed': True, 'status': 'passed', 'authenticated_loopback': True}
    except CredentialError as error:
        if error.win32_error != 1312:
            raise DiagnosticFailure('windows_credentials_unavailable') from error
        result = {'passed': False, 'status': 'environment_unavailable', 'win32_error': 1312}
    finally:
        save_provider(dialog.store, original)
        if reference:
            vault.delete(reference)
    return result


def _create(app, request, workspace, report):
    checks = report['checks']
    database = workspace / 'release_knowledge.sqlite'
    state_path = workspace / 'state.json'
    _check(not database.exists() and not state_path.exists(), 'empty_diagnostic_database', checks)
    markdown = workspace / 'software_knowledge.md'
    csv_path = workspace / 'software_cases.csv'
    json_path = workspace / 'software_cases.json'
    _check(not any(path.exists() for path in (markdown, csv_path, json_path)), 'new_diagnostic_fixtures', checks)
    markdown.write_text('# ' + TITLE + '\n\n' + BODY_V1, encoding='utf-8')
    csv_path.write_text('编号,标题,设备,观察,结果\n1,合成CSV启动记录,SW-CSV,软件字段验收,待人工确认\n2,,SW-CSV,错误行,\n', encoding='utf-8-sig')
    json_path.write_text(json.dumps([{'title': '合成JSON停止记录', 'equipment': 'SW-JSON',
        'observation': '软件导入字段验收'}], ensure_ascii=False), encoding='utf-8')
    hashes = {str(path): _digest(path) for path in (markdown, csv_path, json_path)}
    dialog = _open(app, database)
    env_name = 'AAC_KNOWLEDGE_RELEASE_' + uuid4().hex.upper()
    secret = 'synthetic-release-' + uuid4().hex
    os.environ[env_name] = secret
    try:
        _check(dialog.sources.count() == 0, 'empty_library_visible', checks)
        _check(_import(app, dialog, markdown) == ['added'], 'markdown_imported', checks)
        mapping = {'source_record_id': '编号', 'title': '标题', 'equipment': '设备',
                   'observation': '观察', 'outcome': '结果'}
        _check(_import(app, dialog, csv_path, mapping) == ['added', 'failed'],
               'csv_explicit_mapping_and_partial_error', checks)
        _check(_import(app, dialog, csv_path, mapping) == ['duplicate', 'failed'],
               'duplicate_import_preserves_identity', checks)
        _check(_import(app, dialog, json_path) == ['added'], 'json_declared_fields_imported', checks)
        records = dialog.store.list_sources()
        _check(len(records) == 3, 'three_imported_sources', checks)
        json_record = next(record for record in records if record.entry.title == '合成JSON停止记录')
        _check(json_record.entry.actions == json_record.entry.outcome == '', 'json_does_not_invent_outcome', checks)
        source = next(record for record in records if record.entry.title == TITLE)
        with _endpoint(secret) as (url, requests, control):
            dialog.tabs.setCurrentIndex(2)
            dialog.provider_name.setText('合成资料的软件协议验收')
            dialog.provider_url.setText(url)
            dialog.provider_model.setText('requested-release-model')
            dialog.provider_timeout.setValue(4)
            dialog.credential_ref.setText('env:' + env_name)
            _click(dialog, 'knowledgeSaveProvider')
            config = load_provider(dialog.store)
            _check(config is not None and config.credential_ref == 'env:' + env_name,
                   'ui_provider_config_uses_explicit_reference', checks)
            dialog.tabs.setCurrentIndex(1)
            dialog.kind.setCurrentIndex(dialog.kind.findData('knowledge'))
            dialog.question.setPlainText('蜗杆润滑牌号')
            dialog.preview_button.click()
            _check(dialog.preview is not None and len(dialog.preview.evidence) == 1,
                   'imported_source_retrieved', checks)
            messages = json.loads(dialog.payload.toPlainText())
            evidence = dialog.preview.evidence[0]
            _check(evidence.source_id == source.source_id and evidence.version == 1,
                   'preview_has_stable_v1_source', checks)
            dialog.send_button.click()
            _wait(app, lambda: not dialog.busy, 'successful_http_completed')
            answer = dialog.last_answer
            _check(len(requests) == 1 and requests[0]['authenticated'] and
                   requests[0]['path'] == '/v1/chat/completions' and
                   requests[0]['payload']['messages'] == messages and
                   requests[0]['payload']['model'] == 'requested-release-model',
                   'authenticated_http_matches_preview', checks)
            _check(answer.status == 'ok' and answer.model == ACTUAL_MODEL and
                   answer.cited_ids == (evidence.evidence_id,), 'actual_model_and_valid_citation', checks)
            _check('evidence:' + evidence.evidence_id in dialog.answer.toHtml(),
                   'valid_citation_is_clickable', checks)
            report['windows_credentials'] = _windows_credentials(request, dialog, secret, messages)
            session_id = answer.session_id
            editor = dialog.edit_entry(source.source_id)
            editor.body.setPlainText(BODY_V2)
            editor.save_button.click()
            updated = dialog.store.get_source(source.source_id)
            _check(updated.version == 2 and updated.entry.body == BODY_V2 and
                   dialog.store.get_source(source.source_id, 1).entry.body == source.entry.body,
                   'ui_revision_keeps_v1_and_stable_id', checks)
            dialog.new_session()
            dialog.question.setPlainText('蜗杆润滑牌号')
            dialog.preview_button.click()
            _check(dialog.preview.evidence[0].version == 2 and 'LUB-V1' not in dialog.payload.toPlainText(),
                   'retrieval_uses_only_v2', checks)
            before = len(requests)
            control['delay'] = 2.
            dialog.send_button.click()
            dialog.send_button.click()
            _wait(app, lambda: len(requests) == before + 1, 'delayed_request_started')
            start = time.monotonic()
            dialog.cancel_button.click()
            _wait(app, lambda: not dialog.busy, 'cancel_completes_within_one_second', timeout=1)
            elapsed = time.monotonic() - start
            _check(elapsed < 1 and len(requests) == before + 1 and
                   dialog.last_answer.status == 'cancelled' and not dialog.answer.toPlainText() and
                   len(dialog.store.list_sessions()) == 1,
                   'cancelled_without_second_request_or_history', checks)
            report['cancel_seconds'] = round(elapsed, 4)
            before = len(requests)
            dialog.question.setPlainText('银河股市预算XYZ')
            dialog.preview_button.click()
            dialog.send_button.click()
            app.processEvents()
            _check(not dialog.send_button.isEnabled() and not dialog.preview.evidence and len(requests) == before,
                   'no_evidence_does_not_send', checks)
            context = ssl.create_default_context()
            _check(context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED,
                   'frozen_tls_context_initializes', checks)
            _check(all(_digest(Path(path)) == expected for path, expected in hashes.items()),
                   'import_sources_remain_readonly', checks)
            _check(all(secret.encode() not in path.read_bytes() for path in
                       (database, Path(str(database) + '-wal')) if path.exists()),
                   'database_contains_reference_not_secret', checks)
            state_path.write_text(json.dumps({'create_pid': os.getpid(), 'source_id': source.source_id,
                'source_hash_v1': source.content_hash, 'evidence_id': evidence.evidence_id,
                'session_id': session_id, 'credential_ref': config.credential_ref,
                'source_file_hashes': hashes}, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        _close(app, dialog)
        os.environ.pop(env_name, None)


def _reopen(app, workspace, report):
    checks = report['checks']
    state = json.loads((workspace / 'state.json').read_text(encoding='utf-8'))
    _check(state['create_pid'] != os.getpid(), 'independent_process_boundary', checks)
    database = workspace / 'release_knowledge.sqlite'
    dialog = _open(app, database)
    try:
        config = load_provider(dialog.store)
        _check(config is not None and config.credential_ref == state['credential_ref'] and
               config.model == 'requested-release-model', 'provider_reference_persists_without_secret', checks)
        _check(dialog.sources.count() == 3 and dialog.store.get_source(state['source_id']).version == 2,
               'sources_and_v2_persist_after_process_restart', checks)
        _check(dialog.sessions.count() == 1, 'previous_process_history_reopened', checks)
        dialog.tabs.setCurrentIndex(1)
        dialog.sessions.setCurrentRow(0)
        app.processEvents()
        _check(ACTUAL_MODEL in dialog.answer.toPlainText() and
               state['evidence_id'] in dialog.answer.toPlainText(), 'history_model_and_citation_restored', checks)
        previous = len(dialog._children)
        dialog.answer.anchorClicked.emit(QUrl('evidence:' + state['evidence_id']))
        app.processEvents()
        _check(len(dialog._children) == previous + 1, 'history_citation_opens_dialog', checks)
        source_dialog = dialog._children[-1]
        text = source_dialog.findChild(QTextBrowser).toPlainText()
        old = dialog.store.get_source(state['source_id'], 1)
        _check(BODY_V1 in text and BODY_V2 not in text and '旧版' in text and
               old.content_hash == state['source_hash_v1'], 'old_v1_citation_opens_original_text', checks)
        source_dialog.close()
        previous = len(dialog._children)
        dialog.answer.anchorClicked.emit(QUrl('evidence:not_sent_release_fixture'))
        _check(len(dialog._children) == previous and '不能跳转' in dialog.status.text(),
               'unknown_citation_cannot_open', checks)
        dialog.delete_history_button.click()
        _check(dialog.sessions.count() == 0 and not dialog.store.list_sessions(), 'history_deleted_via_ui', checks)
    finally:
        _close(app, dialog)
    dialog = _open(app, database)
    try:
        _check(dialog.sessions.count() == 0 and not dialog.store.history(state['session_id']) and
               dialog.sources.count() == 3, 'history_deletion_persists_after_reopen', checks)
        _check(all(_digest(Path(path)) == expected for path, expected in state['source_file_hashes'].items()),
               'source_files_unchanged_after_restart', checks)
    finally:
        _close(app, dialog)


def verify_knowledge(app, request_path: Path) -> int:
    """Run one phase; return 0/1 and write a small sanitized JSON report."""
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text(encoding='utf-8-sig'))
    if request.get('schema_version') != 1 or request.get('mode') not in ('create', 'reopen'):
        raise ValueError('知识诊断请求须为 schema_version=1、mode=create 或 reopen。')
    workspace = Path(request['workspace_dir'])
    if not workspace.is_absolute():
        workspace = request_path.parent / workspace
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    report_path = Path(request.get('report_path', workspace / (request['mode'] + '_validation.json')))
    if not report_path.is_absolute():
        report_path = request_path.parent / report_path
    report = {'schema_version': 1, 'mode': request['mode'], 'pid': os.getpid(),
        'frozen': bool(getattr(sys, 'frozen', False)), 'passed': False, 'checks': {},
        'scope': 'isolated synthetic sources and actual loopback HTTP; no real provider',
        'real_API_tested': False, 'real_user_content_tested': False,
        'windows_credentials': {'passed': False, 'status': 'not_requested'},
        'missing_real_conditions': ['user-confirmed knowledge and historical case',
            'user-configured real API; five sourced and two no-source questions with manual semantic review']}
    try:
        if request['mode'] == 'create':
            _create(app, request, workspace, report)
        else:
            _reopen(app, workspace, report)
        report['passed'] = True
    except DiagnosticFailure as error:
        report['failed_step'] = str(error)
    except Exception as error:
        # A protocol or OS error must not expose messages, headers or secrets.
        report['failed_step'] = 'unexpected_' + type(error).__name__
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if report['passed'] else 1
