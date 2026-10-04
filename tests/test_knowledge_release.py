"""Release diagnostics must cross processes, not only reopen one connection."""
import json
import os
from pathlib import Path
import subprocess
import sys


def _run(request):
    repo = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment['QT_QPA_PLATFORM'] = 'offscreen'
    environment['PYTHONPATH'] = str(repo / 'src')
    code = ('from pathlib import Path; import sys; '
            'from PyQt6.QtWidgets import QApplication; '
            'from autoacoustics.knowledge_validation import verify_knowledge; '
            'app=QApplication([]); '
            'raise SystemExit(verify_knowledge(app,Path(sys.argv[1])))')
    return subprocess.run([sys.executable, '-c', code, str(request)],
                          env=environment, capture_output=True, timeout=30)


def test_release_csv_json_authenticated_http_and_history_across_processes(tmp_path):
    workspace = tmp_path / 'isolated'
    reports = []
    for mode in ('create', 'reopen'):
        request = tmp_path / (mode + '.json')
        report = tmp_path / (mode + '_validation.json')
        request.write_text(json.dumps({'schema_version': 1, 'mode': mode,
            'workspace_dir': str(workspace), 'report_path': str(report)}), encoding='utf-8')
        completed = _run(request)
        assert completed.returncode == 0, completed.stderr.decode(errors='replace')
        data = json.loads(report.read_text(encoding='utf-8'))
        assert data['passed'] and not data['real_API_tested']
        assert not data['real_user_content_tested']
        assert all(data['checks'].values())
        assert data['windows_credentials']['status'] == 'not_requested'
        reports.append(data)
    assert reports[0]['pid'] != reports[1]['pid']
    assert reports[0]['checks']['csv_explicit_mapping_and_partial_error']
    assert reports[0]['checks']['json_declared_fields_imported']
    assert reports[0]['checks']['authenticated_http_matches_preview']
    assert reports[0]['checks']['cancelled_without_second_request_or_history']
    assert reports[1]['checks']['previous_process_history_reopened']
    assert reports[1]['checks']['old_v1_citation_opens_original_text']
    assert reports[1]['checks']['history_deletion_persists_after_reopen']


def test_release_create_refuses_existing_database_without_overwriting(tmp_path):
    workspace = tmp_path / 'existing'
    workspace.mkdir()
    database = workspace / 'release_knowledge.sqlite'
    original = b'preserve existing data'
    database.write_bytes(original)
    report = tmp_path / 'refused.json'
    request = tmp_path / 'request.json'
    request.write_text(json.dumps({'schema_version': 1, 'mode': 'create',
        'workspace_dir': str(workspace), 'report_path': str(report)}), encoding='utf-8')
    completed = _run(request)
    assert completed.returncode == 1
    assert database.read_bytes() == original
    result = json.loads(report.read_text(encoding='utf-8'))
    assert not result['passed'] and result['failed_step'] == 'empty_diagnostic_database'
