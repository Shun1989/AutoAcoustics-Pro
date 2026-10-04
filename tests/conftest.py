"""Keep Qt alive while every GUI test and widget teardown runs."""
import os
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PyQt6.QtWidgets import QApplication


def pytest_addoption(parser):
    parser.addoption('--require-external-data', action='store_true',
                     help='Fail collection if a selected private/external data gate is unavailable.')


def pytest_configure(config):
    config.addinivalue_line('markers', 'private_corpus: requires local recordings and calibration receipts')
    config.addinivalue_line('markers', 'external_reference: requires independently acquired numerical reference material')


def pytest_collection_modifyitems(config, items):
    """Keep public regression runnable; absent external gates stay visibly skipped.

    This never supplies invented replacements or weakens numerical assertions.
    --require-external-data retains a strict local acceptance entry point.
    """
    root = Path(__file__).resolve().parents[1]
    private_modules = {'test_real_wav_import.py', 'test_real_head_hdf.py', 'test_real_calibration.py'}
    private_functions = {
        'test_head_acquisition.py': {
            'test_head_file_handoff_is_explicit_readonly_stable_and_uses_product_import',
            'test_head_file_requires_explicit_stop_and_rejects_incomplete_payload',
            'test_changing_device_file_is_not_published_as_complete',
            'test_simulated_managed_recorder_contract_is_labelled_and_has_no_realtime_samples',
            'test_simulated_managed_errors_never_claim_complete_files'},
        'test_acquisition_import.py': {
            'test_real_head_completed_manifest_is_importable_with_manifest_identity_and_provenance',
            'test_real_head_manifest_analyze_batch_project_reopen_keep_same_identity',
            'test_acquisition_manifest_refuses_mapping_that_would_relabel_recorded_units',
            'test_manifest_rejects_corrupted_data_and_recursive_json_data_reference'},
        'test_acquisition_ui.py': {'test_head_stopped_file_handoff_emits_only_verified_session'},
        'test_ui_end_to_end.py': {'test_historical_head_file_handoff_through_main_window_reopens_session'},
        'test_reference_manifest.py': {'test_real_corpus_has_64_aligned_triplets_and_56_regression_pairs'},
    }
    reference_manifests = {
        'test_loudness_reference.py': 'tests/reference/upstream/manifest.json',
        'test_loudness_extended_reference.py': 'tests/reference/upstream/extended/curves_manifest.json',
        'test_loudness_stationary_extended_reference.py': 'tests/reference/upstream/stationary_extended/manifest.json',
    }
    unavailable = set()
    for item in items:
        module = Path(str(item.path)).name
        function = getattr(item, 'originalname', None) or item.name.split('[')[0]
        private = module in private_modules or function in private_functions.get(module, set())
        if module == 'test_loudness_engine.py' and function == 'test_bounded_orchestration_matches_upstream':
            private = getattr(item, 'callspec', None) is not None and item.callspec.params.get('kind') == 'real'
        marker = reason = None
        if private:
            marker = 'private_corpus'
            if not (root / '压缩.zip').is_file():
                reason = 'Private recording corpus is not distributed in the open-source repository.'
            elif module in {'test_real_head_hdf.py', 'test_real_calibration.py', 'test_reference_manifest.py'} and not (root / 'docs/validation/corpus_manifest.json').is_file():
                reason = 'Private corpus provenance/calibration receipts are unavailable.'
        elif module in reference_manifests:
            marker = 'external_reference'
            if not (root / reference_manifests[module]).is_file():
                reason = 'External numerical reference attachments must be acquired separately with applicable rights.'
        if marker:
            item.add_marker(getattr(pytest.mark, marker))
        if reason:
            unavailable.add(reason)
            item.add_marker(pytest.mark.skip(reason=reason))
    if unavailable and config.getoption('--require-external-data'):
        raise pytest.UsageError('\n'.join(sorted(unavailable)))


@pytest.fixture(scope='session', autouse=True)
def qt_application_lifetime():
    application = QApplication.instance() or QApplication([])
    application.setQuitOnLastWindowClosed(False)
    yield application
    from autoacoustics.qt_lifecycle import dispose_gui
    dispose_gui(application)


@pytest.fixture(autouse=True)
def qt_widget_teardown(qt_application_lifetime):
    yield
    from autoacoustics.qt_lifecycle import dispose_gui
    dispose_gui(qt_application_lifetime)
