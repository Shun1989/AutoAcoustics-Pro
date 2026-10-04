"""Supplemental source-adapter acceptance against untouched external curves.

The numerical contract is frozen before calculation in artifacts/loudness_extended_contract.json.
Technical recordings keep their entire source; eight source-proved partial final
2 ms intervals lack a Workbook target and are explicitly disclosed, never hidden.
"""
import hashlib
import json
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from autoacoustics.analysis.loudness import compute_loudness
from autoacoustics.calibration import apply_calibration
from autoacoustics.importers import import_signal
from autoacoustics.model import CalibrationProfile


ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / 'tests/reference/upstream/extended'
CONTRACT = EXT / 'acceptance_contract.json'
FROZEN_CONTRACT_SHA = 'f0f0750ce656ac149470a209eb64fc935a50eb082c766da5f8d39c84660bea1a'
FROZEN_CURVES_MANIFEST_SHA = '6721da3bd3352541107be6a1a1745ef1382a92fca484ea679baa609c0285edca'
RECEIPT = ROOT / 'artifacts/loudness_extended_validation.json'
IDS = (6, 8, 9, 11, 12, 13, *range(14, 26))
BARK = {6: 2.5, 8: 17.5, 9: 17.5, 11: 8.5, 12: 8.5, 13: 8.5}
ALLOWED_TAIL_IDS = {14, 15, 16, 17, 18, 23, 24, 25}
POINTS = {6: 5300, 8: 5300, 9: 5300, 11: 500, 12: 500, 13: 500,
          14: 6577, 15: 5768, 16: 2054, 17: 1454, 18: 1083, 19: 1300,
          20: 1300, 21: 1300, 22: 925, 23: 1275, 24: 1200, 25: 1275}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope='module')
def references():
    manifest = json.loads((EXT / 'curves_manifest.json').read_text(encoding='utf-8'))
    return {item['signal_id']: item for item in manifest['references']}


@pytest.fixture(scope='module')
def receipt():
    state = {'records': [], 'integrity_checks': {}}
    began = time.perf_counter()
    yield state
    records = state['records']
    data = {'schema_version': 1, 'scope': 'Supplemental formal source-adapter validation; not a frozen EXE run or complete ISO certification',
            'contract_sha256': sha(CONTRACT), 'curves_manifest_sha256': sha(EXT / 'curves_manifest.json'),
            'expected_signal_ids': list(IDS), 'integrity_checks': state['integrity_checks'], 'records': records,
            'elapsed_seconds': time.perf_counter() - began,
            'passed': len(state['integrity_checks']) == 3 and all(state['integrity_checks'].values())
                      and len(records) == len(IDS) and all(record.get('passed', False) for record in records)}
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def physical(record):
    source = EXT / record['input_file']
    assert sha(source) == record['input_sha256']
    signal = import_signal(source)
    assert signal.source_hash == record['input_sha256']
    assert signal.frames == record['input_frames'] and signal.sample_rate == 48000
    assert signal.metadata['subtype'] == 'PCM_16' and signal.channel_count == 1
    profile = CalibrationProfile('Extended upstream reference',
        'MoSQITo v1.2.1 fixed PCM16/32767*2sqrt(2) Pa scaling',
        2 * np.sqrt(2) * 32768 / 32767, 'FS', channel=0,
        applicable_hashes=(signal.source_hash,),
        details={'formula': 'PCM16/32767 * 2*sqrt(2) Pa', 'independent_absolute_calibration': False})
    return signal, apply_calibration(signal, profile)


def test_extended_mapping_and_frozen_source_contract(references, receipt):
    receipt['integrity_checks']['mapping_source_contract'] = False
    contract = json.loads(CONTRACT.read_text(encoding='utf-8'))
    assert sha(CONTRACT) == FROZEN_CONTRACT_SHA
    assert sha(EXT / 'curves_manifest.json') == FROZEN_CURVES_MANIFEST_SHA
    curves = json.loads((EXT / 'curves_manifest.json').read_text(encoding='utf-8'))
    assert curves['contract_sha256'] == sha(CONTRACT)
    assert curves['download_manifest_sha256'] == contract['source_download_manifest_sha256']
    assert tuple(references) == IDS
    assert contract['frozen_before_any_application_loudness_result'] is True
    assert set(contract['allowed_unreferenced_final_point_signal_ids']) == ALLOWED_TAIL_IDS
    assert contract['five_percent_outside_ratio_max'] == .01
    assert contract['ten_percent_outside_count_max'] == 0
    assert contract['time_absolute_tolerance_s'] == 1e-14
    assert contract['allowed_unreferenced_final_point_max'] == 1
    assert contract['full_input_recording'] and not contract['crop_input']
    assert not contract['normalize_input'] and not contract['time_shift_or_stretch']
    raw = json.loads((EXT / 'manifest.json').read_text(encoding='utf-8'))
    inputs = {int(item['signal_id']): item for item in raw['files'] if item['local_file'].endswith('.wav')}
    assert sha(EXT / 'manifest.json') == contract['source_download_manifest_sha256']
    assert raw['passed'] and all(item['git_blob_verified'] for item in raw['files'])
    for number, reference in references.items():
        assert reference['specific_bark'] == BARK.get(number)
        assert reference['field'] == ('diffuse' if number == 15 else 'free')
        assert reference['samples'] == POINTS[number]
        assert reference['input_sha256'] == contract['source_sha256'][reference['input_file']] == inputs[number]['sha256']
        assert reference['source_workbook_sha256'] == contract['source_sha256'][reference['source_workbook']]
        assert sha(EXT / reference['input_file']) == reference['input_sha256']
        assert sha(EXT / reference['reference_file']) == reference['reference_sha256']
        assert sha(EXT / reference['source_workbook']) == reference['source_workbook_sha256']
        if number >= 14:
            assert reference['technical_column_L_is_reference'] is False
            assert reference['source_columns'] == ['A', 'B', 'C', 'D', 'G', 'H']
    receipt['integrity_checks']['mapping_source_contract'] = True


def test_all_extended_reference_bounds_time_and_original_peaks(references, receipt):
    receipt['integrity_checks']['reference_bounds_time_peak'] = False
    for number, record in references.items():
        values = np.genfromtxt(EXT / record['reference_file'], delimiter=',', names=True)
        assert len(values) == POINTS[number]
        np.testing.assert_allclose(values['time_s'], np.arange(len(values)) * .002, atol=1e-14, rtol=0)
        assert abs(np.max(values['N_ref_sone']) - record['Nmax_sheet_cell_B5']) <= 1e-10
        for prefix in (['N', 'N_spec'] if number in BARK else ['N']):
            target = values['N_ref_sone' if prefix == 'N' else 'N_spec_ref_sone_per_bark']
            stack = np.array([values[prefix + '_lower_10'], values[prefix + '_lower_5'], target,
                              values[prefix + '_upper_5'], values[prefix + '_upper_10']])
            assert np.all(np.isfinite(stack)) and np.all(stack >= 0)
            assert np.all(np.diff(stack, axis=0) >= 0)
    receipt['integrity_checks']['reference_bounds_time_peak'] = True


def test_extended_pcm16_full_input_scale_is_exactly_upstream(references, receipt):
    receipt['integrity_checks']['full_pcm16_source_scale'] = False
    for record in references.values():
        signal, calibrated = physical(record)
        with wave.open(str(EXT / record['input_file']), 'rb') as source:
            assert source.getsampwidth() == 2 and source.getnchannels() == 1
            original = np.frombuffer(source.readframes(source.getnframes()), dtype='<i2')
        assert len(original) == signal.frames == len(calibrated.samples[0])
        expected = original.astype(float) / 32767 * (2 * np.sqrt(2))
        np.testing.assert_allclose(calibrated.samples[0], expected, rtol=2e-15, atol=1e-15)
    receipt['integrity_checks']['full_pcm16_source_scale'] = True


def curve_stats(actual, reference, prefix):
    finite = np.isfinite(actual)
    outside5 = ~finite | (actual < reference[prefix + '_lower_5']) | (actual > reference[prefix + '_upper_5'])
    outside10 = ~finite | (actual < reference[prefix + '_lower_10']) | (actual > reference[prefix + '_upper_10'])
    return {'compared_samples': len(actual), 'outside_5_count': int(np.count_nonzero(outside5)),
            'outside_5_ratio': float(np.mean(outside5)), 'outside_10_count': int(np.count_nonzero(outside10)),
            'outside_10_ratio': float(np.mean(outside10)), 'nonfinite_count': int(np.count_nonzero(~finite)),
            'passed': bool(np.mean(outside5) <= .01 and not np.any(outside10))}


@pytest.mark.parametrize('number', IDS, ids=lambda number: f'B4-{number}' if number in BARK else f'B5-{number}')
def test_extended_time_varying_external_curves(number, references, receipt):
    record = references[number]
    began = time.perf_counter()
    diagnostic = {'signal_id': number, 'input_file': record['input_file'],
                  'input_sha256': record['input_sha256'], 'reference_sha256': record['reference_sha256'],
                  'source_workbook_sha256': record['source_workbook_sha256'], 'field': record['field'],
                  'specific_bark': record['specific_bark'], 'input_frames': record['input_frames'],
                  'sample_rate_hz': record['sample_rate'], 'input_duration_s': record['input_frames'] / record['sample_rate'],
                  'reference_samples': record['samples'], 'technical_column_L_is_reference': False}
    receipt['records'].append(diagnostic)
    try:
        signal, calibrated = physical(record)
        result = compute_loudness(calibrated.samples[0], signal.sample_rate, field_type=record['field'],
                                  include_specific=number in BARK)
        reference = np.genfromtxt(EXT / record['reference_file'], delimiter=',', names=True)
        times = result['time']; n = result['total']; size = len(reference)
        tail = len(times) - size
        diagnostic.update({'computed_samples': len(times), 'computed_time_end_s': float(times[-1]),
                           'reference_time_end_s': float(reference['time_s'][-1]),
                           'computed_unreferenced_terminal_points': max(0, tail),
                           'uncompared_terminal': [{'time_s': float(t), 'N_sone': float(v)} for t, v in zip(times[size:], n[size:])],
                           'all_reference_samples_compared': len(times) >= size})
        length_ok = tail == (1 if number in ALLOWED_TAIL_IDS else 0)
        time_error = float(np.max(np.abs(times[:size] - reference['time_s']))) if len(times) >= size else None
        diagnostic['time_axis_max_error_s'] = time_error
        axis_ok = length_ok and time_error is not None and time_error <= 1e-14
        diagnostic['time_axis_passed'] = bool(axis_ok)
        expected_peak = record['Nmax_sheet_cell_B5']; actual_peak = float(np.max(n))
        peak_delta = actual_peak - expected_peak; peak_tolerance = max(.1, .05 * expected_peak)
        diagnostic.update({'Nmax_reference_sone': expected_peak, 'Nmax_actual_full_recording_sone': actual_peak,
                           'Nmax_delta_sone': peak_delta, 'Nmax_tolerance_sone': peak_tolerance,
                           'Nmax_passed': abs(peak_delta) <= peak_tolerance})
        if len(times) >= size:
            diagnostic['N'] = curve_stats(n[:size], reference, 'N')
            if number in BARK:
                bark_index = int(round(BARK[number] * 10)) - 1
                np.testing.assert_allclose(result['bark'][bark_index], BARK[number], atol=1e-12, rtol=0)
                diagnostic['N_specific'] = curve_stats(result['specific'][bark_index, :size], reference, 'N_spec')
        diagnostic['passed'] = bool(axis_ok and diagnostic['Nmax_passed'] and diagnostic.get('N', {}).get('passed', False)
                                    and diagnostic.get('N_specific', {'passed': True})['passed'])
        assert diagnostic['passed'], json.dumps(diagnostic, ensure_ascii=False)
    except Exception as error:
        diagnostic['passed'] = False
        diagnostic['failure'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        diagnostic['elapsed_seconds'] = time.perf_counter() - began
