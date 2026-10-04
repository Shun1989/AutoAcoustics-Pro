"""H5: compare actual product WAV/HEAD paths against frozen compatibility gates."""
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from autoacoustics.importers import import_signal
from autoacoustics.calibration import load_profile, apply_calibration
from autoacoustics.analysis.pipeline import analyze
from autoacoustics.model import AnalysisSettings


def main():
    started = time.perf_counter()
    manifest = json.loads((ROOT / 'docs/validation/corpus_manifest.json').read_text(encoding='utf-8'))
    archive = ROOT / '压缩.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['archive']['sha256']
    profile = load_profile(ROOT / 'profiles/head-pa-reference.json')
    entries = {(row['id'], row['kind']): row for row in manifest['members']}
    training = set(manifest['reference_fit']['training_ids'])
    settings = AnalysisSettings(channel=0, stationary_loudness=True)
    rows = []
    names = ('LZeq', 'LAeq', 'LAFmax', 'LASmax', 'Nstationary', 'Nmean', 'Nmax')
    for identity in manifest['paired_ids']:
        wav_entry, head_entry = entries[identity, 'wav'], entries[identity, 'hdf']
        wav = import_signal(ROOT / 'data/corpus' / wav_entry['path'])
        head = import_signal(ROOT / 'data/corpus' / head_entry['path'])
        assert wav.source_hash == wav_entry['sha256']
        assert head.source_hash == head_entry['sha256']
        assert wav.frames == head.frames and wav.sample_rate == head.sample_rate
        pressure = apply_calibration(wav, profile).samples[0]
        expected = head.samples[0]
        rms = math.sqrt(float(np.mean(expected ** 2)))
        row = {'id': identity, 'split': 'training' if identity in training else 'existing_holdout_regression',
               'wav_hash': wav.source_hash, 'hdf_hash': head.source_hash,
               'Pa_RMS_difference_dB': 20 * math.log10(math.sqrt(float(np.mean(pressure ** 2))) / rms),
               'waveform_NRMSE_percent': 100 * math.sqrt(float(np.mean((pressure - expected) ** 2))) / rms,
               'correlation': float(np.corrcoef(pressure, expected)[0, 1])}
        left = analyze(wav, profile, settings, include_detail=False)
        right = analyze(head, None, settings, include_detail=False)
        for name in names:
            a, b = left.summary.metrics[name].value, right.summary.metrics[name].value
            assert a is not None and b is not None and math.isfinite(a) and math.isfinite(b)
            row[f'{name}_wav'] = a
            row[f'{name}_hdf'] = b
            row[f'{name}_difference'] = a - b
            # Frozen H5 rules; never derive tolerances from the holdout results.
            tolerance = max(.02, .02 * b) if name.startswith('N') else .05
            assert abs(a - b) <= tolerance, (identity, name, a, b, tolerance)
        assert abs(row['Pa_RMS_difference_dB']) <= .05
        assert row['waveform_NRMSE_percent'] <= .2
        assert row['correlation'] >= .99999
        rows.append(row)
    extras = []
    for identity in manifest['unpaired_hdf_ids']:
        entry = entries[identity, 'hdf']
        head = import_signal(ROOT / 'data/corpus' / entry['path'])
        result = analyze(head, None, settings, include_detail=False)
        values = {name: result.summary.metrics[name].value for name in names}
        assert all(value is not None and math.isfinite(value) for value in values.values())
        extras.append({'id': identity, 'hash': head.source_hash, 'metrics': values, 'paired': False})
    output = ROOT / 'docs/validation'
    with (output / 'head_analysis_pairs.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    differences = {name: {'maximum_absolute': max(abs(row[f'{name}_difference']) for row in rows),
                         'median_absolute': float(np.median([abs(row[f'{name}_difference']) for row in rows]))}
                   for name in names}
    report = {'scope': 'formal product import/calibration/pipeline, existing paired corpus, not absolute or blind validation',
              'archive_hash': manifest['archive']['sha256'], 'profile_id': profile.profile_id,
              'training_count': len(training), 'holdout_count': len(rows) - len(training),
              'paired_count': len(rows), 'extra_count': len(extras), 'metric_differences': differences,
              'extras': extras, 'method_revision': right.provenance['method_revision'],
              'versions': dict(right.provenance['versions']), 'passed': True,
              'wall_seconds': time.perf_counter() - started}
    (output / 'head_analysis_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['archive']['sha256']
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
