"""Independent signal oracles for现场转频/阶次, never derived by the engine."""
import json

import numpy as np
import pytest

from autoacoustics.analysis.rotation import (
    RotationConfig, analyze_rotation, export_rotation_json,
    export_rotation_report, load_rpm_csv,
)


def test_manual_one_x_preserves_frequency_amplitude_and_declares_assumption():
    # Replacing rpm/60 by rpm or doubling peak amplitude must fail this oracle.
    fs = 12000
    t = np.arange(fs * 2) / fs
    result = analyze_rotation(.5 * np.sin(2 * np.pi * 50 * t), fs, RotationConfig(rpm=3000))
    peak = max(result.peaks, key=lambda p: p.level_db)
    assert peak.frequency == pytest.approx(50, abs=.6)
    assert peak.order == pytest.approx(1, abs=.02)
    assert peak.level_db == pytest.approx(-6.0206, abs=.05)
    assert peak.prominence_db > 30
    assert result.rpm_mode == 'manual'
    assert any('未测量' in n or '不是实测' in n for n in result.notes)
    assert any('1X' in ' '.join(f.evidence) for f in result.findings)
    assert all(f.checks and f.recommendations for f in result.findings)
    assert not result.orders.flags.writeable
    assert result.map_db.shape == (len(result.map_orders), len(result.map_times))


def test_gear_mesh_sidebands_and_actual_output_shaft_markers():
    fs = 12000
    t = np.arange(fs * 2) / fs
    # 20 teeth at 50 Hz input shaft gives GMF=1000, sidebands950/1050.
    x = np.sin(2*np.pi*1000*t) + .25*np.sin(2*np.pi*950*t) + .25*np.sin(2*np.pi*1050*t)
    result = analyze_rotation(x, fs, RotationConfig(rpm=3000, gear_teeth=20, gear_ratio=5, pole_pairs=2))
    markers = {m.label: m for m in result.markers}
    assert markers['GMF'].frequency == 1000
    assert markers['输出轴 1X'].frequency == 10
    assert markers['电磁 2pX'].frequency == 200
    findings = [f for f in result.findings if '齿轮' in f.mechanism]
    assert findings
    assert any('950' in ' '.join(f.evidence) and '1050' in ' '.join(f.evidence) for f in findings)


def test_measured_linear_runup_recovers_constant_order_in_true_order_map():
    # Integral of frot=20+10*t is20*t+5*t² revolutions. A fixed-freq relabel
    # smears this signal and cannot recover a horizontal3X band.
    fs = 8000
    t = np.arange(fs * 4) / fs
    phase_revolutions = 20*t + 5*t*t
    x = .8*np.sin(2*np.pi*3*phase_revolutions)
    result = analyze_rotation(x, fs, RotationConfig(max_order=20),
                              rpm_times=[0, 4], rpm_values=[1200, 3600])
    strongest = max(result.peaks, key=lambda p: p.level_db)
    assert strongest.order == pytest.approx(3, abs=.02)
    assert strongest.level_db == pytest.approx(20*np.log10(.8), abs=.7)
    maxima = result.map_orders[np.argmax(result.map_db, axis=0)]
    assert len(maxima) > 5
    assert np.max(abs(maxima-3)) < .13
    assert result.rpm_mode == 'measured'
    assert result.rpm_values[0] == 1200
    assert result.rpm_values[-1] == pytest.approx(3599.925)
    assert strongest.frequency_kind == 'reference_equivalent'
    assert 0 < result.map_times[0] < result.map_times[-1] < 4


def test_measured_rpm_uses_absolute_event_time_and_rejects_uncovered_curve():
    fs = 4000
    t = np.arange(fs)/fs
    result = analyze_rotation(np.sin(2*np.pi*40*t), fs, RotationConfig(),
                              rpm_times=[10, 11], rpm_values=[2400, 2400], time_origin=10)
    assert result.map_times.min() >= 10
    assert result.rpm_times[0] == 10
    assert result.frequency_reference_rpm == pytest.approx(2400)
    assert max(result.peaks, key=lambda p: p.level_db).order == pytest.approx(1, abs=.02)
    with pytest.raises(ValueError, match='覆盖'):
        analyze_rotation(np.ones(fs), fs, RotationConfig(),
                         rpm_times=[0, .8], rpm_values=[2400, 2400])


@pytest.mark.parametrize('times,rpm', [([0, 1, 1], [100, 100, 100]),
                                    ([0, 1], [100, 0]),
                                    ([0, 1], [100, np.nan])])
def test_invalid_rpm_never_silently_sorts_or_extrapolates(times, rpm):
    with pytest.raises(ValueError):
        analyze_rotation(np.ones(4000), 4000, RotationConfig(), rpm_times=times, rpm_values=rpm)


def test_angular_downsampling_rejects_high_frequency_alias():
    # With50Hz speed and maxorder10, an unfiltered2560Hz component can
    # alias into3.2X on64angle samples/revolution. No false low-order line.
    fs = 12000
    t = np.arange(fs*2)/fs
    signal = .2*np.sin(2*np.pi*100*t) + np.sin(2*np.pi*3060*t)
    result = analyze_rotation(signal, fs, RotationConfig(max_order=10),
                              rpm_times=[0, 2], rpm_values=[3000, 3000])
    strongest = max(result.peaks, key=lambda p: p.level_db)
    assert strongest.order == pytest.approx(2, abs=.02)
    outside = result.order_db[(result.orders > .5) & (abs(result.orders-2) > .15)]
    assert np.max(outside) < -45


def test_requested_order_above_original_sample_nyquist_is_truncated():
    fs = 1000
    t = np.arange(fs*2)/fs
    result = analyze_rotation(np.sin(2*np.pi*100*t), fs, RotationConfig(rpm=6000, max_order=50),
                              rpm_times=[0, 2], rpm_values=[6000, 6000])
    assert result.effective_max_order <= 5
    assert result.orders[-1] <= result.effective_max_order
    assert any('截断' in n for n in result.notes)


@pytest.mark.parametrize('noise', [False, True])
def test_silence_or_white_noise_does_not_generate_fault_candidates(noise):
    samples = np.random.default_rng(7811).normal(0, .1, 48000) if noise else np.zeros(48000)
    result = analyze_rotation(samples, 12000, RotationConfig())
    assert not result.findings
    if not noise:
        assert not result.peaks
        assert np.isneginf(result.spectrum_db).all()


@pytest.mark.parametrize('config', [RotationConfig(rpm=0), RotationConfig(gear_ratio=0),
                                 RotationConfig(gear_teeth=-1), RotationConfig(max_order=0),
                                 RotationConfig(pole_pairs=1.5)])
def test_invalid_physical_configuration_fails_before_diagnostics(config):
    with pytest.raises(ValueError):
        analyze_rotation(np.ones(1000), 1000, config)


def test_short_manual_event_cannot_confuse_unresolved_one_and_two_x():
    # Below two shaft turns, Fourier resolution cannot distinguish1X/2X.
    # A broad bin must not be called several different mechanical candidates.
    fs = 12000
    t = np.arange(100)/fs
    with pytest.raises(ValueError, match='2 转'):
        analyze_rotation(np.sin(2*np.pi*100*t), fs, RotationConfig(rpm=3000))


def test_few_revolutions_are_plottable_but_do_not_claim_resolved_mechanisms():
    fs = 12000
    t = np.arange(720)/fs
    result = analyze_rotation(np.sin(2*np.pi*50*t), fs, RotationConfig(rpm=3000))
    assert len(result.orders) > 2
    assert not result.findings
    assert any('分辨率' in note for note in result.notes)


def test_large_gear_teeth_does_not_mislabel_adjacent_sideband_as_gmf():
    fs = 20000
    t = np.arange(fs*2)/fs
    result = analyze_rotation(np.sin(2*np.pi*6050*t), fs, RotationConfig(rpm=3000, gear_teeth=120, max_order=150))
    assert not any('齿轮啮合' in finding.mechanism for finding in result.findings)


def test_rpm_csv_has_explicit_units_and_strict_original_order(tmp_path):
    good = tmp_path/'rpm.csv'
    good.write_text('time_s,rpm\n0,1200\n1,2400\n', encoding='utf-8-sig')
    times, rpm = load_rpm_csv(good)
    assert times.tolist() == [0, 1]
    assert rpm.tolist() == [1200, 2400]
    good.write_text('time_s,rpm\n1,2400\n0,1200\n', encoding='utf-8')
    with pytest.raises(ValueError):
        load_rpm_csv(good)
    good.write_text('time_ms,rpm\n0,1200\n1000,2400\n', encoding='utf-8')
    with pytest.raises(ValueError, match='time_s'):
        load_rpm_csv(good)


def test_report_and_json_preserve_source_parameters_and_candidate_evidence(tmp_path):
    from docx import Document
    t = np.arange(12000)/6000
    result = analyze_rotation(.4*np.sin(2*np.pi*50*t), 6000, RotationConfig(rpm=3000))
    report = export_rotation_report(result, tmp_path/'现场诊断.docx', source_path='真实录音.wav', source_hash='abc123')
    document = Document(report)
    content = '\n'.join(p.text for p in document.paragraphs)
    content += '\n' + '\n'.join(c.text for tb in document.tables for row in tb.rows for c in row.cells)
    assert '真实录音.wav' in content
    assert 'abc123' in content
    assert '3000' in content
    assert '1X' in content
    assert '候选' in content
    assert len(document.inline_shapes) >= 1
    export_rotation_json(result, tmp_path/'复核.json', source_path='真实录音.wav', source_hash='abc123')
    payload = json.loads((tmp_path/'复核.json').read_text(encoding='utf-8'))
    assert payload['source_hash'] == 'abc123'
    assert payload['config']['rpm'] == 3000
    assert payload['findings'][0]['checks']
    assert payload['map_db']


def test_silent_json_has_null_db_values_instead_of_nonstandard_infinity(tmp_path):
    result = analyze_rotation(np.zeros(6000), 6000, RotationConfig())
    destination = export_rotation_json(result, tmp_path/'silent.json')
    payload = json.loads(destination.read_text(encoding='utf-8'), parse_constant=lambda x: pytest.fail(x))
    assert all(v is None for v in payload['spectrum_db'])


def test_numpy_configuration_scalars_export_as_standard_json_numbers(tmp_path):
    t = np.arange(6000)/6000
    config = RotationConfig(rpm=np.float64(3000), gear_teeth=np.int64(20))
    result = analyze_rotation(np.sin(2*np.pi*50*t), 6000, config)
    path = export_rotation_json(result, tmp_path/'numpy.json')
    assert json.loads(path.read_text(encoding='utf-8'))['config']['gear_teeth'] == 20


def test_report_refuses_changed_settings_without_reanalysis(tmp_path):
    t = np.arange(6000)/6000
    result = analyze_rotation(np.sin(2*np.pi*50*t), 6000, RotationConfig())
    with pytest.raises(ValueError, match='不一致'):
        export_rotation_report(result, tmp_path/'stale.docx', config=RotationConfig(rpm=1200))

