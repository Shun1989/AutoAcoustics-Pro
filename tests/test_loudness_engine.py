"""Equivalence tests for the memory bounded orchestration, not ISO truth tests.

Independent published ISO fixtures are validated by the public adapter suite.
"""
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly
from mosqito.sq_metrics import loudness_zwtv, loudness_zwst

from autoacoustics.analysis._loudness_engine import compute_loudness


def tone(fs=48000, seconds=0.12):
    t = np.arange(round(fs * seconds)) / fs
    return 0.0632455532033676 * np.sin(2 * np.pi * 1000 * t)


@pytest.mark.parametrize("kind", ["pulse", "amplitude_step", "silence", "real"])
@pytest.mark.parametrize("field", ["free", "diffuse"])
def test_bounded_orchestration_matches_upstream(kind, field):
    x = tone(seconds=0.18)
    if kind == "pulse":
        x[:2400] = 0
        x[2880:] = 0
    elif kind == "amplitude_step":
        x[:3000] *= 0.03
        x[5500:] *= 0.1
    elif kind == "silence":
        x[:] = 0
    elif kind == "real":
        with zipfile.ZipFile(Path(__file__).parents[1] / "压缩.zip") as archive:
            name = next(n for n in archive.namelist() if n.lower().endswith(".wav"))
            with archive.open(name) as stream:
                x, fs = sf.read(stream)
            assert fs == 48000
        x = x[:8640] * 0.806908103
    expected, specific, bark, _ = loudness_zwtv(x, 48000, field_type=field)
    actual = compute_loudness(x, 48000, field_type=field)
    np.testing.assert_allclose(actual["total"], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual["specific"], specific, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(actual["bark"], bark)
    np.testing.assert_array_equal(actual["time"], np.arange(len(expected)) * 0.002)


@pytest.mark.parametrize("fs,ratio", [(51200, (15, 16)), (96000, (1, 2)), (24000, (2, 1))])
def test_resampling_to_48k_preserves_original_clock(fs, ratio):
    x = tone(fs)
    expected, _, _, _ = loudness_zwtv(resample_poly(x, *ratio), 48000)
    actual = compute_loudness(x, fs)
    np.testing.assert_allclose(actual["total"], expected, rtol=1e-12, atol=1e-12)
    assert actual["metadata"]["original_sample_rate"] == fs
    assert actual["metadata"]["algorithm_sample_rate"] == 48000
    assert actual["metadata"]["resampling_ratio"] == list(ratio)
    assert bool(actual["metadata"]["warnings"]) == (fs < 48000)


def test_decimal_header_rate_is_only_rounded_within_representation_tolerance():
    declared = 1 / 2.08333333333333e-5
    a = compute_loudness(tone(), declared)
    b = compute_loudness(tone(), 48000)
    np.testing.assert_array_equal(a["total"], b["total"])
    assert a["metadata"]["original_sample_rate"] == declared
    assert a["metadata"]["sample_rate_rounding_applied"]
    np.testing.assert_allclose(a["time"], np.arange(len(a["total"])) * 96 / declared, rtol=0, atol=1e-16)


def test_stationary_is_upstream_model_and_not_temporal_smoothing():
    x = tone(seconds=0.25)
    expected, _, _ = loudness_zwst(x, 48000)
    actual = compute_loudness(x, 48000, stationary=True)
    assert actual["stationary_total"] == pytest.approx(expected, abs=1e-12)


def test_cancel_before_calculation_and_input_validation():
    with pytest.raises(InterruptedError):
        compute_loudness(tone(), 48000, cancel=lambda: True)
    for x, fs in [(np.array([]), 48000), (np.array([np.nan]), 48000), (tone(), 0)]:
        with pytest.raises(ValueError):
            compute_loudness(x, fs)


def test_block_boundary_does_not_reset_temporal_state(monkeypatch):
    import autoacoustics.analysis._loudness_engine as engine
    monkeypatch.setattr(engine, "_SAMPLE_CHUNK", 1001)
    monkeypatch.setattr(engine, "_CORE_CHUNK", 63)
    monkeypatch.setattr(engine, "_SPATIAL_CHUNK", 79)
    x = tone(seconds=0.14)
    x[2048:4096] *= 0.01
    expected, specific, _, _ = loudness_zwtv(x, 48000)
    actual = engine.compute_loudness(x, 48000)
    np.testing.assert_allclose(actual["total"], expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(actual["specific"], specific, atol=1e-12, rtol=1e-12)


def test_batch_summary_omits_specific_without_changing_total():
    x = tone(seconds=0.07)
    full = compute_loudness(x, 48000)
    compact = compute_loudness(x, 48000, include_specific=False)
    np.testing.assert_array_equal(full["total"], compact["total"])
    assert compact["specific"] is None
    assert not compact["metadata"]["specific_output_included"]


def test_input_is_not_modified_and_cancellation_inside_filtering():
    x = tone(seconds=0.03)
    before = x.copy()
    compute_loudness(x, 48000, stationary=True)
    np.testing.assert_array_equal(x, before)
    calls = 0

    def stop_during_filtering():
        nonlocal calls
        calls += 1
        return calls >= 7

    with pytest.raises(InterruptedError):
        compute_loudness(x, 48000, cancel=stop_during_filtering)
