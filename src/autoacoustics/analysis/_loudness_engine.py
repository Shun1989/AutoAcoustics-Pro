"""Memory bounded orchestration of MoSQITo 1.2.1 ISO 532-1 algorithms.

The filter tables and nonlinear recurrence below are reproduced from MoSQITo
1.2.1 (Eomys and contributors, Apache-2.0). This changes storage/iteration,
not the loudness model. Core loudness and Bark integration remain upstream
functions. Nonlinear, filter, and temporal states are never reset at blocks.
See tests/reference/upstream/LICENSE.mosqito.txt and the distribution notices.
"""
from __future__ import annotations

import math
from fractions import Fraction
from importlib.metadata import version

import numpy as np
from scipy.signal import lfilter, resample_poly, sosfilt
from mosqito.sq_metrics import loudness_zwst
from mosqito.sq_metrics.loudness.loudness_zwst._main_loudness import _main_loudness
from mosqito.sq_metrics.loudness.loudness_zwst._calc_slopes import _calc_slopes

try:
    from numba import njit
except ImportError:
    njit = None

_SAMPLE_CHUNK = 96000
_CORE_CHUNK = 4096
_SPATIAL_CHUNK = 4096
_NL_CHUNK = 4096

# Exact tables copied from pinned upstream _third_octave_levels.py.
_THIRD_FILTER_REF = np.array(
    [[1, 2, 1, 1, -2, 1], [1, 0, -1, 1, -2, 1], [1, -2, 1, 1, -2, 1]],
    dtype=np.float64,
)
_THIRD_FILTER = np.array(
    [[[0, 0, 0, 0, -0.00067026, 0.000659453], [0, 0, 0, 0, -0.000375071, 0.000361926],
      [0, 0, 0, 0, -0.000306523, 0.000297634]],
     [[0, 0, 0, 0, -0.000847258, 0.000830131], [0, 0, 0, 0, -0.000476448, 0.000455616],
      [0, 0, 0, 0, -0.000388773, 0.000374685]],
     [[0, 0, 0, 0, -0.0010721, 0.00104496], [0, 0, 0, 0, -0.000606567, 0.000573553],
      [0, 0, 0, 0, -0.000494004, 0.000471677]],
     [[0, 0, 0, 0, -0.00135836, 0.00131535], [0, 0, 0, 0, -0.000774327, 0.000722007],
      [0, 0, 0, 0, -0.000629154, 0.000593771]],
     [[0, 0, 0, 0, -0.0017238, 0.00165564], [0, 0, 0, 0, -0.00099178, 0.000908866],
      [0, 0, 0, 0, -0.000803529, 0.000747455]],
     [[0, 0, 0, 0, -0.00219188, 0.00208388], [0, 0, 0, 0, -0.00127545, 0.00114406],
      [0, 0, 0, 0, -0.00102976, 0.0009409]],
     [[0, 0, 0, 0, -0.00279386, 0.00262274], [0, 0, 0, 0, -0.00164828, 0.00144006],
      [0, 0, 0, 0, -0.0013252, 0.00118438]],
     [[0, 0, 0, 0, -0.00357182, 0.00330071], [0, 0, 0, 0, -0.00214252, 0.00181258],
      [0, 0, 0, 0, -0.00171397, 0.00149082]],
     [[0, 0, 0, 0, -0.00458305, 0.00415355], [0, 0, 0, 0, -0.00280413, 0.00228135],
      [0, 0, 0, 0, -0.00223006, 0.00187646]],
     [[0, 0, 0, 0, -0.00590655, 0.00522622], [0, 0, 0, 0, -0.00369947, 0.00287118],
      [0, 0, 0, 0, -0.00292205, 0.00236178]],
     [[0, 0, 0, 0, -0.00765243, 0.00657493], [0, 0, 0, 0, -0.0049254, 0.00361318],
      [0, 0, 0, 0, -0.00386007, 0.0029724]],
     [[0, 0, 0, 0, -0.0100023, 0.0082961], [0, 0, 0, 0, -0.00663788, 0.00455999],
      [0, 0, 0, 0, -0.00515982, 0.00375306]],
     [[0, 0, 0, 0, -0.013123, 0.010422], [0, 0, 0, 0, -0.00902274, 0.00573132],
      [0, 0, 0, 0, -0.00694543, 0.00471734]],
     [[0, 0, 0, 0, -0.0173693, 0.0130947], [0, 0, 0, 0, -0.0124176, 0.00720526],
      [0, 0, 0, 0, -0.00946002, 0.00593145]],
     [[0, 0, 0, 0, -0.0231934, 0.0164308], [0, 0, 0, 0, -0.0173009, 0.00904761],
      [0, 0, 0, 0, -0.0130358, 0.00744926]],
     [[0, 0, 0, 0, -0.0313292, 0.020637], [0, 0, 0, 0, -0.0244342, 0.0113731],
      [0, 0, 0, 0, -0.0182108, 0.00936778]],
     [[0, 0, 0, 0, -0.0428261, 0.0259325], [0, 0, 0, 0, -0.0349619, 0.0143046],
      [0, 0, 0, 0, -0.0257855, 0.0117912]],
     [[0, 0, 0, 0, -0.0591733, 0.0325054], [0, 0, 0, 0, -0.0506072, 0.0179513],
      [0, 0, 0, 0, -0.0369401, 0.0148094]],
     [[0, 0, 0, 0, -0.0826348, 0.0405894], [0, 0, 0, 0, -0.0740348, 0.0224476],
      [0, 0, 0, 0, -0.0534977, 0.0185371]],
     [[0, 0, 0, 0, -0.117018, 0.0508116], [0, 0, 0, 0, -0.109516, 0.0281387],
      [0, 0, 0, 0, -0.0785097, 0.0232872]],
     [[0, 0, 0, 0, -0.167714, 0.0637872], [0, 0, 0, 0, -0.163378, 0.0353729],
      [0, 0, 0, 0, -0.116419, 0.0293723]],
     [[0, 0, 0, 0, -0.242528, 0.0798576], [0, 0, 0, 0, -0.245161, 0.044337],
      [0, 0, 0, 0, -0.173972, 0.0370015]],
     [[0, 0, 0, 0, -0.353142, 0.099633], [0, 0, 0, 0, -0.369163, 0.0553535],
      [0, 0, 0, 0, -0.261399, 0.0465428]],
     [[0, 0, 0, 0, -0.516316, 0.124177], [0, 0, 0, 0, -0.555473, 0.0689403],
      [0, 0, 0, 0, -0.393998, 0.0586715]],
     [[0, 0, 0, 0, -0.756635, 0.155023], [0, 0, 0, 0, -0.834281, 0.0858123],
      [0, 0, 0, 0, -0.594547, 0.074396]],
     [[0, 0, 0, 0, -1.10165, 0.191713], [0, 0, 0, 0, -1.23939, 0.105243],
      [0, 0, 0, 0, -0.891666, 0.0940354]],
     [[0, 0, 0, 0, -1.58477, 0.239049], [0, 0, 0, 0, -1.80505, 0.128794],
      [0, 0, 0, 0, -1.325, 0.121333]],
     [[0, 0, 0, 0, -2.5063, 0.142308], [0, 0, 0, 0, -2.19464, 0.27647],
      [0, 0, 0, 0, -1.90231, 0.147304]]],
    dtype=np.float64,
)
_FILTER_GAIN = np.array(
    [4.30764e-11, 8.5934e-11, 1.71424e-10, 3.41944e-10, 6.82035e-10, 1.36026e-09,
     2.71261e-09, 5.4087e-09, 1.07826e-08, 2.1491e-08, 4.28228e-08, 8.54316e-08,
     1.70009e-07, 3.38215e-07, 6.7199e-07, 1.33531e-06, 2.65172e-06, 5.25477e-06,
     1.0378e-05, 2.0487e-05, 4.05198e-05, 7.97914e-05, 0.000156511, 0.000304954,
     0.000599157, 0.00116544, 0.00227488, 0.00391006],
    dtype=np.float64,
)


def _check_cancel(cancel):
    if cancel is not None:
        stopped = cancel() if callable(cancel) else cancel.is_set()
        if stopped:
            raise InterruptedError("Loudness calculation cancelled")


def _third_levels_bounded(samples, cancel):
    count = (len(samples) + 23) // 24
    levels = np.empty((28, count), dtype=np.float64)
    for band in range(28):
        _check_cancel(cancel)
        coeff = _THIRD_FILTER_REF - _THIRD_FILTER[band]
        sos_state = np.zeros((3, 2))
        lp_states = [np.zeros(1) for _ in range(3)]
        center = 10 ** ((band - 16) / 10) * 1000
        tau = 2 / (3 * min(center, 1000))
        a1 = np.exp(-1 / (48000 * tau))
        b0 = 1 - a1
        for start in range(0, len(samples), _SAMPLE_CHUNK):
            _check_cancel(cancel)
            stop = min(start + _SAMPLE_CHUNK, len(samples))
            filtered, sos_state = sosfilt(coeff, samples[start:stop], zi=sos_state)
            filtered *= _FILTER_GAIN[band]
            filtered **= 2
            for stage in range(3):
                filtered, lp_states[stage] = lfilter(
                    [b0], [1, -a1], filtered, zi=lp_states[stage]
                )
            offset = (-start) % 24
            first = (start + offset) // 24
            decimated = filtered[offset::24]
            levels[band, first:first + len(decimated)] = 10 * np.log10(
                (decimated + 1e-12) / 4e-10
            )
    return levels


def _nonlinear_chunk(core, next_core, previous_o, previous_u2, constants):
    """Same six-coefficient/24-interpolation recurrence as upstream.

    Per-band order is unchanged. Last-core lookahead must be zero exactly as
    upstream. Returning capacitor states prevents any block-boundary reset.
    """
    output = np.empty_like(core)
    for band in range(core.shape[0]):
        o_last = previous_o[band]
        u2_last = previous_u2[band]
        for col in range(core.shape[1]):
            ui = core[band, col]
            following = core[band, col + 1] if col + 1 < core.shape[1] else next_core[band]
            delta = (following - ui) / 24
            for interpolation in range(24):
                o = ui
                candidate = o_last * constants[2] - u2_last * constants[3]
                if o_last > u2_last and candidate >= ui:
                    o = candidate
                candidate = o_last * constants[4]
                if o_last <= u2_last and candidate >= ui:
                    o = candidate
                u2 = o
                candidate = o_last * constants[0] - u2_last * constants[1]
                if ui < o_last and o_last > u2_last and candidate <= o:
                    u2 = candidate
                candidate = (u2_last - ui) * constants[5] + ui
                if ui >= o_last and not (abs(ui - o_last) < 1e-5 and o <= u2_last):
                    u2 = candidate
                if interpolation == 0:
                    output[band, col] = o
                o_last = o
                u2_last = u2
                ui += delta
        previous_o[band] = o_last
        previous_u2[band] = u2_last
    return output


_nonlinear_compiled = njit(cache=False, fastmath=False)(_nonlinear_chunk) if njit else _nonlinear_chunk


def _nonlinear_bounded(core, cancel):
    t_short, t_long, t_var = 0.005, 0.015, 0.075
    delta_t = 1 / (2000 * 24)
    p = (t_var + t_long) / (t_var * t_short)
    q = 1 / (t_short * t_var)
    l1 = -p / 2 + math.sqrt(p * p / 4 - q)
    l2 = -p / 2 - math.sqrt(p * p / 4 - q)
    den = t_var * (l1 - l2)
    e1, e2 = math.exp(l1 * delta_t), math.exp(l2 * delta_t)
    constants = np.array([
        (e1 - e2) / den,
        ((t_var * l2 + 1) * e1 - (t_var * l1 + 1) * e2) / den,
        ((t_var * l1 + 1) * e1 - (t_var * l2 + 1) * e2) / den,
        (t_var * l1 + 1) * (t_var * l2 + 1) * (e1 - e2) / den,
        math.exp(-delta_t / t_long), math.exp(-delta_t / t_var),
    ])
    # Upstream starts its loop at col=0, reading uo[:, -1] of the expanded
    # interpolation array and u2[:, -1]=0. Preserve that initialization,
    # including the repeated additions (not a reassociated multiply).
    previous_o = core[:, -1].copy()
    last_delta = -previous_o / 24
    for _ in range(23):
        previous_o += last_delta
    previous_u2 = np.zeros(core.shape[0])
    result = np.empty_like(core)
    for start in range(0, core.shape[1], _NL_CHUNK):
        _check_cancel(cancel)
        stop = min(start + _NL_CHUNK, core.shape[1])
        next_core = core[:, stop] if stop < core.shape[1] else np.zeros(core.shape[0])
        result[:, start:stop] = _nonlinear_compiled(
            core[:, start:stop], next_core, previous_o, previous_u2, constants
        )
    return result


def _temporal_bounded(total, cancel):
    outputs = []
    for tau in (0.0035, 0.070):
        a1 = math.exp(-1 / (2000 * 24 * tau))
        b0 = 1 - a1
        state = np.zeros(1)
        result = np.empty_like(total)
        for start in range(0, len(total), _CORE_CHUNK):
            _check_cancel(cancel)
            stop = min(start + _CORE_CHUNK, len(total))
            chunk = total[start:stop]
            following = np.empty_like(chunk)
            following[:-1] = chunk[1:]
            following[-1] = total[stop] if stop < len(total) else 0
            delta = (following - chunk) / 24
            interpolated = np.empty((len(chunk), 24))
            interpolated[:, 0] = chunk
            for i in range(1, 24):
                interpolated[:, i] = interpolated[:, i - 1] + delta
            filtered, state = lfilter([b0], [1, -a1], interpolated.ravel(), zi=state)
            result[start:stop] = filtered.reshape(-1, 24)[:, 0]
        outputs.append(result)
    return 0.47 * outputs[0] + 0.53 * outputs[1]


def compute_loudness(samples_pa: np.ndarray, sample_rate: float,
                     field_type="free", stationary=False, cancel=None,
                     include_specific=True) -> dict:
    """Calculate continuous loudness, retaining full 2 ms/Bark output.

    Input is one physical acoustic channel in Pa. No SPL Fast/Slow weighting,
    DC removal, event truncation, or signal padding is added here. The caller
    computes the original full context and selects event output afterwards.
    Cancellation is polled between bounded stages; process isolation is still
    required to interrupt initial compilation or external resampling promptly.
    """
    _check_cancel(cancel)
    original = np.asarray(samples_pa, dtype=np.float64)
    fs = float(sample_rate)
    if original.ndim != 1 or not len(original):
        raise ValueError("Loudness requires a nonempty one-dimensional Pa signal")
    if not np.isfinite(fs) or fs <= 0 or not np.isfinite(original).all():
        raise ValueError("Loudness samples and positive sample rate must be finite")
    if field_type not in ("free", "diffuse"):
        raise ValueError("Sound field must be free or diffuse")
    if version("mosqito") != "1.2.1":
        raise RuntimeError("Memory-bounded orchestration is verified only with MoSQITo 1.2.1")
    tolerance = max(1e-6, abs(fs) * 1e-10)
    rounded = abs(fs - 48000) <= tolerance
    warnings = []
    if rounded:
        prepared = original
        up, down = 1, 1
    else:
        ratio = Fraction(48000 / fs).limit_denominator(100000)
        up, down = ratio.numerator, ratio.denominator
        prepared = resample_poly(original, up, down)
        if fs < 48000:
            warnings.append("Input bandwidth below 48 kHz requirement: loudness is an estimate; resampling does not restore missing bands.")
        if abs(fs * up / down - 48000) > tolerance:
            warnings.append("Rational resampling ratio approximates the declared sample rate; output retains the actual mapped clock.")
    _check_cancel(cancel)
    levels = _third_levels_bounded(prepared, cancel)
    core = np.empty((21, levels.shape[1]))
    for start in range(0, levels.shape[1], _CORE_CHUNK):
        _check_cancel(cancel)
        stop = min(start + _CORE_CHUNK, levels.shape[1])
        core[:, start:stop] = np.asarray(_main_loudness(levels[:, start:stop], field_type)).reshape(21, -1)
    del levels
    nonlinear = _nonlinear_bounded(core, cancel)
    del core
    ntime = nonlinear.shape[1]
    total = np.empty(ntime)
    specific = np.empty((240, (ntime + 3) // 4), dtype=np.float64) if include_specific else None
    for start in range(0, ntime, _SPATIAL_CHUNK):
        _check_cancel(cancel)
        stop = min(start + _SPATIAL_CHUNK, ntime)
        block_total, block_specific = _calc_slopes(nonlinear[:, start:stop])
        total[start:stop] = np.asarray(block_total).reshape(-1)
        if include_specific:
            offset = (-start) % 4
            selected = np.asarray(block_specific).reshape(240, -1)[:, offset::4]
            first = (start + offset) // 4
            specific[:, first:first + selected.shape[1]] = selected
    del nonlinear
    weighted = _temporal_bounded(total, cancel)[::4].copy()
    _check_cancel(cancel)
    stationary_total = None
    stationary_fs = None
    if stationary:
        if fs >= 48000 and not rounded:
            stationary_fs = fs
            stationary_total = float(loudness_zwst(original, fs, field_type)[0])
        else:
            stationary_fs = 48000
            stationary_total = float(loudness_zwst(prepared, 48000, field_type)[0])
    mapped_step = 96 * down / up / fs
    if njit is None:
        warnings.append("Compiled recurrence unavailable: memory bounded Python fallback may be slow on long recordings.")
    return {
        "time": np.arange(len(weighted), dtype=np.float64) * mapped_step,
        "total": weighted,
        "specific": specific,
        "bark": np.linspace(0.1, 24, 240),
        "stationary_total": stationary_total,
        "metadata": {
            "original_sample_rate": fs,
            "algorithm_sample_rate": 48000,
            "stationary_algorithm_sample_rate": stationary_fs,
            "effective_resampled_rate": fs * up / down,
            "sample_rate_rounding_applied": rounded and fs != 48000,
            "sample_rate_rounding_tolerance_hz": tolerance,
            "resampling_ratio": [up, down],
            "time_axis_rule": "k * 96 * down / up / original_sample_rate; no linspace endpoint; no causal delay removal",
            "algorithm_time_step_seconds": 0.002,
            "mapped_time_step_seconds": mapped_step,
            "backend": "MoSQITo 1.2.1 memory-bounded orchestration",
            "compiled_nonlinear_recurrence": njit is not None,
            "specific_output_included": bool(include_specific),
            "nonlinear_initialization": "upstream expanded-array wrap initialization preserved",
            "warnings": warnings,
        },
    }
