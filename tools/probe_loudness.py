"""Measure actual RSS/time in isolated fresh Windows-compatible processes.

No long upstream call is made: that is the known resource hazard. The bounded
engine retains full specific output by default; --summary omits that output.
Each channel is selected and calculated in sequence, not averaged together.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import platform
import sys
import time
import traceback

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def child_probe(seconds, channels, summary, queue):
    try:
        import numpy as np
        from autoacoustics.analysis._loudness_engine import compute_loudness

        fs = 48000
        count = round(fs * seconds)
        signal = np.empty((channels, count), dtype=np.float64)
        for start in range(0, count, 480000):
            stop = min(start + 480000, count)
            t = np.arange(start, stop, dtype=np.float64) / fs
            # Small continuous background plus an intermittent motor-like
            # tonal event. Synthetic benchmark only, not real motor evidence.
            gain = np.where((t % 8) < 0.8, 0.1, 0.002)
            for channel in range(channels):
                signal[channel, start:stop] = gain * np.sin(2 * np.pi * (1000 + 50 * channel) * t)
        results = []
        for channel in range(channels):
            begin = time.perf_counter()
            result = compute_loudness(signal[channel], fs, include_specific=not summary)
            results.append({
                "channel": channel,
                "calculation_seconds": time.perf_counter() - begin,
                "output_frames": len(result["total"]),
                "mean_sone": float(np.mean(result["total"])),
                "max_sone": float(np.max(result["total"])),
                "all_finite": bool(np.isfinite(result["total"]).all()),
                "all_nonnegative": bool(np.all(result["total"] >= 0)),
                "specific_shape": list(result["specific"].shape) if result["specific"] is not None else None,
                "metadata": result["metadata"],
            })
            del result
        queue.put({"status": "completed", "results": results})
    except BaseException:
        queue.put({"status": "failed", "traceback": traceback.format_exc()})


def one_probe(seconds, channels, summary, peak_limit_gib):
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=child_probe, args=(seconds, channels, summary, queue))
    begin = time.perf_counter()
    process.start()
    observed = psutil.Process(process.pid)
    peak = 0
    stopped = False
    while process.is_alive():
        try:
            peak = max(peak, observed.memory_info().rss)
        except psutil.NoSuchProcess:
            break
        if peak > peak_limit_gib * 1024**3:
            process.terminate()
            stopped = True
            break
        time.sleep(0.02)
    process.join(10)
    result = queue.get(timeout=5) if not stopped and process.exitcode == 0 else {
        "status": "terminated_memory_limit" if stopped else "process_failed",
        "exitcode": process.exitcode,
    }
    result.update({
        "input_seconds": seconds,
        "channels": channels,
        "selected_channel_sequence": True,
        "specific_included": not summary,
        "wall_seconds": time.perf_counter() - begin,
        "peak_rss_bytes": peak,
        "peak_rss_gib": peak / 1024**3,
        "rss_sampling_interval_seconds": 0.02,
    })
    queue.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, nargs="+", default=[1, 6, 30, 60, 600])
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--peak-limit-gib", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/validation/loudness_resource_probe.json")
    args = parser.parse_args()
    if args.channels not in (1, 2) or any(value <= 0 for value in args.seconds):
        parser.error("positive durations and one or two channels required")
    from importlib.metadata import version
    report = {
        "schema": 1,
        "observed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": sys.version,
        "versions": {name: version(name) for name in ("numpy", "scipy", "mosqito", "numba", "psutil")},
        "host_ram_bytes": psutil.virtual_memory().total,
        "scope": "synthetic 48 kHz selected-channel sequential computation; process RSS includes imports and original input; fresh process for each duration; wall time includes imports and JIT",
        "engineering_budget": {"peak_rss_gib": 3.0, "per_channel_600s_calculation_seconds": 120.0},
        "probes": [],
    }
    if args.append and args.output.exists():
        report = json.loads(args.output.read_text(encoding="utf-8"))
    for seconds in args.seconds:
        result = one_probe(seconds, args.channels, args.summary, args.peak_limit_gib)
        report["probes"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(args.output)
        print(json.dumps({key: result[key] for key in ("input_seconds", "channels", "specific_included", "status", "wall_seconds", "peak_rss_gib")}), flush=True)
        if result["status"] != "completed":
            return 1
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
