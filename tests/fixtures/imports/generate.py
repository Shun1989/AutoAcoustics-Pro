"""Generate small, real encoded fixtures for standalone import probes.

These are synthetic known signals, not recordings or independent acoustic
reference data. Run with the project Python; FFmpeg is a fixture dependency.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np
from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter
from scipy.io import savemat
import soundfile as sf


DIRECTORY = Path(__file__).resolve().parent


def main():
    rate, frames = 48000, 24000
    t = np.arange(frames) / rate
    logical = np.asarray([0.25 * np.sin(2 * np.pi * 1000 * t), 0.125 * np.sin(2 * np.pi * 2500 * t)])
    cases = []

    def register(name, kind, units=("FS", "FS"), *, mapping=None, frame_tolerance=0, note=""):
        path = DIRECTORY / name
        cases.append({"path": name, "kind": kind, "sample_rate": rate, "channels": 2,
                      "source_frames": frames, "frame_tolerance": frame_tolerance, "units": list(units),
                      "mapping": mapping, "note": note, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    source = DIRECTORY / "样例.wav"
    sf.write(source, logical.T, rate, subtype="PCM_16")
    register(source.name, "wav", note="PCM16 normalized FS")
    sf.write(DIRECTORY / "24位.wav", logical.T, rate, subtype="PCM_24")
    register("24位.wav", "wav", note="PCM24 normalized FS")
    sf.write(DIRECTORY / "浮点.wav", logical.T, rate, subtype="FLOAT")
    register("浮点.wav", "wav", note="Float digital samples, ADC full scale unknown")
    sf.write(DIRECTORY / "样例.flac", logical.T, rate, subtype="PCM_24")
    register("样例.flac", "flac", note="Lossless PCM24")
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("ffmpeg is required to generate real codec fixtures")
    for suffix, codec, video in [("mp3", "libmp3lame", False), ("m4a", "aac", False),
                                 ("aac", "aac", False), ("ogg", "libvorbis", False),
                                 ("mp4", "aac", True), ("avi", "pcm_s16le", True),
                                 ("mkv", "pcm_s16le", True)]:
        path = DIRECTORY / f"样例.{suffix}"
        args = [executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
        if video:
            args += ["-f", "lavfi", "-i", "color=c=black:s=32x32:r=10:d=0.5"]
        args += ["-i", str(source), "-c:a", codec]
        if video:
            args += ["-c:v", "mpeg4", "-shortest"]
        args += [str(path)]
        subprocess.run(args, check=True, capture_output=True, timeout=30)
        register(path.name, suffix, frame_tolerance=2048 if codec in {"libmp3lame", "aac", "libvorbis"} else 0,
                 note=f"Actual codec {codec}; MPEG4 video" if video else f"Actual codec {codec}")
    for suffix, delimiter, comments, header in [("csv", ",", "", "time[s],Pressure[Pa],Voltage[V]\n"),
                                                ("txt", "\t", "# sample_rate: 48000\n# units: Pa,V\n", "Pressure\tVoltage\n"),
                                                ("dat", " ", "", "")]:
        path = DIRECTORY / f"样例.{suffix}"
        with path.open("w", encoding="utf-8", newline="") as output:
            output.write(comments + header)
            for index in range(frames):
                values = ([f"{t[index]:.12f}"] if suffix == "csv" else []) + [f"{value:.17g}" for value in logical[:, index]]
                output.write(delimiter.join(values) + "\n")
        mapping = {"sample_rate": rate, "data_columns": [0, 1], "channel_units": ["Pa", "V"]} if suffix == "dat" else None
        register(path.name, suffix, ("Pa", "V"), mapping=mapping, note="Explicit synthetic physical units")
    path = DIRECTORY / "样例.mat"
    savemat(path, {"signal": logical, "sample_rate": rate, "units": ["Pa", "V"],
                  "channel_names": ["Pressure", "Voltage"], "channel_axis": 0})
    register(path.name, "mat", ("Pa", "V"), note="Classic MATLAB numeric matrix with known metadata")
    for name, kind in [("样例.hdf5", "hdf5"), ("MAT73样例.mat", "mat73")]:
        path = DIRECTORY / name
        matlab = kind == "mat73"
        with h5py.File(path, "w", userblock_size=512 if matlab else 0) as output:
            dataset = output.create_dataset("measurement/signal", data=logical.T if matlab else logical)
            dataset.attrs["sample_rate"] = rate
            dataset.attrs["units"] = np.asarray(["Pa", "V"], dtype=h5py.string_dtype())
            dataset.attrs["channel_names"] = np.asarray(["Pressure", "Voltage"], dtype=h5py.string_dtype())
            dataset.attrs["channel_axis"] = 0
            if matlab:
                dataset.attrs["MATLAB_class"] = np.bytes_("double")
        if matlab:
            with path.open("r+b") as stream:
                stream.write(b"MATLAB 7.3 MAT-file, Platform: synthetic validation fixture\0")
        register(name, kind, ("Pa", "V"), note="Numeric HDF5 layout; MATLAB stored axes reversed" if matlab else "Generic HDF5 dataset metadata")
    path = DIRECTORY / "样例.tdms"
    with TdmsWriter(path) as output:
        output.write_segment([RootObject(), GroupObject("Motor"),
                              ChannelObject("Motor", "Pressure", logical[0], {"wf_increment": 1 / rate, "unit_string": "Pa"}),
                              ChannelObject("Motor", "Voltage", logical[1], {"wf_increment": 1 / rate, "unit_string": "V"})])
    register(path.name, "tdms", ("Pa", "V"), note="Two channels with independent physical units")
    manifest = {"schema_version": 1, "evidence_scope": "synthetic codec/layout import fidelity; not real instrument validation",
                "cases": cases}
    (DIRECTORY / "fixture_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(cases)} actual files and fixture_manifest.json")


if __name__ == "__main__":
    main()
