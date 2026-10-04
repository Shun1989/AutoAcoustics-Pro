import hashlib
import json
from pathlib import Path
import subprocess
import sys

from tools.collect_distribution_notices import collect_notices


def fixture_environment(tmp_path: Path, *, version="1.2", license_file=True):
    site = tmp_path / "site"
    info = site / f"Example_Pkg-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.4\nName: Example-Pkg\nVersion: {version}\n"
        "License-Expression: BSD-3-Clause\nLicense-File: LICENSE\n"
        "Project-URL: Source, https://example.org/source\n", encoding="utf-8"
    )
    paths = [f"{info.name}/METADATA,,", f"{info.name}/licenses/LICENSE,,"]
    if license_file:
        (info / "licenses").mkdir()
        (info / "licenses" / "LICENSE").write_bytes(b"Actual original license\r\n")
        notice = site / "example_pkg" / "vendor" / "NOTICE"
        notice.parent.mkdir(parents=True)
        notice.write_bytes(b"Vendored copyright notice\n")
        paths.append("example_pkg/vendor/NOTICE,,")
    (info / "RECORD").write_text("\n".join(paths), encoding="utf-8")
    lock = tmp_path / "requirements.lock"
    lock.write_text("Example-Pkg==1.2\n", encoding="utf-8")
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nname="sample"\nversion="0.1"\ndependencies=["Example-Pkg"]\n', encoding="utf-8")
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.mkdir()
    (ffmpeg / "LICENSE").write_bytes(b"GPL version 3 original text\n")
    (ffmpeg / "README.txt").write_text(
        "Version: 8.1.2-essentials_build-www.gyan.dev\nLicense: GPL v3\n"
        "Source Code: https://github.com/FFmpeg/FFmpeg/commit/38b88335f9\n", encoding="utf-8"
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    runtime = tmp_path / "python"
    runtime.mkdir()
    (runtime / "LICENSE.txt").write_bytes(b"PSF original license\n")
    return dict(site_packages=site, lockfile=lock, projectfile=project,
                ffmpeg_root=ffmpeg, bundle=bundle, python_base=runtime)


def test_original_installed_and_vendored_licenses_copied_with_hashes(tmp_path):
    env = fixture_environment(tmp_path)
    destination = tmp_path / "delivery"
    result = collect_notices(destination, **env)
    package = result["packages"][0]
    assert package["name"] == "Example-Pkg"
    assert package["version"] == "1.2"
    assert package["locked_version"] == "1.2"
    assert package["role"] == "direct-runtime"
    assert package["license_expression"] == "BSD-3-Clause"
    assert package["license_files"]
    assert result["license_inventory_status"] == "complete"
    assert result["inventory_status"] == "incomplete"  # Fixture intentionally has no media executables.
    copies = [r for r in result["files"] if r["component"] == "Example-Pkg" and r["kind"] == "license"]
    assert len(copies) == 2
    for record in result["files"]:
        copied = destination / record["destination"]
        assert copied.read_bytes() == Path(record["source"]).read_bytes()
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == record["sha256"]
    assert json.loads((destination / "licenses" / "distribution_manifest.json").read_text(encoding="utf-8"))["packages"] == result["packages"]
    assert (destination / "THIRD_PARTY_NOTICES.md").is_file()
    assert result["project"]["license"] is None
    assert "license" not in env["projectfile"].read_text(encoding="utf-8")


def test_missing_or_mismatched_dependency_not_reported_as_complete(tmp_path):
    env = fixture_environment(tmp_path, version="1.1", license_file=False)
    env["lockfile"].write_text("Example-Pkg==1.2\nMissing-Pkg==4.0\n", encoding="utf-8")
    result = collect_notices(tmp_path / "delivery", **env)
    assert result["inventory_status"] == "incomplete"
    codes = {warning["code"] for warning in result["warnings"]}
    assert {"version_mismatch", "license_missing", "package_missing"} <= codes
    assert result["missing_packages"] == ["missing-pkg"]
    assert result["packages"][0]["license_files"] == []
    assert result["public_redistribution_ready"] is False


def test_bundle_notice_and_binary_evidence_exclude_reference_attachments(tmp_path):
    env = fixture_environment(tmp_path)
    bundled_notice = env["bundle"] / "_internal" / "_soundfile_data" / "COPYING"
    bundled_notice.parent.mkdir(parents=True)
    bundled_notice.write_bytes(b"Actual bundled libsndfile notice\n")
    reference = env["bundle"] / "tests" / "reference" / "upstream" / "LICENSE"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"Reference material only\n")
    binary = env["bundle"] / "_internal" / "bin" / "ffmpeg.EXE"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"test binary, not an executable")
    source_binary = env["ffmpeg_root"] / "bin" / "ffmpeg.exe"
    source_binary.parent.mkdir()
    source_binary.write_bytes(binary.read_bytes())
    result = collect_notices(tmp_path / "delivery", **env)
    assert any(f["source"] == str(bundled_notice.resolve()) for f in result["files"])
    assert not any("reference" in f["source"] for f in result["files"])
    assert result["ffmpeg"]["readme_version"] == "8.1.2-essentials_build-www.gyan.dev"
    assert result["ffmpeg"]["source_code_url"].endswith("38b88335f9")
    assert result["ffmpeg"]["binaries"][0]["bundle_matches_source"] is True
    assert result["ffmpeg"]["probe"]["status"] == "unavailable"


def test_record_cannot_copy_license_from_outside_environment(tmp_path):
    env = fixture_environment(tmp_path)
    escaped = tmp_path / "LICENSE-secret"
    escaped.write_bytes(b"Must not be copied")
    info = next(env["site_packages"].glob("*.dist-info"))
    with (info / "RECORD").open("a", encoding="utf-8") as stream:
        stream.write("\n../LICENSE-secret,,\n")
    result = collect_notices(tmp_path / "delivery", **env)
    assert not any(f["source"] == str(escaped.resolve()) for f in result["files"])
    assert "source_outside_environment" in {w["code"] for w in result["warnings"]}


def test_cli_destination_and_repeat_collection_preserve_input_bytes(tmp_path):
    env = fixture_environment(tmp_path)
    destination = tmp_path / "custom delivery"
    script = Path(__file__).resolve().parents[1] / "tools" / "collect_distribution_notices.py"
    command = [sys.executable, str(script), "--destination", str(destination)]
    for key, value in env.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    source_hash = hashlib.sha256(env["projectfile"].read_bytes()).hexdigest()
    for _ in range(2):
        process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=15)
        assert process.returncode == 0, process.stderr
        result = json.loads(process.stdout)
        assert result["destination"] == str(destination.resolve())
        assert result["public_redistribution_ready"] is False
        assert result["installed_locked_packages"] == 1
    assert hashlib.sha256(env["projectfile"].read_bytes()).hexdigest() == source_hash
    assert not list(destination.rglob(".notice-*"))
    assert len(list((destination / "licenses" / "python-packages").glob("*"))) == 1
    strict = subprocess.run(command + ["--strict"], capture_output=True, text=True, timeout=15)
    assert strict.returncode == 2
