"""Collect installed license originals for a local Windows distribution.

This creates an evidence inventory, not a redistribution compliance certificate.
No network downloads or changes to the application's license are performed.
Run after freezing, for example:
  .venv/Scripts/python tools/collect_distribution_notices.py --destination dist/AutoAcousticsPro
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SOURCES = {
    "PyQt": "https://www.riverbankcomputing.com/software/pyqt/intro",
    "PyQt license FAQ": "https://www.riverbankcomputing.com/commercial/license-faq",
    "Qt LGPL obligations": "https://www.qt.io/development/open-source-lgpl-obligations",
    "Qt third-party components": "https://doc.qt.io/qt-6.11/licenses-used-in-qt.html",
    "FFmpeg licensing": "https://ffmpeg.org/legal.html",
    "FFmpeg Windows build provider": "https://www.gyan.dev/ffmpeg/builds/",
    "GPLv3": "https://www.gnu.org/licenses/gpl-3.0.html",
    "LGPLv3": "https://www.gnu.org/licenses/lgpl-3.0.html",
}
EXCLUDED = ["tests/reference/upstream", "ISO reference attachments", "HEAD reference attachments"]


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".notice-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _is_notice(path: Path) -> bool:
    return bool(re.search(r"license|licence|copying|notice|copyright", path.name, re.I)) and path.suffix.lower() not in {
        ".py", ".pyc", ".dll", ".pyd", ".exe", ".pdf", ".zip", ".png", ".wav", ".csv"
    }


def _excluded(path: Path) -> bool:
    parts = [p.lower() for p in path.parts]
    return any(parts[i:i + 3] == ["tests", "reference", "upstream"] for i in range(len(parts))) or (
        "reference" in parts and any(p in parts for p in ("iso", "head", "attachments"))
    )


def _windows_version(path: Path) -> dict:
    """Read PE fixed version resources; never infer a version from its filename."""
    if os.name != "nt":
        return {}
    try:
        api = ctypes.windll.version
        api.GetFileVersionInfoSizeW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong)]
        api.GetFileVersionInfoW.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p]
        api.VerQueryValueW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint)]
        length = api.GetFileVersionInfoSizeW(str(path), None)
        if not length:
            return {}
        buffer = ctypes.create_string_buffer(length)
        if not api.GetFileVersionInfoW(str(path), 0, length, buffer):
            return {}
        pointer = ctypes.c_void_p()
        count = ctypes.c_uint()
        if not api.VerQueryValueW(buffer, "\\", ctypes.byref(pointer), ctypes.byref(count)) or count.value < 52:
            return {}
        values = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint32 * 13)).contents
        def version(high, low):
            return ".".join(str(v) for v in (high >> 16, high & 65535, low >> 16, low & 65535))
        result = {"file_version_fixed": version(values[2], values[3]), "product_version_fixed": version(values[4], values[5])}
        if api.VerQueryValueW(buffer, "\\VarFileInfo\\Translation", ctypes.byref(pointer), ctypes.byref(count)) and count.value >= 4:
            translation = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint16 * (count.value // 2))).contents
            language, codepage = translation[0], translation[1]
            for field, key in (("FileVersion", "file_version"), ("ProductVersion", "product_version"), ("LegalCopyright", "legal_copyright")):
                query = f"\\StringFileInfo\\{language:04x}{codepage:04x}\\{field}"
                if api.VerQueryValueW(buffer, query, ctypes.byref(pointer), ctypes.byref(count)) and count.value:
                    result[key] = ctypes.wstring_at(pointer.value).strip()
        return result
    except (OSError, AttributeError, ValueError):
        return {}


def collect_notices(destination: Path, *, site_packages: Path, lockfile: Path,
                    projectfile: Path, bundle: Path, ffmpeg_root: Path,
                    python_base: Path) -> dict:
    destination, site_packages, bundle = map(lambda p: Path(p).resolve(), (destination, site_packages, bundle))
    lockfile, projectfile, ffmpeg_root, python_base = map(lambda p: Path(p).resolve(), (lockfile, projectfile, ffmpeg_root, python_base))
    project = tomllib.loads(projectfile.read_text(encoding="utf-8"))["project"]
    locked = {}
    for line in lockfile.read_text(encoding="utf-8-sig").splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^;\s]+)", clean)
        if not match:
            raise ValueError(f"Unsupported non-pinned lock entry: {clean}")
        locked[_normalize(match[1])] = match[2]
    requirement_name = lambda text: _normalize(re.match(r"[A-Za-z0-9_.-]+", text).group())
    direct = {requirement_name(s) for s in project.get("dependencies", [])}
    development = {requirement_name(s) for values in project.get("optional-dependencies", {}).values() for s in values}
    result = {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "locked installed environment plus observed frozen files; conservative superset, not a minimal runtime SBOM",
        "project": {"name": project["name"], "version": project.get("version"), "license": project.get("license")},
        "inputs": {"site_packages": str(site_packages), "lockfile": str(lockfile), "lockfile_sha256": _sha256(lockfile),
                   "projectfile": str(projectfile), "projectfile_sha256": _sha256(projectfile), "bundle": str(bundle)},
        "packages": [], "files": [], "warnings": [], "missing_packages": [],
        "excluded_material": EXCLUDED, "official_sources": dict(OFFICIAL_SOURCES),
        "public_redistribution_ready": False,
        "remaining_redistribution_work": [
            "Choose the application's distribution license or obtain an appropriate commercial PyQt license; this collector does not assign one.",
            "Provide complete corresponding source and applicable build/install information for conveyed GPL/LGPL components; upstream URLs alone are not a verified source-offer mechanism.",
            "Collect version-matched Qt/Chromium/Qt Multimedia and other native-library third-party attributions; installed wheel notices may be incomplete.",
            "Preserve LGPL users' ability to replace/relink libraries and run modified versions, including the frozen Python environment.",
            "Verify redistribution rights for native runtime DLLs and for every third-party library statically linked into FFmpeg.",
        ],
    }
    def warn(code, component, message):
        result["warnings"].append({"code": code, "component": component, "message": message})

    def copy(source: Path, target: Path, component: str, kind="license") -> dict:
        content = source.read_bytes()
        _atomic_bytes(destination / target, content)
        record = {"component": component, "kind": kind, "source": str(source.resolve()),
                  "destination": target.as_posix(), "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        result["files"].append(record)
        return record

    found = set()
    for info in sorted(site_packages.glob("*.dist-info"), key=lambda p: p.name.lower()):
        distribution = importlib.metadata.Distribution.at(info)
        metadata = distribution.metadata
        name, version = metadata.get("Name"), metadata.get("Version")
        if not name:
            continue
        normalized = _normalize(name)
        if normalized not in locked:
            continue
        found.add(normalized)
        prefix = Path("licenses") / "python-packages" / f"{normalized}-{version}"
        package = {
            "name": name, "version": version, "locked_version": locked[normalized],
            "role": "direct-runtime" if normalized in direct else "build-or-test" if normalized in development or normalized in {"altgraph", "pefile", "pyinstaller-hooks-contrib", "iniconfig", "pluggy", "pygments"} else "transitive-environment",
            "license_expression": metadata.get("License-Expression"), "license_field": metadata.get("License"),
            "license_classifiers": [v for v in metadata.get_all("Classifier", []) if v.startswith("License ::")],
            "project_urls": metadata.get_all("Project-URL", []), "home_page": metadata.get("Home-page"),
            "declared_license_files": metadata.get_all("License-File", []), "license_files": [],
            "bundle_evidence": [], "dist_info_source": str(info),
        }
        if version != locked[normalized]:
            warn("version_mismatch", name, f"Installed {version}, locked {locked[normalized]}")
        for filename in ("METADATA", "WHEEL", "direct_url.json"):
            source = info / filename
            if source.is_file():
                copy(source, prefix / filename, name, "installed_metadata")
        notices = {p.resolve() for p in info.rglob("*") if p.is_file() and _is_notice(p)}
        for entry in distribution.files or []:
            relative = Path(str(entry))
            source = Path(distribution.locate_file(entry)).resolve()
            if _is_notice(relative):
                if not source.is_relative_to(site_packages):
                    warn("source_outside_environment", name, f"Excluded RECORD path: {entry}")
                elif source.is_file() and not _excluded(relative):
                    notices.add(source)
            bundled = bundle / "_internal" / relative
            if bundled.is_file() and not _excluded(relative):
                package["bundle_evidence"].append(relative.as_posix())
        for source in sorted(notices):
            relative = source.relative_to(info) if source.is_relative_to(info) else Path("vendored") / source.relative_to(site_packages)
            record = copy(source, prefix / relative, name)
            package["license_files"].append(record["destination"])
        if not notices:
            warn("license_missing", name, "No original license/notice file found in installed distribution")
        result["packages"].append(package)
        if normalized == "pyqt6-qt6":
            result["official_sources"]["Qt third-party components"] = f"https://doc.qt.io/qt-{'.'.join(version.split('.')[:2])}/licenses-used-in-qt.html"
    result["missing_packages"] = sorted(locked.keys() - found)
    for name in result["missing_packages"]:
        warn("package_missing", name, "Locked dependency is not installed in the selected environment")

    runtime = {"version": sys.version, "source": str(python_base), "license_files": []}
    for source in sorted(python_base.glob("*LICENSE*")):
        if source.is_file():
            runtime["license_files"].append(copy(source, Path("licenses") / "python-runtime" / source.name, "CPython")["destination"])
    if not runtime["license_files"]:
        warn("license_missing", "CPython", "Python runtime license was not found")
    result["python_runtime"] = runtime

    result["bundle_native_files"] = []
    if bundle.is_dir():
        for source in sorted(bundle.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(bundle)
            if _excluded(relative) or any(part == "licenses" for part in relative.parts) or source.name == "THIRD_PARTY_NOTICES.md":
                continue
            if _is_notice(source):
                copy(source, Path("licenses") / "bundled-notices" / relative, "frozen bundle")
            if source.suffix.lower() in {".dll", ".pyd", ".exe"}:
                result["bundle_native_files"].append({"path": relative.as_posix(), "size_bytes": source.stat().st_size,
                                                       "sha256": _sha256(source), **_windows_version(source)})
    else:
        warn("bundle_missing", "frozen bundle", "No frozen bundle found; environment notices were still collected")

    ffmpeg = {"root": str(ffmpeg_root), "license_files": [], "binaries": [], "readme_version": None,
              "source_code_url": None, "readme_license": None, "probe": {"status": "unavailable"}}
    for filename in ("LICENSE", "README.txt"):
        source = ffmpeg_root / filename
        if source.is_file():
            ffmpeg["license_files"].append(copy(source, Path("licenses") / "ffmpeg" / filename, "FFmpeg", "license" if filename == "LICENSE" else "build_provenance")["destination"])
            if filename == "README.txt":
                content = source.read_text(encoding="utf-8", errors="replace")
                for label, key in [("Version", "readme_version"), ("License", "readme_license"), ("Source Code", "source_code_url")]:
                    match = re.search(rf"^{label}:\s*(.+)$", content, re.M)
                    if match:
                        ffmpeg[key] = match[1].strip()
        else:
            warn("ffmpeg_notice_missing", "FFmpeg", f"Missing {source}")
    for name in ("ffmpeg", "ffprobe"):
        source = ffmpeg_root / "bin" / f"{name}.exe"
        if not source.is_file():
            warn("ffmpeg_binary_missing", "FFmpeg", f"Cannot verify original {source}")
            continue
        record = {"name": name, "source": str(source), "sha256": _sha256(source), "size_bytes": source.stat().st_size,
                  "bundled": [], "bundle_matches_source": None}
        for evidence in result["bundle_native_files"]:
            if Path(evidence["path"]).name.lower() == f"{name}.exe":
                record["bundled"].append(evidence)
        if record["bundled"]:
            record["bundle_matches_source"] = all(r["sha256"] == record["sha256"] for r in record["bundled"])
            if not record["bundle_matches_source"]:
                warn("ffmpeg_binary_mismatch", "FFmpeg", f"Bundled {name} differs from original executable")
        elif bundle.is_dir():
            warn("ffmpeg_bundle_missing", "FFmpeg", f"No bundled {name} executable found")
        ffmpeg["binaries"].append(record)
    executable = ffmpeg_root / "bin" / "ffmpeg.exe"
    if executable.is_file():
        try:
            if os.name == "nt":
                with executable.open("rb") as stream:
                    if stream.read(2) != b"MZ":
                        raise RuntimeError("File is not a Windows PE executable; version probe skipped")
            process = subprocess.run([str(executable), "-version"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                                     timeout=15, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            output = process.stdout + process.stderr
            if process.returncode != 0:
                raise RuntimeError(f"Exit {process.returncode}: {output[:400]}")
            ffmpeg["probe"] = {"status": "ok", "output": output, "gpl_enabled": "--enable-gpl" in output,
                               "version3_enabled": "--enable-version3" in output, "static_enabled": "--enable-static" in output,
                               "nonfree_enabled": "--enable-nonfree" in output}
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            ffmpeg["probe"] = {"status": "unavailable", "error": str(error)}
            warn("ffmpeg_probe_unavailable", "FFmpeg", str(error))
    result["ffmpeg"] = ffmpeg
    # These are actionable redistribution gaps, distinct from missing inventory evidence.
    result["qt_wheel_attribution_status"] = "only installed license texts collected; complete native third-party attribution unverified"
    failure_codes = {"version_mismatch", "license_missing", "package_missing", "source_outside_environment",
                     "ffmpeg_notice_missing", "ffmpeg_binary_mismatch"}
    result["license_inventory_status"] = "incomplete" if any(w["code"] in failure_codes for w in result["warnings"]) else "complete"
    result["inventory_status"] = "incomplete" if result["warnings"] else "complete"
    result["summary"] = {"locked_packages": len(locked), "installed_locked_packages": len(result["packages"]),
                         "copied_original_files": len(result["files"]), "native_files_observed": len(result["bundle_native_files"]),
                         "warning_count": len(result["warnings"])}
    _atomic_bytes(destination / "THIRD_PARTY_NOTICES.md", _notice_markdown(result).encode("utf-8"))
    _atomic_bytes(destination / "licenses" / "distribution_manifest.json", (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return result


def _notice_markdown(data: dict) -> str:
    ffmpeg = data["ffmpeg"]
    sources = data["official_sources"]
    qt = next((p for p in data["packages"] if _normalize(p["name"]) == "pyqt6-qt6"), None)
    qt_version = qt["version"] if qt else "未取得"
    pyqt = next((p for p in data["packages"] if _normalize(p["name"]) == "pyqt6"), None)
    gpl_link = pyqt["license_files"][0] if pyqt and pyqt["license_files"] else sources["GPLv3"]
    native_qt = [v for v in data["bundle_native_files"] if "/PyQt6/" in "/" + v["path"]]
    special = [(v["path"], v.get("product_version", v.get("product_version_fixed", "版本资源不可用"))) for v in native_qt if Path(v["path"]).name in {"Qt6Core.dll", "Qt6Pdf.dll"} or Path(v["path"]).name.startswith("avcodec-")]
    rows = []
    for package in data["packages"]:
        label = package["license_expression"] or package["license_field"] or "; ".join(package["license_classifiers"]) or "元数据未声明；见原文"
        if len(label) > 150 or "\n" in label:
            label = "见安装包许可原文"
        label = label.replace("|", "\\|")
        link = package["license_files"][0] if package["license_files"] else None
        rows.append(f"| {package['name']} | {package['version']} | {package['role']} | {label} | " + (f"[原文]({link})" if link else "缺失") + " |")
    notices = f"""# AutoAcoustics Pro 第三方许可与来源清单

本清单生成于 {data['collected_at_utc']}，面向用户当前 Windows 本机交付。项目自身许可证未由本工具设定；许可归各权利人所有，复制的原文保持原字节。

收集范围：锁定环境的 {data['summary']['installed_locked_packages']}/{data['summary']['locked_packages']} 个已安装依赖、实际冻结包中的原始 notice、Python 运行时，以及本机 FFmpeg。共复制 {data['summary']['copied_original_files']} 个原始文件，观察 {data['summary']['native_files_observed']} 个原生二进制。库存状态：**{data['inventory_status']}**。完整来源路径、版本元数据、锁定文件哈希、逐文件 SHA-256 和未解决项见 [JSON 清单](licenses/distribution_manifest.json)。

锁定环境是保守的来源清单，包含开发/构建依赖；未声称每个环境依赖都在冻结包内。`bundle_evidence` 和原生文件表记录实物证据。纯 Python 模块可能压缩于 PYZ，未观察到独立文件不代表未使用。

## PyQt6 与 Qt

当前交付环境的 PyQt6 是 GPLv3 版本，PyQt6-Qt6 许可文本为 LGPLv3。PyQt 本身不能按 LGPL 授权；其官方选择为 GPLv3 或 Riverbank 商业许可。公开或向其他收件人分发基于当前 PyQt 的组合程序前，需要确定 GPL 兼容的项目许可与对应源码，或使用适用的商业授权。[Riverbank 官方许可说明]({sources['PyQt']})、[官方许可 FAQ]({sources['PyQt license FAQ']})。

Qt 的 LGPL 条件包括显著告知及许可原文、库的对应源码（含修改）、允许替换/重新链接并运行修改库，且不能限制为调试修改而逆向工程。动态 DLL 形式本身不完成全部要求；冻结的 Python 库同样需要验证用户能实际替换。部分 Qt 模块或其中第三方组件有不同条款。[Qt 官方义务]({sources['Qt LGPL obligations']})。

已安装 Qt wheel 的 LICENSE 已复制，但没有把它当作 Qt 所有第三方代码的完整 attribution。实际包含的模块应按 **{qt_version} 对应版本**补齐许可与源码记录；当前 Qt/Chromium、Qt Multimedia 原生第三方声明尚未完整核实。[Qt 对应版本第三方组件清单]({sources['Qt third-party components']})。

## FFmpeg

独立工具来源为 `{ffmpeg['root']}`，README 版本 **{ffmpeg['readme_version'] or '未取得'}**，声明许可 **{ffmpeg['readme_license'] or '未取得'}**。实际 `ffmpeg -version` 配置和 ffmpeg/ffprobe 的来源与冻结包哈希在 JSON 中；README 与 GPL 原文位于 [licenses/ffmpeg](licenses/ffmpeg)。README 的上游提交：{ffmpeg['source_code_url'] or '未取得'}。来源提供方：[Gyan Windows builds]({sources['FFmpeg Windows build provider']})。

本机实际探针记录 GPL/version3/static 开关：`{ffmpeg['probe'].get('gpl_enabled', '未取得')}/{ffmpeg['probe'].get('version3_enabled', '未取得')}/{ffmpeg['probe'].get('static_enabled', '未取得')}`。FFmpeg 的可选 GPL 部件会使 FFmpeg 本身适用 GPL；交付这些二进制时须满足 GPLv3 的许可、版权及完整对应源码交付方式，涵盖静态链接库和复现构建所需信息。一个上游提交链接没有覆盖全部静态依赖，也不能当作已验证的完整源码交付。[FFmpeg 官方许可说明]({sources['FFmpeg licensing']})、[GPLv3 正文]({sources['GPLv3']})。

应用以独立子进程使用这些工具；不能仅凭 FFmpeg GPL 标签推断应用整体许可证。PyQt 的组合程序要求需要单独处理。Qt Multimedia 随带的 FFmpeg DLL 是另一组二进制，版本和许可核实不能用独立工具的 8.1.2 README 替代。

## 本机使用与再次分发

运行或私下修改 GPL 程序不要求把修改公开。**“用户本机使用”描述目标用途，不免除把副本交给收件人时可能产生的分发义务。** 本清单提供原始声明和可核对来源，`public_redistribution_ready=false`；尚未把当前包认定为可公开发布的安装包。项目源码许可、对应源码交付、Qt 第三方声明和静态媒体库来源仍需按实际分发方式完成。[GPLv3]({gpl_link})、[LGPLv3]({sources['LGPLv3']})。

NI-DAQmx 驱动、HEAD SDK、ISO 标准正文与 HEAD 参考附件不在该依赖许可授权范围。参考测试材料 `tests/reference/upstream` 未被复制；它们不能从测试可访问推定为可公开再分发。该清单也不赋予真实音频数据分发权。

## 已锁定依赖

`direct-runtime` 是 pyproject 直接依赖，`build-or-test` 为构建/测试工具，`transitive-environment` 为锁定环境其他依赖。各文件原文和元数据以 JSON 全表为准。

| 组件 | 实际版本 | 范围 | 安装元数据许可 | 许可原文 |
|---|---|---|---|---|
"""
    notices += "\n".join(rows) + "\n"
    if special:
        notices += "\n## Qt 原生实物版本\n\n| 文件 | PE 产品版本 |\n|---|---|\n" + "\n".join(f"| `{path}` | {version} |" for path, version in special) + "\n"
    notices += "\n## 库存警告\n\n" + ("\n".join(f"- `{w['code']}` / {w['component']}: {w['message']}" for w in data["warnings"]) if data["warnings"] else "未发现缺失的锁定依赖或原始许可文件；这不等于完成公开再分发义务。") + "\n"
    notices += "\n## 重新收集\n\n```powershell\n.venv\\Scripts\\python.exe tools\\collect_distribution_notices.py --destination dist\\AutoAcousticsPro\n```\n\n该命令只补充外部 notice 和 licenses 目录，不重编译程序。默认输入为本项目 requirements.lock、pyproject.toml、.venv 和 dist/AutoAcousticsPro；可用 --bundle 指定另一个实际包。\n"
    return notices


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT)
    parser.add_argument("--site-packages", type=Path, default=ROOT / ".venv/Lib/site-packages")
    parser.add_argument("--lockfile", type=Path, default=ROOT / "requirements.lock")
    parser.add_argument("--projectfile", type=Path, default=ROOT / "pyproject.toml")
    parser.add_argument("--bundle", type=Path, default=ROOT / "dist/AutoAcousticsPro")
    parser.add_argument("--ffmpeg-root", type=Path, default=Path("C:/ffmpeg"))
    parser.add_argument("--python-base", type=Path, default=Path(sys.base_prefix))
    parser.add_argument("--strict", action="store_true", help="Exit 2 if original inventory evidence is incomplete; not a redistribution approval")
    args = parser.parse_args(argv)
    result = collect_notices(args.destination, site_packages=args.site_packages, lockfile=args.lockfile,
                             projectfile=args.projectfile, bundle=args.bundle, ffmpeg_root=args.ffmpeg_root, python_base=args.python_base)
    print(json.dumps({"destination": str(args.destination.resolve()), "inventory_status": result["inventory_status"],
                      "public_redistribution_ready": False, **result["summary"]}, ensure_ascii=False))
    return 2 if args.strict and result["inventory_status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
