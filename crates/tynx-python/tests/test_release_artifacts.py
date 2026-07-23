from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import cast

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CHECK_RELEASE = _ROOT / "scripts" / "check_release.py"
_PYTHON_PROJECT = _ROOT / "crates" / "tynx-python"


def _workspace_version() -> str:
    result = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    packages = json.loads(result.stdout)["packages"]
    return cast(
        str,
        next(package["version"] for package in packages if package["name"] == "tynx-python"),
    )


def _write_wheel(path: Path, version: str, *, include_licenses: bool) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"tynx-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: tynx\nVersion: {version}\n",
        )
        if include_licenses:
            archive.writestr(f"tynx-{version}.dist-info/licenses/LICENSE-APACHE", "Apache-2.0")
            archive.writestr(f"tynx-{version}.dist-info/licenses/LICENSE-MIT", "MIT")


@pytest.mark.parametrize("include_licenses", [True, False])
def test_release_check_requires_both_license_files(tmp_path: Path, include_licenses: bool) -> None:
    version = _workspace_version()
    wheel = tmp_path / f"tynx-{version}-py3-none-any.whl"
    _write_wheel(wheel, version, include_licenses=include_licenses)

    result = subprocess.run(
        [
            sys.executable,
            str(_CHECK_RELEASE),
            "wheels",
            "--expected",
            version,
            "--max-mib",
            "1",
            str(wheel),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    if include_licenses:
        assert result.returncode == 0, result.stderr
        assert f"version {version}" in result.stdout
    else:
        assert result.returncode != 0
        assert "missing license files" in result.stderr


def test_python_project_licenses_are_regular_copies() -> None:
    for filename in ("LICENSE-APACHE", "LICENSE-MIT"):
        packaged = _PYTHON_PROJECT / filename
        source = _ROOT / filename

        assert packaged.is_file()
        assert not packaged.is_symlink()
        assert packaged.read_bytes() == source.read_bytes()


def test_source_distribution_contains_both_license_files(tmp_path: Path) -> None:
    maturin = Path(sys.executable).with_name("maturin")
    result = subprocess.run(
        [
            str(maturin),
            "sdist",
            "--manifest-path",
            str(_PYTHON_PROJECT / "Cargo.toml"),
            "--out",
            str(tmp_path),
        ],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    [sdist] = tmp_path.glob("tynx-*.tar.gz")
    with tarfile.open(sdist, "r:gz") as archive:
        names = set(archive.getnames())

    root = sdist.name.removesuffix(".tar.gz")
    assert f"{root}/LICENSE-APACHE" in names
    assert f"{root}/LICENSE-MIT" in names
