#!/usr/bin/env python3
"""Validate Tynx release metadata and built wheel artifacts."""

from __future__ import annotations

import argparse
import email
import importlib.metadata
import json
import re
import subprocess
import zipfile
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
PACKAGE = "tynx-python"
BUILD_PATHS = (
    b"/root/.cargo",
    b"/home/runner",
    b"/Users/runner",
    b"\\Users\\runneradmin",
)
MAX_MANYLINUX = (2, 28)
MANYLINUX_TAG = re.compile(r"manylinux_(\d+)_(\d+)")
REQUIRED_LICENSES = {"LICENSE-APACHE", "LICENSE-MIT"}


def workspace_version() -> str:
    result = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=WORKSPACE,
        check=True,
        capture_output=True,
        text=True,
    )
    packages = json.loads(result.stdout)["packages"]
    return next(
        package["version"] for package in packages if package["name"] == PACKAGE
    )


def check_tag(tag: str) -> None:
    expected = f"v{workspace_version()}"
    if tag != expected:
        raise SystemExit(f"tag {tag} does not match package version {expected}")


def wheel_metadata(wheel: Path) -> email.message.Message:
    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise SystemExit(
                f"{wheel}: expected one METADATA file, found {metadata_files}"
            )
        return email.message_from_bytes(archive.read(metadata_files[0]))


def check_wheels(wheels: list[Path], expected: str, max_mib: int) -> None:
    wheels = [wheel for wheel in wheels if wheel.is_file()]
    if not wheels:
        raise SystemExit("no wheels built")

    max_bytes = max_mib * 1024 * 1024
    for wheel in wheels:
        metadata_version = wheel_metadata(wheel)["Version"]
        if metadata_version != expected:
            raise SystemExit(
                f"{wheel}: metadata version {metadata_version!r} != expected {expected!r}"
            )
        if not wheel.name.startswith(f"tynx-{expected}-"):
            raise SystemExit(
                f"{wheel}: filename does not contain expected version {expected}"
            )
        match = MANYLINUX_TAG.search(wheel.name)
        if match is not None:
            required_glibc = tuple(map(int, match.groups()))
            if required_glibc > MAX_MANYLINUX:
                required = ".".join(map(str, required_glibc))
                maximum = ".".join(map(str, MAX_MANYLINUX))
                raise SystemExit(
                    f"{wheel}: requires glibc {required}; release wheels must support "
                    f"glibc {maximum} or older"
                )

        size = wheel.stat().st_size
        if size > max_bytes:
            raise SystemExit(
                f"{wheel}: {size / 1024 / 1024:.1f} MiB exceeds the {max_mib} MiB budget"
            )

        with zipfile.ZipFile(wheel) as archive:
            licenses = {
                Path(name).name
                for name in archive.namelist()
                if ".dist-info/licenses/" in name
            }
            missing_licenses = REQUIRED_LICENSES - licenses
            if missing_licenses:
                raise SystemExit(
                    f"{wheel}: wheel is missing license files {sorted(missing_licenses)}"
                )
            for name in archive.namelist():
                if not name.endswith((".so", ".pyd", ".dylib")):
                    continue
                data = archive.read(name)
                leaks = [path.decode() for path in BUILD_PATHS if path in data]
                if leaks:
                    raise SystemExit(f"{wheel}:{name} leaks build paths: {leaks}")

        print(f"{wheel}: version {metadata_version}, {size / 1024 / 1024:.1f} MiB")


def smoke_test(expected: str) -> None:
    import tynx

    installed = importlib.metadata.version("tynx")
    if installed != expected or tynx.__version__ != expected:
        raise SystemExit(
            f"installed versions do not match {expected}: metadata={installed}, "
            f"module={tynx.__version__}"
        )

    tensor = tynx.Tensor([1.0, 2.0], requires_grad=True)
    (tensor * tensor).sum().backward()
    if tensor.grad is None or tensor.grad.tolist() != [2.0, 4.0]:
        raise SystemExit(f"unexpected autograd result: {tensor.grad}")
    if not hasattr(tynx, "Session"):
        raise SystemExit("installed wheel does not export Session")

    mask = tynx.Tensor([-1.0, 2.0]) > 0.0
    selected = tynx.where(mask, tynx.Tensor([10.0, 20.0]), 0.0)
    if selected.tolist() != [0.0, 20.0]:
        raise SystemExit(f"unexpected where result: {selected.tolist()}")

    indices = tynx.Tensor([[1, 0]], dtype="int64")
    gathered = tynx.Tensor([[3.0, 4.0]]).gather(1, indices)
    if gathered.tolist() != [[4.0, 3.0]]:
        raise SystemExit(f"unexpected gather result: {gathered.tolist()}")

    print("installed wheel smoke test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version")

    tag = commands.add_parser("tag")
    tag.add_argument("tag")

    wheels = commands.add_parser("wheels")
    wheels.add_argument("--expected", required=True)
    wheels.add_argument("--max-mib", required=True, type=int)
    wheels.add_argument("wheels", nargs="+", type=Path)

    smoke = commands.add_parser("smoke")
    smoke.add_argument("--expected", required=True)

    args = parser.parse_args()
    if args.command == "version":
        print(workspace_version())
    elif args.command == "tag":
        check_tag(args.tag)
    elif args.command == "wheels":
        check_wheels(args.wheels, args.expected, args.max_mib)
    else:
        smoke_test(args.expected)


if __name__ == "__main__":
    main()
