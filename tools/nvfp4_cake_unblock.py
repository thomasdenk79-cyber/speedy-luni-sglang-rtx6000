#!/usr/bin/env python3
"""Unblock FlashInfer's Cake-generated QSA-NVFP4 kernels in a wheel install.

A Cake package (schema ``cake-build/1``) keeps its whole build contract in
``build.json`` next to the loader: one entry per ``<program>[_<spec>]_<arch>``
library with package-relative sources, digests, flags and the ``aot/<name>.so``
path.  The published ``flashinfer-python`` wheel ships ``build.json`` and the
loader for ``flashinfer.experimental.cake_qsa_nvfp4`` but **no** ``csrc/`` and
**no** ``aot/`` payload, so ``_verified_sources`` dies with
``FileNotFoundError`` before the compiler is ever asked - the kernel looks
"broken" while the wheel is merely incomplete.

The sources do exist upstream, and every one of them must match the digest in
``build.json`` byte for byte, otherwise the loader raises ``RuntimeError``.
This tool fetches, verifies and (only with ``--apply``) installs them into the
package tree of the container you are running.

CPU only: nothing here imports CUDA or launches a kernel, so it can run before
a GPU window is open.

Contract verified on ``lmsysorg/sglang:nightly-dev-cu13-20261117-25c1c439``
(flashinfer 0.7.4): 22 libraries, 4 of them ``sm_120a``, all with
``min_cuda 12.8``; 8 sources plus one shared header ``csrc/cake_ffi_checks.h``.

Examples
    python3 tools/nvfp4_cake_unblock.py status
    python3 tools/nvfp4_cake_unblock.py verify --sources-dir /tmp/csrc \
        --build-json /tmp/fi-cake-build.json
    python3 tools/nvfp4_cake_unblock.py unblock --ref v0.7.4 --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_PACKAGE = "flashinfer.experimental.cake_qsa_nvfp4"
RAW_BASE = "https://raw.githubusercontent.com/flashinfer-ai/flashinfer"
USER_AGENT = "speedy-nvfp4-cake-unblock/1.0"
SCHEMA = "cake-build/1"
#: header shared by every kernel/binding pair, package-relative
EXTRA_SOURCES = ("csrc/cake_ffi_checks.h",)


# --------------------------------------------------------------------------- #
# manifest access
# --------------------------------------------------------------------------- #
def package_dir(package: str, build_json: str | None) -> tuple[Path, Path]:
    """``(package_dir, build.json)`` of an installed Cake package.

    ``--build-json`` lets the digest check run against a copy of the manifest on
    a machine without the wheel (the host has no torch); the package directory is
    then only used as a label.
    """
    bj = Path(build_json) if build_json else None
    if bj is not None:
        if not bj.is_file():
            raise SystemExit(f"--build-json {bj} does not exist")
        return bj.parent, bj
    try:
        module = __import__(package, fromlist=["__file__"])
    except Exception as exc:  # pragma: no cover - import failure is a finding
        raise SystemExit(
            f"{package} not importable ({type(exc).__name__}: {exc}); "
            "pass --build-json with an extracted manifest"
        ) from exc
    pkg = Path(module.__file__).parent
    bj = pkg / "build.json"
    if not bj.is_file():
        raise SystemExit(f"{pkg} has no build.json; not a Cake package")
    return pkg, bj


def manifest(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: unsupported schema {data.get('schema')!r}, want {SCHEMA!r}")
    return data


def entries(data: dict, arch: str) -> list[dict]:
    return [lib for lib in data.get("libraries", []) if lib.get("arch") == arch]


def source_map(data: dict, arch: str) -> dict[str, str]:
    """``{package-relative source path: sha256}`` for one architecture."""
    wanted: dict[str, str] = {}
    for lib in entries(data, arch):
        for relative, digest in lib.get("sha256", {}).items():
            if relative in wanted and wanted[relative] != digest:
                raise SystemExit(f"{relative}: two digests claimed for one path")
            wanted[relative] = digest
    if not wanted:
        raise SystemExit(f"build.json lists no sources for arch={arch}")
    return wanted


def repo_prefix(data: dict) -> str:
    """Repo path of the package, from ``build.json``'s dotted ``package`` key.

    ``build.json`` lists sources package-relatively (``csrc/<name>.cu``); upstream
    they live under the package directory
    (``flashinfer/experimental/cake_qsa_nvfp4/csrc/<name>.cu``).  Joining the two
    is the only correct way to fetch them - the repository root has no ``csrc/``
    for Cake packages.
    """
    dotted = data.get("package")
    if not dotted:
        raise SystemExit("build.json has no 'package' key; cannot map sources to the repo")
    return dotted.replace(".", "/") + "/"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def toolkit_ok(data: dict, arch: str, local_cuda: str | None) -> list[str]:
    """Library names of ``arch`` whose ``min_cuda`` the local toolkit cannot reach."""
    if local_cuda is None:
        return []
    def as_tuple(value: str) -> tuple[int, ...]:
        return tuple(int(part) for part in value.split(".")[:3] if part.isdigit())
    too_new: list[str] = []
    for lib in entries(data, arch):
        minimum = lib.get("min_cuda")
        if minimum and as_tuple(local_cuda) < as_tuple(minimum):
            too_new.append(f"{lib['name']} needs {minimum}")
    return too_new


def fetch(relative: str, ref: str, prefix: str = "") -> bytes:
    url = f"{RAW_BASE}/{ref}/{prefix}{relative}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read()


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #
def cmd_status(args: argparse.Namespace) -> int:
    pkg, bj = package_dir(args.package, args.build_json)
    data = manifest(bj)
    wanted = source_map(data, args.arch)
    libs = entries(data, args.arch)
    have = [name for name in wanted if (pkg / name).is_file()]
    prebuilt = [lib for lib in libs if (pkg / lib["path"]).is_file()]
    print(f"package_dir   {pkg}")
    print(f"build.json    {bj}")
    print(f"generator     {data.get('generator')}")
    print(f"arch          {args.arch}: {len(libs)} libraries, {len(wanted)} sources")
    print(f"csrc          {len(have)}/{len(wanted)} present")
    print(f"aot           {len(prebuilt)}/{len(libs)} prebuilt .so present")
    print(f"include_dirs  {data['common'].get('include_dirs')}")
    print(f"requires      {json.dumps(data['common'].get('requires'))}")
    if args.cuda_version:
        blocked = toolkit_ok(data, args.arch, args.cuda_version)
        print(f"toolkit       {args.cuda_version} -> "
              + (", ".join(blocked) if blocked else "all libraries buildable"))
    missing = [name for name in wanted if name not in have]
    for name in missing:
        print(f"  missing {name}")
    if missing or not prebuilt:
        print("verdict: incomplete wheel -> load_cake_qsa_nvfp4() cannot work as installed")
        return 1
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    _, bj = package_dir(args.package, args.build_json)
    wanted = source_map(manifest(bj), args.arch)
    sources_dir = Path(args.sources_dir)
    bad: list[str] = []
    checked = 0
    for relative, digest in sorted(wanted.items()):
        candidate = sources_dir / Path(relative).name
        if not candidate.is_file():
            bad.append(f"{relative}: {candidate} missing")
            continue
        actual = sha256_of(candidate)
        checked += 1
        if actual != digest:
            bad.append(f"{relative}: digest {actual} != build.json {digest}")
    print(f"verified {checked}/{len(wanted)} source(s) against {bj.name}, {len(bad)} problem(s)")
    for line in bad:
        print("  FAIL", line)
    return 1 if bad else 0


def cmd_unblock(args: argparse.Namespace) -> int:
    pkg, bj = package_dir(args.package, args.build_json)
    data = manifest(bj)
    wanted = source_map(data, args.arch)
    if args.cuda_version:
        blocked = toolkit_ok(data, args.arch, args.cuda_version)
        if blocked:
            print("toolkit too old for: " + ", ".join(blocked), file=sys.stderr)
            return 4
    staging = Path(args.workdir) / args.ref / args.arch
    staging.mkdir(parents=True, exist_ok=True)
    prefix = repo_prefix(data)

    to_install = dict(wanted)
    for extra in EXTRA_SOURCES:  # shared header, no digest in build.json
        to_install.setdefault(extra, "")
    verified: list[str] = []
    for relative, digest in sorted(to_install.items()):
        target = staging / Path(relative).name
        if not target.is_file():
            try:
                target.write_bytes(fetch(relative, args.ref, prefix))
            except urllib.error.HTTPError as exc:
                print(f"FETCH-FAIL {relative}: HTTP {exc.code} at ref {args.ref}", file=sys.stderr)
                return 3
            except urllib.error.URLError as exc:
                print(f"FETCH-FAIL {relative}: {exc.reason}", file=sys.stderr)
                return 3
        actual = sha256_of(target)
        if digest and actual != digest:
            print(f"DIGEST-FAIL {relative}: {actual} != build.json {digest}", file=sys.stderr)
            return 1
        verified.append(f"{relative} sha256={actual[:12]}"
                        + ("" if digest else " (header, unverified by manifest)"))
    manifest_files = [line for line in verified if "unverified" not in line]
    print(f"fetched {len(verified)} file(s) from {args.ref}: "
          f"{len(manifest_files)} match build.json digests, "
          f"{len(verified) - len(manifest_files)} header(s) outside the manifest")
    for line in verified:
        print(f"  ok {line}")

    install_dir = pkg / "csrc"
    if not args.apply:
        print(f"dry run: staged in {staging}; add --apply to install into {install_dir}")
        return 0
    install_dir.mkdir(parents=True, exist_ok=True)
    for relative in to_install:
        (install_dir / Path(relative).name).write_bytes((staging / Path(relative).name).read_bytes())
    print(f"installed {len(to_install)} file(s) into {install_dir}")
    print("next: nvcc compiles on first use (min_cuda 12.8) - run the compare harness")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", default=DEFAULT_PACKAGE)
    parser.add_argument("--build-json", help="extracted manifest, for hosts without the wheel")
    parser.add_argument("--arch", default="sm_120a")
    parser.add_argument("--cuda-version", help="local nvcc version, e.g. 13.0.88")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status", help="report wheel completeness")
    p_status.set_defaults(func=cmd_status)

    p_verify = sub.add_parser("verify", help="check a directory against the manifest digests")
    p_verify.add_argument("--sources-dir", required=True)
    p_verify.set_defaults(func=cmd_verify)

    p_unblock = sub.add_parser("unblock", help="fetch, verify and install the sources")
    p_unblock.add_argument("--ref", default="v0.7.4", help="flashinfer tag or commit")
    p_unblock.add_argument("--workdir", default="/tmp/opencode/nvfp4/csrc-check")
    p_unblock.add_argument("--apply", action="store_true", help="write into the package tree")
    p_unblock.set_defaults(func=cmd_unblock)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
