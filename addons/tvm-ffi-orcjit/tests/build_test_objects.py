#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Build test objects and quick-start example via direct compiler invocation.

Detects platform and available compilers, then builds all applicable variants:
  Linux:   LLVM Clang (default) + GCC
  macOS:   LLVM Clang (default) + Apple Clang
  Windows: LLVM Clang (default) + MSVC + clang-cl

Objects are cached: files are only recompiled when the source is newer than the
output.  Re-running this script is near-instant when nothing changed.

Usage:
    python build_test_objects.py [--llvm-prefix /opt/llvm]
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
ADDON_DIR = TESTS_DIR.parent
QUICKSTART_DIR = ADDON_DIR / "examples" / "quick-start"
SOURCES_C = TESTS_DIR / "sources" / "c"
SOURCES_CC = TESTS_DIR / "sources" / "cc"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_tvm_ffi_includedir() -> str:
    """Return the tvm-ffi C/C++ include directory."""
    result = subprocess.run(
        [sys.executable, "-m", "tvm_ffi.config", "--includedir"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _needs_build(source: Path, output: Path) -> bool:
    """True if *output* is missing or older than *source*."""
    if not output.exists():
        return True
    return source.stat().st_mtime > output.stat().st_mtime


def _compile(compiler: str, source: Path, output: Path, flags: list[str],
             include_dirs: list[str]) -> None:
    """Compile *source* → *output*.  Skips if cached."""
    if not _needs_build(source, output):
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = [compiler] + flags
    for d in include_dirs:
        if compiler in ("cl", "cl.exe", "clang-cl", "clang-cl.exe"):
            cmd.append(f"/I{d}")
        else:
            cmd += ["-I", d]
    if compiler in ("cl", "cl.exe", "clang-cl", "clang-cl.exe"):
        cmd += ["/c", f"/Fo{output}", str(source)]
    else:
        cmd += ["-c", "-o", str(output), str(source)]
    print(f"  {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)


# ---------------------------------------------------------------------------
# Variant builder
# ---------------------------------------------------------------------------


def _build_variant(
    name: str,
    *,
    c_compiler: str,
    cxx_compiler: str | None,
    c_flags: list[str],
    cxx_flags: list[str],
    include_dirs: list[str],
    c_outdir: Path,
    cc_outdir: Path | None,
) -> None:
    """Build all test objects for one compiler variant."""
    print(f"\n--- {name} ---", flush=True)
    for src in sorted(SOURCES_C.glob("*.c")):
        _compile(c_compiler, src, c_outdir / f"{src.stem}.o", c_flags, include_dirs)
    if cxx_compiler and cc_outdir:
        for src in sorted(SOURCES_CC.glob("*.cc")):
            _compile(cxx_compiler, src, cc_outdir / f"{src.stem}.o", cxx_flags, include_dirs)


# ---------------------------------------------------------------------------
# Platform dispatch
# ---------------------------------------------------------------------------


def _default_llvm_prefix() -> str:
    if platform.system() == "Windows":
        return "C:/opt/llvm"
    return "/opt/llvm"


def _find_llvm_bin(prefix: str) -> Path:
    """Return the LLVM bin directory under *prefix*."""
    p = Path(prefix)
    # conda-forge on Windows: Library/bin; elsewhere: bin
    win_lib = p / "Library" / "bin"
    if win_lib.exists():
        return win_lib
    return p / "bin"


def _base_flags(system: str, machine: str) -> tuple[list[str], list[str]]:
    """Return (c_flags, cxx_flags) base lists for *system*."""
    c: list[str] = ["-O2"]
    cxx: list[str] = ["-std=c++17", "-O2"]
    if system != "Windows":
        c.append("-fPIC")
        cxx.append("-fPIC")
    if machine in ("aarch64", "arm64"):
        c.append("-mno-outline-atomics")
        cxx.append("-mno-outline-atomics")
    return c, cxx


def _build_all(llvm_prefix: str) -> None:
    """Detect platform and build every applicable compiler variant."""
    system = platform.system()
    machine = platform.machine()
    include_dirs = [_get_tvm_ffi_includedir()]
    c_flags, cxx_flags = _base_flags(system, machine)
    llvm_bin = _find_llvm_bin(llvm_prefix)

    print(f"Platform: {system} {machine}", flush=True)
    print(f"LLVM bin: {llvm_bin}", flush=True)
    print(f"Include:  {include_dirs[0]}", flush=True)

    if system in ("Linux", "Darwin"):
        clang = str(llvm_bin / "clang")
        clangxx = str(llvm_bin / "clang++")
        _build_variant(
            "LLVM Clang",
            c_compiler=clang, cxx_compiler=clangxx,
            c_flags=c_flags, cxx_flags=cxx_flags,
            include_dirs=include_dirs,
            c_outdir=TESTS_DIR / "c", cc_outdir=TESTS_DIR / "cc",
        )
        if system == "Linux" and shutil.which("gcc"):
            _build_variant(
                "GCC",
                c_compiler="gcc", cxx_compiler="g++",
                c_flags=c_flags, cxx_flags=cxx_flags,
                include_dirs=include_dirs,
                c_outdir=TESTS_DIR / "c-gcc", cc_outdir=TESTS_DIR / "cc-gcc",
            )
        if system == "Darwin" and Path("/usr/bin/clang").exists():
            _build_variant(
                "Apple Clang",
                c_compiler="/usr/bin/clang", cxx_compiler="/usr/bin/clang++",
                c_flags=c_flags, cxx_flags=cxx_flags,
                include_dirs=include_dirs,
                c_outdir=TESTS_DIR / "c-appleclang",
                cc_outdir=TESTS_DIR / "cc-appleclang",
            )

    elif system == "Windows":
        # Windows: C-only for all variants.
        # Default: find clang from LLVM install or PATH.
        clang = shutil.which("clang")
        if not clang:
            candidate = llvm_bin / "clang.exe"
            if candidate.exists():
                clang = str(candidate)
        if clang:
            _build_variant(
                "LLVM Clang",
                c_compiler=clang, cxx_compiler=None,
                c_flags=c_flags, cxx_flags=[],
                include_dirs=include_dirs,
                c_outdir=TESTS_DIR / "c", cc_outdir=None,
            )
        # MSVC
        if shutil.which("cl"):
            _build_variant(
                "MSVC",
                c_compiler="cl", cxx_compiler=None,
                c_flags=["/O2", "/GS-"], cxx_flags=[],
                include_dirs=include_dirs,
                c_outdir=TESTS_DIR / "c-msvc", cc_outdir=None,
            )
        # clang-cl
        if shutil.which("clang-cl"):
            _build_variant(
                "clang-cl",
                c_compiler="clang-cl", cxx_compiler=None,
                c_flags=["/O2", "/GS-"], cxx_flags=[],
                include_dirs=include_dirs,
                c_outdir=TESTS_DIR / "c-clang-cl", cc_outdir=None,
            )


# ---------------------------------------------------------------------------
# Quick-start example
# ---------------------------------------------------------------------------


def _build_quickstart(llvm_prefix: str) -> None:
    """Compile the quick-start example objects."""
    system = platform.system()
    machine = platform.machine()
    include_dirs = [_get_tvm_ffi_includedir()]
    c_flags, cxx_flags = _base_flags(system, machine)
    llvm_bin = _find_llvm_bin(llvm_prefix)

    print(f"\n{'='*60}\nBuilding quick-start example\n{'='*60}", flush=True)

    # C variant (all platforms)
    c_compiler: str | None = None
    if system == "Windows":
        c_compiler = shutil.which("clang") or shutil.which("cl")
        if c_compiler and Path(c_compiler).name in ("cl.exe", "cl"):
            c_flags = ["/O2", "/GS-"]
    else:
        c_compiler = str(llvm_bin / "clang")

    if c_compiler:
        _compile(
            c_compiler, QUICKSTART_DIR / "add_c.c",
            QUICKSTART_DIR / "add_c.o", c_flags, include_dirs,
        )

    # C++ variant (Linux/macOS only)
    if system != "Windows":
        clangxx = str(llvm_bin / "clang++")
        _compile(
            clangxx, QUICKSTART_DIR / "add.cc",
            QUICKSTART_DIR / "add.o", cxx_flags, include_dirs,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ensure_built(llvm_prefix: str | None = None) -> None:
    """Build all test objects and quick-start example if needed."""
    prefix = llvm_prefix or os.environ.get("LLVM_PREFIX", _default_llvm_prefix())
    print(f"{'='*60}\nBuilding test objects\n{'='*60}", flush=True)
    _build_all(prefix)
    _build_quickstart(prefix)
    print(f"\n{'='*60}\nAll builds completed\n{'='*60}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llvm-prefix", default=None,
                        help=f"LLVM install prefix (default: {_default_llvm_prefix()})")
    args = parser.parse_args()
    ensure_built(args.llvm_prefix)


if __name__ == "__main__":
    main()
