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
"""Reproducer for __dso_handle Delta32 relocation overflow on x86_64.

GCC-compiled C++ objects built with -fpie reference __dso_handle via
R_X86_64_PC32 (hidden visibility, direct PC-relative, ±2GB range).
ELFNixPlatform defines __dso_handle via a separate
DSOHandleMaterializationUnit whose allocation can land >2GB from
JIT code when a VA blocker forces distant mmap placement.

With -fPIC (default), GCC uses R_X86_64_GOTPCRELX (GOT-relative) which
has no ±2GB limit.  Using -fpie forces the problematic PC32 relocations.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import platform
import sys
from pathlib import Path

import pytest
from tvm_ffi_orcjit import ExecutionSession
from utils import build_test_objects

OBJ_DIR = build_test_objects()

_is_linux_x86 = sys.platform == "linux" and platform.machine() in ("x86_64", "AMD64")

_PROT_NONE = 0
_MAP_PRIVATE_ANON = 0x22  # MAP_PRIVATE | MAP_ANONYMOUS
_MAP_FIXED_NOREPLACE = 0x100000
_BLOCK_RADIUS = 3 * 1024 * 1024 * 1024  # 3 GB — exceeds Delta32 ±2GB range


def _get_libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    libc.munmap.restype = ctypes.c_int
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    return libc


def _parse_maps() -> list[tuple[int, int]]:
    regions = []
    with Path("/proc/self/maps").open() as f:
        for line in f:
            addrs = line.split()[0].split("-")
            regions.append((int(addrs[0], 16), int(addrs[1], 16)))
    return sorted(regions)


def _find_new_mappings(
    before: set[tuple[int, int]], after: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    return [(s, e) for s, e in after if (s, e) not in before]


def block_nearby_va(center: int, radius: int = _BLOCK_RADIUS) -> list[tuple[int, int]]:
    """Block free VA gaps near *center* to force distant mmap placement."""
    libc = _get_libc()
    maps = _parse_maps()
    blockers: list[tuple[int, int]] = []
    low = max(center - radius, 0)
    high = center + radius

    for i in range(len(maps) - 1):
        gap_start = maps[i][1]
        gap_end = maps[i + 1][0]
        if gap_end <= low or gap_start >= high or gap_end <= gap_start:
            continue
        block_start = max(gap_start, low)
        block_end = min(gap_end, high)
        block_size = block_end - block_start
        if block_size <= 0:
            continue
        addr = libc.mmap(
            block_start, block_size, _PROT_NONE, _MAP_PRIVATE_ANON | _MAP_FIXED_NOREPLACE, -1, 0
        )
        if addr != ctypes.c_void_p(-1).value and addr is not None:
            blockers.append((addr, block_size))
    return blockers


def free_blockers(blockers: list[tuple[int, int]]) -> None:
    libc = _get_libc()
    for addr, size in blockers:
        libc.munmap(addr, size)


def obj(name: str) -> str:
    path = OBJ_DIR / f"{name}.o"
    if not path.exists():
        pytest.skip(f"{path.name} not found (not built)")
    return str(path)


def _discover_pie_cpp_variants() -> list[str]:
    return [s for s in ["cc-gcc-pie"] if (OBJ_DIR / s / "test_funcs.o").exists()]


_pie_cpp_variants = _discover_pie_cpp_variants()


@pytest.mark.skipif(not _is_linux_x86, reason="Linux x86_64 only")
@pytest.mark.skipif(not _pie_cpp_variants, reason="No GCC PIE C++ objects available")
@pytest.mark.parametrize("variant", _pie_cpp_variants)
def test_dso_handle_pc32_overflow(variant: str) -> None:
    """__dso_handle PC32 overflows under VA pressure without arena.

    PIE GCC objects (-fpie) use R_X86_64_PC32 for __dso_handle (±2GB
    range), unlike -fPIC objects which use R_X86_64_GOTPCRELX.
    A 3GB VA blocker pushes the second library's code >2GB from
    __dso_handle (materialized with the first library), causing
    a Delta32 relocation overflow.
    """
    maps_before = set(_parse_maps())

    session = ExecutionSession()
    lib1 = session.create_library("lib1")
    lib1.add(obj(f"{variant}/test_funcs"))
    assert lib1.get_function("test_add")(10, 20) == 30

    # Block 3GB of VA around the first allocation to force scatter
    maps_after = _parse_maps()
    new_maps = _find_new_mappings(maps_before, maps_after)
    jit_center = max(s for s, e in new_maps) if new_maps else 0xFFFF00000000

    blockers = block_nearby_va(jit_center, radius=_BLOCK_RADIUS)
    try:
        lib2 = session.create_library("lib2")
        lib2.add(obj(f"{variant}/test_funcs_conflict"))
        assert lib2.get_function("test_add")(10, 20) == 1030
    finally:
        free_blockers(blockers)
