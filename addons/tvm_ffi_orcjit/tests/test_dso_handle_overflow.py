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

GCC-compiled C++ objects reference __dso_handle via R_X86_64_PC32 (hidden
visibility, direct PC-relative).  ELFNixPlatform defines __dso_handle via a
separate DSOHandleMaterializationUnit whose allocation can land >2GB from
JIT code when:

1. A prior materialization fails (duplicate symbol), leaking its mmap slab.
2. The kept-alive traceback (pytest, sys.exc_info) prevents the old LLJIT's
   memory from being reclaimed.
3. The kernel places new mmap reservations far from the old ones.

A VA blocker (PROT_NONE MAP_FIXED_NOREPLACE) makes this deterministic on CI.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
from pathlib import Path

import pytest
from tvm_ffi_orcjit import ExecutionSession
from utils import build_test_objects

OBJ_DIR = build_test_objects()

_is_linux_x86 = sys.platform == "linux" and "x86_64" in (
    __import__("platform").machine(),
    "",
)

_PROT_NONE = 0
_MAP_PRIVATE_ANON = 0x22  # MAP_PRIVATE | MAP_ANONYMOUS
_MAP_FIXED_NOREPLACE = 0x100000
_BLOCK_RADIUS = 2 * 1024 * 1024 * 1024  # 2 GB — must exceed Delta32 ±2GB range


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


def _discover_cpp_gcc_variants() -> list[str]:
    return [
        s
        for s in ["cc-gcc"]
        if (OBJ_DIR / s / "test_funcs.o").exists()
    ]


_cpp_gcc_variants = _discover_cpp_gcc_variants()


@pytest.mark.skipif(not _is_linux_x86, reason="Linux x86_64 only")
@pytest.mark.skipif(not _cpp_gcc_variants, reason="No GCC C++ objects available")
@pytest.mark.parametrize("variant", _cpp_gcc_variants)
def test_dso_handle_overflow_with_leaked_sessions(variant: str) -> None:
    """Demonstrate Delta32 overflow for __dso_handle after leaked materializations.

    Steps:
    1. Create sessions with failed materializations (duplicate symbol).
       Keep them alive to leak their mmap slabs.
    2. Block nearby VA so the next session's allocations scatter.
    3. Create a fresh session and load GCC C++ objects into two libraries.
    4. Without an arena allocator, the __dso_handle symbol (defined by
       ELFNixPlatform in a separate allocation) can be >2GB from JIT code,
       causing a Delta32 relocation overflow.
    """
    # Step 1: Leak JIT memory via failed materializations.
    # Keep sys.exc_info tracebacks alive (like pytest does) so old sessions'
    # LLJIT mmap regions are not freed — this is critical for triggering the
    # address space fragmentation.
    leaked = []  # (session, lib, exc_info_tuple)
    for _ in range(10):
        s = ExecutionSession()
        lib = s.create_library("warmup")
        lib.add(obj(f"{variant}/test_funcs"))
        lib.get_function("test_add")(10, 20)
        exc_info = None
        try:
            lib.add(obj(f"{variant}/test_funcs_conflict"))
        except Exception:
            exc_info = sys.exc_info()  # keeps traceback → keeps frame locals → keeps LLJIT alive
        leaked.append((s, lib, exc_info))

    # Step 2: Find where LLVM's JIT memory ended up, then block nearby VA
    # gaps so the *next* session's InProcessMemoryMapper is forced to
    # reserve a new region far from the leaked slabs.
    maps_before = set(_parse_maps())
    probe = ExecutionSession()
    probe_lib = probe.create_library("probe")
    probe_lib.add(obj(f"{variant}/test_funcs"))
    maps_after = _parse_maps()
    new_maps = _find_new_mappings(maps_before, maps_after)
    jit_center = max(s for s, e in new_maps) if new_maps else 0
    # Keep probe alive — its mmap slab adds pressure.
    leaked.append((probe, probe_lib, None))

    blockers = block_nearby_va(jit_center) if jit_center else []

    try:
        # Step 3: Fresh session — __dso_handle (allocated by ELFNixPlatform
        # in DSOHandleMaterializationUnit during createJITDylib) may end up
        # in a different slab than the code (allocated during lib.add),
        # exceeding ±2GB Delta32 range.
        session = ExecutionSession()
        lib1 = session.create_library("lib1")
        lib1.add(obj(f"{variant}/test_funcs"))
        assert lib1.get_function("test_add")(10, 20) == 30

        lib2 = session.create_library("lib2")
        lib2.add(obj(f"{variant}/test_funcs_conflict"))
        assert lib2.get_function("test_add")(10, 20) == 1030
    finally:
        free_blockers(blockers)
