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
"""Tests for JIT memory arena — verifies co-location and relocation safety.

Tests verify the arena memory manager by:

1. **Co-location tests**: Load objects into separate libraries, get their code
   addresses, and verify they are within the arena range.  A 256MB VA blocker
   between loads forces the default allocator to scatter objects, while the
   arena keeps them close.

2. **Before/after tests**: Compare arena-on vs arena-off behavior.  With arena
   disabled (arena_size=-1), objects scatter beyond 256MB.  With arena enabled,
   objects stay within the arena range.

3. **Hidden-symbol tests**: Load hidden-visibility objects with VA blocker and
   call across them — verifies ADRP/PC32 relocations stay in range.

All tests use a small arena (16MB) and 256MB VA blocker — safe for CI containers.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
from pathlib import Path

import pytest
from tvm_ffi_orcjit import ExecutionSession
from utils import build_test_objects

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

OBJ_DIR = build_test_objects()

_KNOWN_SUBDIRS = [
    "c",
    "c-gcc",
    "cc",
    "cc-gcc",
    "c-appleclang",
    "cc-appleclang",
    "c-msvc",
    "c-clang-cl",
]


def obj(name: str) -> str:
    """Return path to a pre-built test object file, or skip if missing."""
    path = OBJ_DIR / f"{name}.o"
    if not path.exists():
        pytest.skip(f"{path.name} not found (not built)")
    return str(path)


def _discover_c_variants() -> list[str]:
    """Discover available C-only compiler variants."""
    return [
        s
        for s in _KNOWN_SUBDIRS
        if s.startswith("c") and not s.startswith("cc") and (OBJ_DIR / s / "test_funcs.o").exists()
    ]


def _discover_cpp_variants() -> list[str]:
    """Discover available C++ compiler variants (for __dso_handle tests)."""
    return [
        s for s in _KNOWN_SUBDIRS if s.startswith("cc") and (OBJ_DIR / s / "test_funcs.o").exists()
    ]


_c_variants = _discover_c_variants()
_cpp_variants = _discover_cpp_variants()
_all_variants = _c_variants + _cpp_variants

_is_linux = sys.platform == "linux"

# Arena test parameters
_ARENA_SIZE = 16 * 1024 * 1024  # 16MB — small arena for testing
_BLOCK_RADIUS = 256 * 1024 * 1024  # 256MB — safe for CI containers

_PROT_NONE = 0
_MAP_PRIVATE_ANON = 0x22  # MAP_PRIVATE | MAP_ANONYMOUS
_MAP_FIXED_NOREPLACE = 0x100000


# ---------------------------------------------------------------------------
# VA blocker — fills nearby free VA gaps to force distant mmap placement
# ---------------------------------------------------------------------------


def _get_libc() -> ctypes.CDLL:
    """Get a ctypes handle to libc with correct mmap/munmap signatures."""
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
    """Parse /proc/self/maps into sorted list of (start, end) tuples."""
    regions = []
    with Path("/proc/self/maps").open() as f:
        for line in f:
            addrs = line.split()[0].split("-")
            regions.append((int(addrs[0], 16), int(addrs[1], 16)))
    return sorted(regions)


def _find_new_mappings(
    before: set[tuple[int, int]], after: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Find mappings present in *after* but not in *before*."""
    return [(s, e) for s, e in after if (s, e) not in before]


def block_nearby_va(center: int, radius: int = _BLOCK_RADIUS) -> list[tuple[int, int]]:
    """Block all free VA gaps within *radius* of *center*.

    Uses MAP_FIXED_NOREPLACE to place PROT_NONE mappings in every free gap
    within [center - radius, center + radius].  This forces subsequent
    mmap(NULL, ...) calls to land outside the blocked region.

    Returns list of (addr, size) blockers to be freed later.
    """
    libc = _get_libc()
    maps = _parse_maps()
    blockers = []
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
    """Free all VA blockers."""
    libc = _get_libc()
    for addr, size in blockers:
        libc.munmap(addr, size)


# ---------------------------------------------------------------------------
# Test 1: Arena co-location — objects stay within arena range
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _is_linux, reason="Arena is Linux-only")
@pytest.mark.parametrize("variant", _all_variants)
def test_arena_colocation(variant: str) -> None:
    """With arena, objects in separate libraries have close code addresses.

    Uses a 16MB arena and inserts a 256MB VA blocker between object loads.
    Without the arena, the blocker would push the second object far away.
    With the arena, both objects land within the 16MB region.
    """
    maps_before = set(_parse_maps())

    session = ExecutionSession(arena_size=_ARENA_SIZE)
    lib1 = session.create_library("lib1")
    lib1.add(obj(f"{variant}/test_addr"))
    addr1 = lib1.get_function("code_address")()

    # Find where LLVM placed the first allocation and block nearby VA
    maps_after = _parse_maps()
    new_maps = _find_new_mappings(maps_before, maps_after)
    jit_center = max(s for s, e in new_maps) if new_maps else addr1

    blockers = block_nearby_va(jit_center)
    try:
        lib2 = session.create_library("lib2")
        lib2.add(obj(f"{variant}/test_addr"))
        addr2 = lib2.get_function("code_address")()
    finally:
        free_blockers(blockers)

    distance = abs(addr1 - addr2)
    assert distance < _ARENA_SIZE, (
        f"Objects should be within {_ARENA_SIZE} bytes, "
        f"but distance is {distance} ({distance / (1024**2):.1f} MB)"
    )


# ---------------------------------------------------------------------------
# Test 2: Without arena, objects scatter beyond blocker radius
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _is_linux, reason="Arena is Linux-only")
@pytest.mark.parametrize("variant", _all_variants)
def test_no_arena_scattered(variant: str) -> None:
    """Without arena, VA blocker pushes objects far apart.

    Combined with test_arena_colocation, this proves the arena is the
    cause of co-location — not lucky mmap placement.
    """
    maps_before = set(_parse_maps())

    session = ExecutionSession(arena_size=-1)  # arena disabled
    lib1 = session.create_library("lib1")
    lib1.add(obj(f"{variant}/test_addr"))
    addr1 = lib1.get_function("code_address")()

    maps_after = _parse_maps()
    new_maps = _find_new_mappings(maps_before, maps_after)
    jit_center = max(s for s, e in new_maps) if new_maps else addr1

    blockers = block_nearby_va(jit_center)
    try:
        lib2 = session.create_library("lib2")
        lib2.add(obj(f"{variant}/test_addr"))
        addr2 = lib2.get_function("code_address")()
    finally:
        free_blockers(blockers)

    distance = abs(addr1 - addr2)
    # Objects should be far apart — at least 128MB (half the blocker radius).
    # The exact distance depends on kernel mmap placement, but it must be
    # much larger than the 16MB arena size to prove the arena is responsible.
    min_expected = _BLOCK_RADIUS // 2
    assert distance > min_expected, (
        f"Without arena, objects should be >{min_expected} bytes apart, "
        f"but distance is {distance} ({distance / (1024**2):.1f} MB)"
    )


# ---------------------------------------------------------------------------
# Test 3: Hidden-symbol ADRP/PC32 relocation with arena + blocker
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _is_linux, reason="Arena is Linux-only")
@pytest.mark.parametrize("variant", _all_variants)
def test_arena_hidden_symbol_with_blocker(variant: str) -> None:
    """Arena prevents hidden-visibility relocation overflow under VA pressure.

    Loads two objects with hidden-visibility cross-references (ADRP+ADD
    on AArch64, PC32 on x86_64) with a VA blocker between them.
    Without arena, the blocker would push objects apart causing overflow.
    With the arena, both objects are co-located and the call succeeds.
    """
    maps_before = set(_parse_maps())

    session = ExecutionSession(arena_size=_ARENA_SIZE)
    lib = session.create_library("hidden_test")

    # Load helper and force materialization
    lib.add(obj(f"{variant}/test_hidden_helper"))
    assert lib.get_function("hidden_add")(1, 2) == 3

    # Block nearby VA to force scatter
    maps_after = _parse_maps()
    new_maps = _find_new_mappings(maps_before, maps_after)
    jit_center = max(s for s, e in new_maps) if new_maps else 0xFFFF00000000

    blockers = block_nearby_va(jit_center)
    try:
        lib.add(obj(f"{variant}/test_hidden_caller"))
        fn = lib.get_function("call_hidden_add")
        assert fn(10, 20) == 30
    finally:
        free_blockers(blockers)


# ---------------------------------------------------------------------------
# Test 4: Large data section (simulated .nv_fatbin)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _is_linux, reason="Arena is Linux-only")
@pytest.mark.parametrize("variant", _all_variants)
def test_large_data_section(variant: str) -> None:
    """Load object with a 4MB .nv_fatbin section — basic correctness.

    The .nv_fatbin section is referenced only by absolute relocations,
    so it can live anywhere.  This test verifies the object loads and
    the function works.  The 4MB section fits in the 256MB arena.
    """
    session = ExecutionSession()
    lib = session.create_library("fatbin")
    lib.add(obj(f"{variant}/fake_fatbin"))
    fn = lib.get_function("get_fatbin_size")
    assert fn() == 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# Test 5: __dso_handle Delta32 overflow after leaked materialization
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Leaked materialization crashes LLVM ORC runtime on session teardown")
@pytest.mark.skipif(not _is_linux, reason="ELF/GCC-specific __dso_handle test")
@pytest.mark.parametrize("variant", _cpp_variants)
def test_dso_handle_relocation_after_failed_materialization(variant: str) -> None:
    """__dso_handle PC32 relocation works after leaked JIT memory.

    GCC-compiled C++ objects emit R_X86_64_PC32 relocations for
    __dso_handle (used by __cxa_atexit for DSO identification).
    When a prior materialization fails (duplicate symbol), LLVM leaks
    the mmap'd slab.  Subsequent allocations land at higher addresses,
    eventually exceeding the +/-2GB PC32 range.

    The arena prevents this because all allocations are within a
    contiguous 256MB region regardless of prior leaks.

    Without arena: may FAIL on x86_64 GCC after repeated leaked
                   materializations push slabs >2GB apart.
    With arena:    PASSES (all allocations in same arena).
    """
    # Step 1: Trigger leaked materializations to consume low VA space.
    leaked_sessions = []
    for i in range(3):
        s0 = ExecutionSession()
        lib0 = s0.create_library("warmup")
        lib0.add(obj(f"{variant}/test_funcs"))
        lib0.get_function("test_add")(10, 20)
        try:
            lib0.add(obj(f"{variant}/test_funcs_conflict"))
        except Exception:
            pass
        leaked_sessions.append((s0, lib0))

    # Step 2: Fresh session — cross-library resolution must still work.
    session = ExecutionSession()
    lib1 = session.create_library("lib1")
    lib1.add(obj(f"{variant}/test_funcs"))
    assert lib1.get_function("test_add")(10, 20) == 30

    lib2 = session.create_library("lib2")
    lib2.add(obj(f"{variant}/test_funcs_conflict"))
    assert lib2.get_function("test_add")(10, 20) == 1030
