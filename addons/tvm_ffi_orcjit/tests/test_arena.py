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
"""Tests for JIT memory arena — deterministic relocation overflow reproduction.

Tests are grouped into two categories:

1. **VA blocker tests** (Linux only): After the first JIT allocation, parse
   /proc/self/maps, find all free VA gaps within ±5GB of the allocation, and
   fill them with PROT_NONE mappings.  This forces the next mmap to land far
   away, triggering PC-relative relocation overflow.

   - AArch64: hidden-visibility ADRP overflow (R_AARCH64_ADR_PREL_PG_HI21,
     ±4GB).  LLVM issue #173269.
   - x86_64: hidden-visibility PC32 overflow (R_X86_64_PC32, ±2GB).
     Same mechanism, smaller range.

2. **Leaked-materialization tests** (cross-platform): A prior JIT session
   triggers a duplicate-symbol error, leaking mmap'd memory.  Subsequent
   sessions allocate at progressively higher addresses, eventually exceeding
   the ±2GB range of Delta32 relocations for symbols like __dso_handle.

   - x86_64 GCC only: __dso_handle Delta32 overflow (/root/error.md).

Reference: LLVM issue #173269, lhames' comment on MapperJITLinkMemoryManager.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import sys

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
    """Discover available C-only compiler variants (for fake_fatbin)."""
    return [
        s
        for s in _KNOWN_SUBDIRS
        if s.startswith("c") and not s.startswith("cc") and (OBJ_DIR / s / "test_funcs.o").exists()
    ]


def _discover_cpp_variants() -> list[str]:
    """Discover available C++ compiler variants (for __dso_handle tests)."""
    return [s for s in _KNOWN_SUBDIRS if s.startswith("cc") and (OBJ_DIR / s / "test_funcs.o").exists()]


def _discover_all_variants() -> list[str]:
    """Discover all available compiler variants."""
    return [s for s in _KNOWN_SUBDIRS if (OBJ_DIR / s / "test_funcs.o").exists()]


_c_variants = _discover_c_variants()
_cpp_variants = _discover_cpp_variants()
_all_variants = _discover_all_variants()

# ---------------------------------------------------------------------------
# VA blocker — fills nearby free VA gaps to force distant mmap placement
# ---------------------------------------------------------------------------

_is_linux = sys.platform == "linux"
_is_aarch64 = platform.machine() in ("aarch64", "arm64")
# VA blocker tests aggressively fill process VA space and may crash in constrained
# environments (e.g., cibuildwheel containers).  Require explicit opt-in.
_va_blocker_enabled = _is_linux and os.environ.get("TVM_ORCJIT_VA_BLOCKER_TESTS") == "1"

# Radius around first allocation to block: must exceed relocation range.
# AArch64 ADRP: ±4GB → block 5GB radius
# x86_64 PC32/PLT32: ±2GB → block 3GB radius
_BLOCK_RADIUS = (5 if _is_aarch64 else 3) * (1 << 30)

_PROT_NONE = 0
_MAP_PRIVATE_ANON = 0x22  # MAP_PRIVATE | MAP_ANONYMOUS
_MAP_FIXED_NOREPLACE = 0x100000


def _get_libc():
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
    with open("/proc/self/maps") as f:
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
# Test 1: ADRP overflow via hidden-visibility cross-object reference
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _va_blocker_enabled, reason="set TVM_ORCJIT_VA_BLOCKER_TESTS=1 to enable")
@pytest.mark.parametrize("variant", _c_variants)
def test_adrp_hidden_symbol_overflow(variant: str) -> None:
    """Hidden-visibility ADRP overflows when objects are >4GB apart.

    test_hidden_caller.c takes the address of hidden_helper_add via
    ADRP+ADD (R_AARCH64_ADR_PREL_PG_HI21).  JITLink does NOT create
    a GOT stub for hidden symbols — the ADRP directly encodes the
    offset to the target.  When the two objects are in different mmap
    slabs >4GB apart, the 21-bit page offset overflows.

    This reproduces LLVM issue #173269 deterministically.

    Without arena: FAILS (ADRP overflow → segfault or JITLink error).
    With arena:    PASSES (both objects in same contiguous VA region).
    """
    # Snapshot maps before any JIT activity
    maps_before = set((s, e) for s, e in _parse_maps())

    session = ExecutionSession()
    lib = session.create_library("hidden_test")

    # Load helper object and FORCE MATERIALIZATION by looking up its function.
    # ORC JIT materializes lazily — without this lookup, both objects would
    # be materialized back-to-back at get_function() time, defeating the
    # VA blocker placed between add() calls.
    lib.add(obj(f"{variant}/test_hidden_helper"))
    helper_fn = lib.get_function("hidden_add")
    assert helper_fn(1, 2) == 3  # forces materialization → mmap at address A

    # Find where LLVM placed the helper's allocation
    maps_after = _parse_maps()
    new_maps = _find_new_mappings(maps_before, maps_after)
    if new_maps:
        jit_region_center = max(s for s, e in new_maps)
    else:
        jit_region_center = 0xFFFF00000000

    # Block all free VA gaps near the helper allocation
    blockers = block_nearby_va(jit_region_center)
    try:
        # Load caller object — mmap MUST land >4GB from helper
        lib.add(obj(f"{variant}/test_hidden_caller"))

        # Trigger materialization — ADRP relocation applied here
        fn = lib.get_function("call_hidden_add")
        assert fn(10, 20) == 30
    finally:
        free_blockers(blockers)


# ---------------------------------------------------------------------------
# Test 2: Arena immunity to VA fragmentation (meta-test)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _va_blocker_enabled, reason="set TVM_ORCJIT_VA_BLOCKER_TESTS=1 to enable")
@pytest.mark.parametrize("variant", _c_variants)
def test_adrp_immune_with_arena(variant: str) -> None:
    """With arena, VA fragmentation does not cause ADRP overflow.

    Same setup as test_adrp_hidden_symbol_overflow, but once the arena
    is implemented, this test passes because all JIT allocations land
    in the same contiguous VA region regardless of VA fragmentation.

    Without arena: FAILS (same as test_adrp_hidden_symbol_overflow).
    With arena:    PASSES.
    """
    maps_before = set((s, e) for s, e in _parse_maps())

    session = ExecutionSession()
    lib = session.create_library("hidden_test")

    # Force materialization of helper first
    lib.add(obj(f"{variant}/test_hidden_helper"))
    assert lib.get_function("hidden_add")(1, 2) == 3

    maps_after = _parse_maps()
    new_maps = _find_new_mappings(maps_before, maps_after)
    jit_region_center = max(s for s, e in new_maps) if new_maps else 0xFFFF00000000

    blockers = block_nearby_va(jit_region_center)
    try:
        lib.add(obj(f"{variant}/test_hidden_caller"))
        fn = lib.get_function("call_hidden_add")
        assert fn(10, 20) == 30
    finally:
        free_blockers(blockers)


# ---------------------------------------------------------------------------
# Test 3: Cross-library calls (GOT-mediated, should always work)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _va_blocker_enabled, reason="set TVM_ORCJIT_VA_BLOCKER_TESTS=1 to enable")
@pytest.mark.parametrize("variant", _c_variants)
def test_cross_library_got_mediated(variant: str) -> None:
    """Cross-library calls via GOT/stubs work even with distant slabs.

    JITLink creates per-object GOT entries and PLT stubs for
    default-visibility cross-JITDylib calls.  The GOT is always in
    the caller's slab, so ADRP to the GOT is always in range.
    The final jump uses an absolute address (BR x16 on AArch64).

    This test verifies that GOT-mediated calls are NOT affected
    by VA fragmentation — they work with or without the arena.

    Note: uses C-only variants.  C++ (GCC) objects reference __dso_handle
    via ADRP (not GOT), which overflows independently of the cross-library
    call mechanism.
    """
    maps_before = set((s, e) for s, e in _parse_maps())

    session = ExecutionSession()

    lib_base = session.create_library("base")
    lib_base.add(obj(f"{variant}/test_link_order_base"))
    # Force materialization of base library
    lib_base.get_function("helper_add")(1, 2)

    maps_after = _parse_maps()
    new_maps = _find_new_mappings(maps_before, maps_after)
    jit_region_center = max(s for s, e in new_maps) if new_maps else 0xFFFF00000000

    blockers = block_nearby_va(jit_region_center)
    try:
        lib_caller = session.create_library("caller")
        lib_caller.set_link_order(lib_base)
        lib_caller.add(obj(f"{variant}/test_link_order_caller"))

        cross_add = lib_caller.get_function("cross_lib_add")
        assert cross_add(10, 20) == 30
    finally:
        free_blockers(blockers)


# ---------------------------------------------------------------------------
# Test 4: Large data section (simulated .nv_fatbin)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Phase 3: requires overflow region for large sections")
@pytest.mark.parametrize("variant", _c_variants)
def test_large_data_section(variant: str) -> None:
    """Load object with a 4MB .nv_fatbin section — basic correctness.

    The .nv_fatbin section is referenced only by absolute relocations,
    so it can live anywhere.  This test verifies the object loads and
    the function works.

    Phase 2 (arena only): passes, but .nv_fatbin consumes arena space.
    Phase 3 (arena + overflow): passes, .nv_fatbin goes to overflow region.
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
    eventually exceeding the ±2GB PC32 range.

    The arena prevents this because all allocations are within a
    contiguous 256MB region regardless of prior leaks.

    Without arena: may FAIL on x86_64 GCC after repeated leaked
                   materializations push slabs >2GB apart.
    With arena:    PASSES (all allocations in same arena).
    """
    # Step 1: Trigger leaked materializations to consume low VA space.
    # Each failed load leaks an mmap slab that can't be reclaimed.
    # Keep sessions alive — destroying a session with leaked materialization
    # state causes LLVM to crash during ORC runtime teardown.
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
