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
"""Basic tests for tvm-ffi-orcjit functionality."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import tvm_ffi
from tvm_ffi_orcjit import ExecutionSession

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_DIR = Path(__file__).parent

skip_on_windows = pytest.mark.skipif(
    sys.platform == "win32",
    reason="C++ object files not built on Windows (C-only strategy)",
)


def obj(name: str) -> str:
    """Return path to a pre-built test object file, or skip if missing."""
    path = TEST_DIR / f"{name}.o"
    if not path.exists():
        pytest.skip(f"{path.name} not found (not built)")
    return str(path)


def make_lib(*obj_names: str, session: ExecutionSession | None = None, name: str = ""):
    """Create a library and load one or more object files into it."""
    if session is None:
        session = ExecutionSession()
    lib = session.create_library(name)
    for o in obj_names:
        lib.add(obj(o))
    return session, lib


# ---------------------------------------------------------------------------
# Variants: C uses "_c" suffix on object files and function names;
# C++ uses no suffix.  Tests are parametrized over both variants where the
# logic is identical.
# ---------------------------------------------------------------------------


class Variant:
    """Describes a C or C++ test variant (object file / function name mapping)."""

    def __init__(self, suffix: str):
        self.suffix = suffix  # "" for C++, "_c" for C

    def funcs_obj(self):
        return f"test_funcs{self.suffix}"

    def funcs2_obj(self):
        return f"test_funcs2{self.suffix}"

    def conflict_obj(self):
        return f"test_funcs_conflict{self.suffix}"

    def call_global_obj(self):
        return f"test_call_global{self.suffix}"

    def fn(self, base_name: str) -> str:
        return f"{base_name}{self.suffix}"

    def __repr__(self) -> str:
        return "C" if self.suffix else "C++"


CPP = Variant("")
C = Variant("_c")

# On Windows, only C variant is available
_all_variants = [CPP, C] if sys.platform != "win32" else [C]
_cpp_only = [CPP] if sys.platform != "win32" else []


def _variant_id(v: Variant) -> str:
    return repr(v)


# ---------------------------------------------------------------------------
# Session / library creation
# ---------------------------------------------------------------------------


def test_create_session() -> None:
    session = ExecutionSession()
    assert session is not None


def test_create_library() -> None:
    session = ExecutionSession()
    lib = session.create_library()
    assert lib is not None


def test_multiple_libraries() -> None:
    session = ExecutionSession()
    lib1 = session.create_library("lib1")
    lib2 = session.create_library("lib2")
    assert lib1 is not None
    assert lib2 is not None


# ---------------------------------------------------------------------------
# Load & execute — parametrized over C / C++
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_load_and_execute(v: Variant) -> None:
    _, lib = make_lib(v.funcs_obj())
    assert lib.get_function(v.fn("test_add"))(10, 20) == 30
    assert lib.get_function(v.fn("test_multiply"))(7, 6) == 42


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_load_and_execute_second_set(v: Variant) -> None:
    _, lib = make_lib(v.funcs2_obj())
    assert lib.get_function(v.fn("test_subtract"))(10, 3) == 7
    assert lib.get_function(v.fn("test_divide"))(20, 4) == 5


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_function_not_found(v: Variant) -> None:
    _, lib = make_lib(v.funcs_obj())
    with pytest.raises(AttributeError, match="Module has no function"):
        lib.get_function("nonexistent_function")


# ---------------------------------------------------------------------------
# Multi-object / multi-library — parametrized over C / C++
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_gradually_add_objects(v: Variant) -> None:
    session, lib = make_lib(v.funcs_obj())

    add_func = lib.get_function(v.fn("test_add"))
    mul_func = lib.get_function(v.fn("test_multiply"))
    assert add_func(5, 3) == 8
    assert mul_func(4, 5) == 20

    lib.add(obj(v.funcs2_obj()))
    assert lib.get_function(v.fn("test_subtract"))(10, 3) == 7
    assert lib.get_function(v.fn("test_divide"))(20, 4) == 5

    # First object's functions still work
    assert add_func(10, 20) == 30
    assert mul_func(7, 6) == 42


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_two_separate_libraries(v: Variant) -> None:
    session = ExecutionSession()
    _, lib1 = make_lib(v.funcs_obj(), session=session, name="lib1")
    _, lib2 = make_lib(v.funcs2_obj(), session=session, name="lib2")

    assert lib1.get_function(v.fn("test_add"))(5, 3) == 8
    assert lib1.get_function(v.fn("test_multiply"))(4, 5) == 20
    assert lib2.get_function(v.fn("test_subtract"))(10, 3) == 7
    assert lib2.get_function(v.fn("test_divide"))(20, 4) == 5

    with pytest.raises(AttributeError, match="Module has no function"):
        lib1.get_function(v.fn("test_subtract"))
    with pytest.raises(AttributeError, match="Module has no function"):
        lib2.get_function(v.fn("test_add"))


# ---------------------------------------------------------------------------
# Symbol conflicts — parametrized over C / C++
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_symbol_conflict_same_library(v: Variant) -> None:
    _, lib = make_lib(v.funcs_obj())
    assert lib.get_function(v.fn("test_add"))(10, 20) == 30
    with pytest.raises(Exception):
        lib.add(obj(v.conflict_obj()))


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_symbol_conflict_different_libraries(v: Variant) -> None:
    session = ExecutionSession()
    _, lib1 = make_lib(v.funcs_obj(), session=session, name="lib1")
    _, lib2 = make_lib(v.conflict_obj(), session=session, name="lib2")

    assert lib1.get_function(v.fn("test_add"))(10, 20) == 30
    assert lib2.get_function(v.fn("test_add"))(10, 20) == 1030
    assert lib1.get_function(v.fn("test_multiply"))(5, 6) == 30
    assert lib2.get_function(v.fn("test_multiply"))(5, 6) == 60


# ---------------------------------------------------------------------------
# Global function callbacks — parametrized over C / C++
# ---------------------------------------------------------------------------


@pytest.fixture()
def _register_host_functions():
    """Register host add/multiply functions for JIT code to call."""

    @tvm_ffi.register_global_func("test_host_add", override=True)
    def _host_add(a: int, b: int) -> int:
        return a + b

    @tvm_ffi.register_global_func("test_host_multiply", override=True)
    def _host_mul(a: int, b: int) -> int:
        return a * b


@pytest.mark.usefixtures("_register_host_functions")
@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_call_global(v: Variant) -> None:
    _, lib = make_lib(v.call_global_obj())
    add_func = lib.get_function(v.fn("test_call_global_add"))
    assert add_func(10, 20) == 30
    assert add_func(100, 200) == 300

    mul_func = lib.get_function(v.fn("test_call_global_mul"))
    assert mul_func(7, 6) == 42
    assert mul_func(11, 11) == 121


# ---------------------------------------------------------------------------
# CUDA (optional)
# ---------------------------------------------------------------------------


def test_load_and_execute_cuda_function() -> None:
    _, lib = make_lib("test_funcs_cuda")
    assert lib.get_function("test_add")(10, 20) == 30
    assert lib.get_function("test_multiply")(7, 6) == 42


# ---------------------------------------------------------------------------
# Constructor / destructor (C++ only, Linux/macOS)
# ---------------------------------------------------------------------------


@skip_on_windows
def test_ctor_dtor() -> None:
    log = ""

    @tvm_ffi.register_global_func("append_log")
    def _append_ctor_log(x: str) -> None:
        nonlocal log
        log += x

    _, lib = make_lib("test_ctor_dtor")
    lib.get_function("main")()
    del lib

    if sys.platform == "linux":
        main_idx = log.index("<main>")
        pre = log[:main_idx]
        post = log[main_idx:]
        assert pre.index("<init_array.101>") < pre.index("<init_array.102>")
        assert pre.index("<init_array.102>") < pre.index("<init_array.103>")
        assert pre.index("<init_array.103>") < pre.index("<init_array>")
        assert pre.index("<ctors.103>") < pre.index("<ctors.102>")
        assert pre.index("<ctors.102>") < pre.index("<ctors.101>")
        assert pre.index("<ctors.101>") < pre.index("<ctors>")
        assert "<dtors>" in post
    elif sys.platform == "darwin":
        main_idx = log.index("<main>")
        pre = log[:main_idx]
        post = log[main_idx:]
        assert pre.index("<init_array.101>") < pre.index("<init_array.102>")
        assert pre.index("<init_array.102>") < pre.index("<init_array.103>")
        assert pre.index("<init_array.103>") < pre.index("<init_array>")
        assert post.index("<fini_array>") < post.index("<fini_array.103>")
        assert post.index("<fini_array.103>") < post.index("<fini_array.102>")
        assert post.index("<fini_array.102>") < post.index("<fini_array.101>")
        assert "<ctors>" not in log
        assert "<dtors>" not in log


if __name__ == "__main__":
    test_ctor_dtor()
