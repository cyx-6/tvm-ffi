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
"""Tests for the process-wide shared ExecutionSession (``global_session``).

Pins three contracts: (1) ``global_session()`` always returns the same
underlying session; (2) two libraries created on it can ``set_link_order``
across each other (impossible across separate sessions); (3) concurrent
create/load/call/drop from multiple threads is safe (a regression guard for the
mutex on the pending init/fini maps — a liveness/crash smoke test, since the GIL
can mask the underlying C++ race).
"""

from __future__ import annotations

import gc
import threading

import pytest

# Reuse the compiler-variant matrix and per-object helpers from test_basic so
# the global-session tests run across the same C / C++ / multi-compiler variants
# the rest of the suite covers (importing test_basic also builds the objects).
from test_basic import Variant, _all_variants, _variant_id, obj
from tvm_ffi_orcjit import ExecutionSession, global_session

# ---------------------------------------------------------------------------
# Identity — the singleton contract
# ---------------------------------------------------------------------------


def test_global_session_is_singleton() -> None:
    """Repeated global_session() calls return the same underlying session."""
    a = global_session()
    b = global_session()
    assert a.same_as(b)
    # __wrapped__ bypasses functools.cache, re-crossing FFI into C++ Global() so
    # the singleton is proven in C++, not just by the Python cache.
    c = global_session.__wrapped__()
    d = global_session.__wrapped__()
    assert c.same_as(d)
    assert c.same_as(a)


def test_independent_session_is_not_global() -> None:
    """A freshly constructed ExecutionSession is not the shared global one."""
    assert not ExecutionSession().same_as(global_session())


# ---------------------------------------------------------------------------
# Cross-library linking on the shared session
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_global_session_cross_library_link_order(v: Variant) -> None:
    """Two libraries on the global session resolve symbols across each other.

    This is the capability separate sessions cannot provide: a caller library
    links against a base library because both live in the one shared
    ``llvm::orc::ExecutionSession``.
    """
    sess = global_session()
    # Auto-named libraries avoid name collisions in the shared namespace.
    base = sess.create_library()
    base.add(obj(v.link_order_base_obj()))

    caller = sess.create_library()
    caller.set_link_order(base)
    caller.add(obj(v.link_order_caller_obj()))

    cross_add = caller.get_function(v.fn("cross_lib_add"))
    assert cross_add(10, 20) == 30
    assert cross_add(100, 200) == 300

    # Per-dylib teardown on the shared session must not disturb it.
    del caller, base
    gc.collect()


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_global_session_load_and_execute(v: Variant) -> None:
    """A library created on the global session loads and runs normally."""
    sess = global_session()
    lib = sess.create_library()
    lib.add(obj(v.funcs_obj()))
    assert lib.get_function(v.fn("test_add"))(10, 20) == 30
    assert lib.get_function(v.fn("test_multiply"))(7, 6) == 42
    del lib
    gc.collect()


# ---------------------------------------------------------------------------
# Concurrency — multiple threads sharing one session
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", _all_variants[:1], ids=_variant_id)
def test_global_session_concurrent_create_and_call(v: Variant) -> None:
    """Many threads create / load / call / drop libraries on the shared session.

    The FFI call path releases the GIL, so these workers genuinely run
    createJITDylib / addObjectFile / lookup / removeJITDylib concurrently on the
    one shared ExecutionSession. Those compound operations are not atomic with
    respect to one another, and overlapping them corrupts linker state — an
    abort on macOS's strict allocator (glibc tolerated it). The session-wide
    lock added for the shared session serializes them; with it every worker
    completes.
    """
    sess = global_session()
    funcs = obj(v.funcs_obj())  # resolve path once; build is not thread-safe
    n_threads = 4
    n_iters = 8
    errors: list[BaseException] = []
    barrier = threading.Barrier(n_threads)

    def worker(tid: int) -> None:
        try:
            barrier.wait()  # release all threads together to maximize overlap
            for j in range(n_iters):
                lib = sess.create_library()
                lib.add(funcs)
                assert lib.get_function(v.fn("test_add"))(tid, j) == tid + j
                assert lib.get_function(v.fn("test_multiply"))(tid, j) == tid * j
                del lib
        except BaseException as exc:  # surface to the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    gc.collect()
    assert not errors, errors
