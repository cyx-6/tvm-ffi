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
"""Before/after benchmark for PR #593 (PyObject-tying).

Runs identically on three configs (one binary each, head toggled by env):
  - main      (commit 88e066e): pre-PR baseline.
  - head-off  (HEAD + TVM_FFI_DISABLE_TYING=1): machinery present but inert.
  - head-on   (HEAD): full tying.

Core ops across object width (#fields) and access depth (nesting), C++->Python
callback arity, and Array indexing. Plus a C++ make_object microbench and a memory
probe. Output lines are machine-parseable by make_table.py.

Two attribute-access regimes, deliberately distinct:
  * attr_hit   -- the child wrapper is HELD alive (anchored), so each access is a
                  pure Active-hit (incref of the canonical wrapper). Tying's win.
  * attr_churn -- only the ROOT is held; each accessed wrapper dies by refcount at
                  end of iteration, so tying-on revives the cached allocation and
                  main/head-off allocate fresh. No gc.collect() in the timed loop.

Measurement order: all ops are REGISTERED first, then sampled ROUND-ROBIN -- in each
round we take one trial of every op before moving to the next round. A transient host
noise window (vCPU steal / memory-bandwidth contention) then perturbs one trial across
many ops rather than every trial of a single op, so each op's min-of-K stays robust.
This replaces the earlier "all K trials of op A, then op B" order, which let one bad
~1s window inflate an entire op (it inflated attr_churn in a prior run).

time.perf_counter_ns; min + median over K trials; gc disabled around timed regions
(tying transitions are refcount-driven, not GC-driven).
"""

from __future__ import annotations

import gc
import itertools
import os
import statistics
import sys
import time
from typing import Any, Callable

import tvm_ffi
import tvm_ffi.cpp
from tvm_ffi import dataclasses as dc

# --------------------------------------------------------------------------------------
# Axes / knobs.
# --------------------------------------------------------------------------------------

REPEAT = 25_000  # inner iterations per trial for Python ops (>=ms-scale sample even at width32)
TRIALS = 15  # min-of-K outer trials (round-robin across ops, so K need not be large)
CALL_N = 200_000  # inner iterations for the C++ invoke_n function-call loops
CPP_N = 1_000_000  # iterations for the make_object microbench (~5ns/call; sampled adaptively)

# Axes are kept linear-with-endpoints (low / mid / high): the table reads as a slope, and the
# in-between points only re-confirm it. Trim to 3 each to roughly halve the row count.
DEPTHS = [1, 3, 5]  # attribute-access nesting depth
WIDTHS = [1, 8, 32]  # object size = number of int fields
ARITIES = [0, 1, 3, 8]  # function-call argument counts


# --------------------------------------------------------------------------------------
# Output helpers (machine-parseable: "RESULT <op> <stat> <value>").
# --------------------------------------------------------------------------------------


def print_speed(name: str, speed: float) -> None:
    print(f"{name:<40} {speed * 1e9:10.1f} ns/call")


def emit(op: str, stat: str, value: float) -> None:
    print(f"RESULT\t{op}\t{stat}\t{value:.6e}")


# --------------------------------------------------------------------------------------
# Round-robin benchmark registry.
#
# Each op registers a `sample()` returning ONE sec/call measurement. The driver warms
# every op once, then runs TRIALS rounds, taking one sample per op per round, so noise
# is spread across ops rather than concentrated in one op's consecutive trials.
# --------------------------------------------------------------------------------------


class Bench:
    def __init__(self) -> None:
        self.order: list[str] = []  # Python ops, fixed-K round-robin
        self.samplers: dict[str, Callable[[], float]] = {}
        self.cpp_ops: list[tuple[str, Callable[[], float]]] = []  # adaptive floor certification
        self._keepalive: list[Any] = []  # hold anchor objects live for the whole run

    def keep(self, *objs: Any) -> None:
        self._keepalive.extend(objs)

    def add(self, op: str, sample: Callable[[], float]) -> None:
        if op in self.samplers:
            raise ValueError(f"duplicate op {op}")
        self.order.append(op)
        self.samplers[op] = sample

    def add_py(self, op: str, fn: Callable[[int], None], repeat: int = REPEAT) -> None:
        """Register a Python op: ``fn(repeat)`` runs the inner loop."""

        def sample(_fn: Any = fn, _r: int = repeat) -> float:
            t0 = time.perf_counter_ns()
            _fn(_r)
            return (time.perf_counter_ns() - t0) / _r / 1e9

        self.add(op, sample)

    def add_cpp(self, op: str, fn: Callable[[int], float]) -> None:
        """Register a C++ microbench. These are ~7ms/trial and memory-bandwidth bound, so
        they are sampled ADAPTIVELY (see _run_adaptive), not in the fixed-K round-robin."""
        self.cpp_ops.append((op, lambda _f=fn: float(_f(CPP_N))))

    def run(self, trials: int = TRIALS) -> None:
        # Python ops: fixed-K round-robin (noise spread across ops).
        for op in self.order:
            self.samplers[op]()  # warmup
        results: dict[str, list[float]] = {op: [] for op in self.order}
        gc_was = gc.isenabled()
        gc.disable()
        try:
            for _ in range(trials):
                for op in self.order:
                    results[op].append(self.samplers[op]())
        finally:
            if gc_was:
                gc.enable()
        for op in self.order:
            self._report(op, results[op])
        # C++ microbenches: adaptive floor certification.
        for op, sample in self.cpp_ops:
            self._run_adaptive(op, sample)

    def _run_adaptive(self, op: str, sample: Callable[[], float],
                      min_k: int = 15, max_k: int = 400, target: float = 0.08, batch: int = 5) -> None:
        """Sample until the FLOOR is certified -- p20 within ``target`` of min (several trials
        hit the clean floor) -- or ``max_k`` is reached. Emits converged=1/0 so the table can
        flag a cell whose floor this host slice was too contended to confirm, instead of
        silently trusting a noisy number or blindly re-running the whole matrix."""
        sample()  # warmup
        xs = [sample() for _ in range(min_k)]
        gc_was = gc.isenabled()
        gc.disable()
        try:
            while True:
                s = sorted(xs)
                p20 = s[len(s) // 5]
                converged = (p20 - s[0]) / s[0] <= target
                if converged or len(xs) >= max_k:
                    break
                xs += [sample() for _ in range(batch)]
        finally:
            if gc_was:
                gc.enable()
        self._report(op, xs, converged=converged)

    def _report(self, op: str, xs: list[float], converged: bool | None = None) -> None:
        s = sorted(xs)
        p20 = s[max(1, len(s) // 5)]
        print_speed(f"{op} [min]", s[0])
        emit(op, "min", s[0])
        emit(op, "median", statistics.median(s))
        emit(op, "p20", p20)
        if converged is not None:
            emit(op, "converged", 1.0 if converged else 0.0)


# --------------------------------------------------------------------------------------
# Dynamic type zoo.
# --------------------------------------------------------------------------------------

_counter = itertools.count()


def _key(base: str) -> str:
    return f"bench.tying.{base}_{next(_counter)}"


def _make_class(name: str, annotations: dict[str, Any]) -> Any:
    """Create and register a py_class dataclass with the given typed fields."""
    cls = type(name, (dc.Object,), {"__annotations__": dict(annotations), "__module__": __name__})
    return dc.py_class(_key(name))(cls)


WIDTH_CLS: dict[int, Any] = {n: _make_class(f"W{n}", {f"f{i}": int for i in range(n)}) for n in WIDTHS}


def _mk_width(n: int) -> Any:
    return WIDTH_CLS[n](*([0] * n))


# Depth chain: LEAF (1 field) <- D1.x <- D2.x <- ... <- D{max}.x
_MAXD = max(DEPTHS)
LEAF = _make_class("Leaf", {"f0": int})
_CHAIN: list[Any] = [LEAF]
for _k in range(1, _MAXD + 1):
    _CHAIN.append(_make_class(f"D{_k}", {"x": _CHAIN[_k - 1]}))


def _build_root(depth: int) -> Any:
    obj = LEAF(0)
    for k in range(1, depth + 1):
        obj = _CHAIN[k](obj)
    return obj


def _gen_access(depth: int) -> Callable[[Any, int], None]:
    """Compile ``for _ in range(n): root.x.x...`` (depth dots) -- no per-level Python loop."""
    ns: dict[str, Any] = {}
    exec(f"def _f(root, n):\n    for _ in range(n):\n        root{'.x' * depth}\n", ns)  # noqa: S102
    return ns["_f"]


# --------------------------------------------------------------------------------------
# Registration of all ops (no timing happens here; the driver samples them round-robin).
# --------------------------------------------------------------------------------------


def register_attr(b: Bench) -> None:
    """attr_hit (anchored Active-hit) and attr_churn (root-only, revive) per depth."""
    for d in DEPTHS:
        access = _gen_access(d)

        # HIT: hold every wrapper along the chain so each access is an Active-hit.
        root_hit = _build_root(d)
        anchors = []
        cur = root_hit
        for _ in range(d):
            cur = cur.x
            anchors.append(cur)
        b.keep(root_hit, anchors)  # MUST stay alive across the whole round-robin run
        b.add_py(f"attr_hit.depth{d}", lambda n, _a=root_hit, _f=access: _f(_a, n))

        # CHURN: hold only the root; accessed wrappers die by refcount each iteration.
        root_churn = _build_root(d)
        b.keep(root_churn)
        b.add_py(f"attr_churn.depth{d}", lambda n, _a=root_churn, _f=access: _f(_a, n))


def register_create(b: Bench) -> None:
    for n in WIDTHS:
        cls = WIDTH_CLS[n]
        args = tuple([0] * n)

        def mk(it: int, _c: Any = cls, _ar: tuple = args) -> None:
            for _ in range(it):
                _c(*_ar)

        b.add_py(f"create.width{n}", mk)

    for d in (3, 5):

        def mk_nested(it: int, _d: int = d) -> None:
            for _ in range(it):
                _build_root(_d)

        b.add_py(f"create.depth{d}_nested", mk_nested)


def _invoke_cpp_src() -> str:
    parts = ["#include <tvm/ffi/function.h>", ""]
    for k in ARITIES:
        params = ", ".join(f"tvm::ffi::AnyView a{i}" for i in range(k))
        callargs = ", ".join(f"a{i}" for i in range(k))
        sig = f"void invoke_n_{k}(tvm::ffi::Function f, int64_t n{', ' + params if params else ''})"
        parts += [sig + " {", "    for (int64_t i = 0; i < n; ++i) {", f"        f({callargs});", "    }", "}", ""]
    return "\n".join(parts)


def register_callbacks(b: Bench) -> None:
    """C++ -> Python callback path: a C++ loop (``invoke_n_K``) invokes a Python callable
    ``CALL_N`` times, so the per-iteration cost is materializing each ``AnyView`` arg into a
    Python wrapper to enter the callee (the tying path), NOT C++->C++ dispatch.

    The C++ loop is the harness, not the subject: it keeps the Python interpreter (loop
    bytecode, args-tuple build, outbound Python->AnyView conversion) OUT of the per-call
    measurement, leaving only the inbound crossing + wrapper materialization. Args are
    converted to AnyView once on entry and reused; the single ``arg`` is anchored, so each
    arg is an Active-hit (tying increfs the canonical wrapper; off allocates+frees a fresh one).

      * callback.discard.argsK -- callee ``lambda *a: None`` drops its args; isolates the
                                  inbound per-arg wrapper cost, scaled by arity.
      * callback.identity.args1 -- callee ``lambda y: y`` returns its arg; also exercises the
                                  RETURN-path wrapper (identity-return reuses the cached wrapper).
    """
    mod = tvm_ffi.cpp.load_inline(
        name="bench_tying_invoke_n",
        cpp_sources=_invoke_cpp_src(),
        functions=[f"invoke_n_{k}" for k in ARITIES],
    )
    invokers = {k: mod.get_function(f"invoke_n_{k}") for k in ARITIES}
    arg = _mk_width(1)
    b.keep(arg, mod, invokers)

    for k in ARITIES:
        sink = tvm_ffi.convert(lambda *a: None)
        b.keep(sink)
        invoke = invokers[k]
        a = [arg] * k

        def sample(_inv: Any = invoke, _f: Any = sink, _a: list = a) -> float:
            t0 = time.perf_counter_ns()
            _inv(_f, CALL_N, *_a)
            return (time.perf_counter_ns() - t0) / CALL_N / 1e9

        b.add(f"callback.discard.args{k}", sample)

    identity = tvm_ffi.convert(lambda y: y)
    b.keep(identity)
    inv1 = invokers[1]

    def sample_id(_inv: Any = inv1, _f: Any = identity, _x: Any = arg) -> float:
        t0 = time.perf_counter_ns()
        _inv(_f, CALL_N, _x)
        return (time.perf_counter_ns() - t0) / CALL_N / 1e9

    b.add("callback.identity.args1", sample_id)


def register_array(b: Bench) -> None:
    K = 64
    objs = [_mk_width(1) for _ in range(K)]
    arr_obj = tvm_ffi.convert(objs)
    arr_int = tvm_ffi.convert(list(range(K)))
    held = [arr_obj[i] for i in range(K)]  # anchor element wrappers -> Active-hit
    b.keep(objs, arr_obj, arr_int, held)

    def obj_hit(n: int) -> None:
        for _ in range(n):
            arr_obj[0]

    def obj_churn(n: int) -> None:
        for _ in range(n):
            w = arr_obj[0]
            del w

    def int_floor(n: int) -> None:
        for _ in range(n):
            arr_int[0]

    b.add_py("array.obj_hit", obj_hit)
    b.add_py("array.obj_churn", obj_churn)
    b.add_py("array.int_floor", int_floor)


_MAKE_OBJECT_CPP = r"""
#include <tvm/ffi/memory.h>
#include <tvm/ffi/object.h>
#include <chrono>

namespace bench {
using namespace tvm::ffi;
class SmallObj : public Object {
 public:
  int64_t v{0};
  SmallObj() = default;
  explicit SmallObj(UnsafeInit) {}
  TVM_FFI_DECLARE_OBJECT_INFO("bench.SmallObj", SmallObj, Object);
};
class WideObj : public Object {
 public:
  int64_t v[32] = {0};
  WideObj() = default;
  explicit WideObj(UnsafeInit) {}
  TVM_FFI_DECLARE_OBJECT_INFO("bench.WideObj", WideObj, Object);
};
}  // namespace bench

double churn_make_object_small(int64_t n) {
    auto t0 = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < n; ++i) { auto o = tvm::ffi::make_object<bench::SmallObj>(); (void)o; }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count() / static_cast<double>(n);
}
double churn_make_object_wide(int64_t n) {
    auto t0 = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < n; ++i) { auto o = tvm::ffi::make_object<bench::WideObj>(); (void)o; }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count() / static_cast<double>(n);
}
"""


def register_cpp(b: Bench) -> None:
    try:
        mod = tvm_ffi.cpp.load_inline(
            name="bench_tying_make_object",
            cpp_sources=_MAKE_OBJECT_CPP,
            functions=["churn_make_object_small", "churn_make_object_wide"],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"# cpp.make_object microbench SKIPPED: {exc}")
        return
    b.keep(mod)
    b.add_cpp("cpp.make_object.small", mod.get_function("churn_make_object_small"))
    b.add_cpp("cpp.make_object.wide", mod.get_function("churn_make_object_wide"))


# --------------------------------------------------------------------------------------
# Memory footprint -- a single RSS measurement (not a min-of-K timing; run separately).
# --------------------------------------------------------------------------------------


def _rss_bytes() -> int:
    with open("/proc/self/statm") as f:  # noqa: PTH123
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def bench_memory() -> None:
    M = 1_000_000
    for n in (WIDTHS[0], WIDTHS[len(WIDTHS) // 2], WIDTHS[-1]):  # narrow / mid / wide
        gc.collect()
        hold: list[Any] = []
        rss0 = _rss_bytes()
        for _ in range(M):
            hold.append(_mk_width(n))
        rss1 = _rss_bytes()
        per_obj = (rss1 - rss0) / M
        print(f"mem.width{n:<34} {per_obj:8.1f} bytes/obj (RSS delta, incl ~8B list slot)")
        emit(f"mem.width{n}", "bytes_per_obj", per_obj)
        del hold
        gc.collect()


# --------------------------------------------------------------------------------------
# Header + tying probe (verification gate) + main.
# --------------------------------------------------------------------------------------


def header_and_probe() -> bool:
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    disable_env = os.environ.get("TVM_FFI_DISABLE_TYING", "<unset>")
    parent = _build_root(1)
    tying_on = parent.x is parent.x
    del parent

    print("=" * 78)
    print(f"CONFIG\tpython\t{sys.version.split()[0]}")
    print(f"CONFIG\tgil_enabled\t{gil}")
    print(f"CONFIG\tcore_so\t{os.path.basename(tvm_ffi.core.__file__)}")
    print(f"CONFIG\ttvm_ffi_version\t{getattr(tvm_ffi, '__version__', '?')}")
    print(f"CONFIG\tdisable_tying_env\t{disable_env}")
    print(f"CONFIG\ttying_on\t{tying_on}")
    print("=" * 78)

    if disable_env == "1":
        assert not tying_on, "TVM_FFI_DISABLE_TYING=1 but tying probe is ON -- toggle broken"
    return tying_on


def main() -> None:
    tying_on = header_and_probe()
    print(f"# tying_on={tying_on} repeat={REPEAT} trials={TRIALS} call_n={CALL_N} "
          f"cpp_n={CPP_N} (round-robin order)")
    print(f"# depths={DEPTHS} widths={WIDTHS} arities={ARITIES}")
    print("-" * 78)

    b = Bench()
    register_attr(b)
    register_create(b)
    register_callbacks(b)
    register_array(b)
    register_cpp(b)
    b.run(TRIALS)

    bench_memory()
    print("-" * 78)
    print("# done")


if __name__ == "__main__":
    main()
