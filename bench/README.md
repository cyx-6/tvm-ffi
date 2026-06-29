<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# PR #593 PyObject-tying benchmark

Before/after micro-benchmark for **PR #593** ("Tie Python wrapper lifetime to the
underlying C++ FFI object"). It quantifies the latency and memory overhead the PR adds
to the hottest Python-binding paths — attribute access, object creation, function
return, `Array` indexing — and the C++ allocator itself.

This directory is **self-contained tooling**: a fresh clone reproduces the whole study
with one command. Nothing here is imported by the package or exercised by CI.

## One command

```bash
git checkout pyobject     # the branch that carries PR #593
bench/run_bench.sh        # bootstrap everything + run the matrix + print the table
```

`run_bench.sh` builds both the *before* and *after* trees in throwaway **git worktrees**,
creates the **uv** venvs (uv auto-downloads any missing interpreter), runs the matrix, and
prints the comparison table — **without modifying your live checkout**. Re-running is
incremental: existing worktrees/venvs/builds are reused (`FORCE_BUILD=1` forces a rebuild).

Requirements: `git`, `uv`, and a C++17 toolchain (plus `ninja`, which the script installs
into each venv for `load_inline`). **Linux only** — the memory probe reads
`/proc/self/statm`.

```bash
bench/run_bench.sh --pythons 3.13     # fast single-interpreter smoke
bench/run_bench.sh --pin 2            # taskset -c 2 to cut migration noise
bench/run_bench.sh table              # re-render the table from existing logs
bench/run_bench.sh clean              # remove all scratch (worktrees, venvs, caches)
```

Subcommands: `all` (default) · `bootstrap` · `run` · `table` · `clean`.
Env overrides: `BENCH_WORK`, `BASE_REF`, `HEAD_REF`, `PYTHONS`, `FORCE_BUILD`
(see `run_bench.sh --help`).

## What it measures: the three configs

The PR bundles two changes that a naive before/after would conflate: a **binary-layout /
codegen** change (a +16-byte `PyCustomAllocHeader` per object, extra branches, the
`abi3` → version-specific ABI switch in `CMakeLists.txt`) and the **tying logic** itself.
We separate them with three configs built from **two commits**:

| Config     | Built from        | Runtime env              | Isolates                       |
| ---------- | ----------------- | ------------------------ | ------------------------------ |
| `main`     | `BASE_REF`        | —                        | pre-PR baseline                |
| `head-off` | `HEAD_REF` + patch | `TVM_FFI_DISABLE_TYING=1` | layout/codegen (vs `main`)     |
| `head-on`  | `HEAD_REF` + patch | (unset)                  | tying logic (vs `head-off`)    |

Derived columns in the table:

- **`Δlayout` = head-off − main** — PR machinery present but inert (struct size, branches, ABI).
- **`Δtying`  = head-on − head-off** — the tying logic, with binary layout held constant.
- **`Δtotal`  = head-on − main** — end-to-end, what a user actually pays.
- **Free-threaded spin-lock tax** = `Δtying(3.14t) − Δtying(3.13)` — the per-word atomic
  cost the free-threaded build pays that the GIL build does not.

### Why a patch, and why it's safe

All tying behavior is gated by one fact: whether a handle is *canonical*
(`TVMFFIPyIsCanonical`), which is stamped **only** by the custom allocator registered in
`python/tvm_ffi/cython/base.pxi`. Every hot path already has a tested non-canonical fast
path (the pre-PR behavior). So **not registering that allocator turns tying off on the
same head binary** — giving an honest "after-binary, tying-off" reference that isolates
`Δtying` from `Δlayout`.

`toggle_tying.patch` is exactly that: it guards the single `RegisterDefaultAllocator()`
call behind `TVM_FFI_DISABLE_TYING != "1"`. It is **bench-only, never merged**, and is
applied only inside the throwaway `wt-head` worktree — your checkout is never touched.
(Turning tying off leaks one process-short allocation window; harmless for a short bench,
not a production mode.)

## Files

| File                      | Role                                                              |
| ------------------------- | ----------------------------------------------------------------- |
| `run_bench.sh`            | one-command orchestrator (bootstrap → run → table; `clean`)       |
| `bench_pyobject_tying.py` | the benchmark; emits machine-parseable `RESULT` lines             |
| `make_table.py`           | parses `results/*.log` into the comparison table (stdlib only)    |
| `toggle_tying.patch`      | the bench-only `base.pxi` tying toggle (applied to `wt-head`)     |

Scratch (worktrees, venvs, per-config build dirs, `load_inline` caches, `results/`) is
regenerated under `BENCH_WORK` (default `venv/bench-work/`, which is gitignored). Delete it
anytime with `bench/run_bench.sh clean`.

## Reading the numbers

- `ns/call`, **min-of-K** (the robust estimator for sub-100 ns deltas); `gc` is disabled
  around timed regions because tying transitions are refcount-driven, not GC-driven.
- A trailing `*` marks a cell whose **floor was not certified** — host noise (shared-VM
  memory-bandwidth contention) kept the fast samples from converging. The C++ microbench
  self-certifies adaptively and flags honestly rather than silently trusting a noisy number.
- Expect: `Δtying` is largest for **attribute-hit** and **identity-return** (tying reuses the
  cached wrapper instead of allocating a fresh one), modest for churn (revive vs malloc), and
  near-zero for the C++ allocator. On 3.12/3.13, `main` is an `abi3` build, so `Δlayout` there
  is dominated by the limited-API → version-specific ABI switch, not by tying.

## Methodology caveats

- **Single-threaded** — measures the *uncontended* free-threaded lock floor; it under-reports
  real multi-thread contention (out of scope by choice).
- **Shared host** — absolute ns vary run to run; the within-run `main`/`head-off`/`head-on`
  *deltas* are the trustworthy output, not the absolute floors.
