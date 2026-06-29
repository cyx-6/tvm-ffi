#!/usr/bin/env bash
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
#
# One-command, portable before/after benchmark for PR #593 (PyObject-tying).
#
# It builds the BEFORE (base commit) and AFTER (head commit) trees in throwaway
# git worktrees, creates the uv venvs, runs the 3x3 config matrix, and prints the
# comparison table -- WITHOUT touching your live checkout. On a fresh clone:
#
#   git checkout pyobject && bench/run_bench.sh
#
# is the entire reproduction (uv auto-downloads the interpreters; a C++17
# toolchain must be present). See bench/README.md for the methodology.
#
# Usage:
#   bench/run_bench.sh [all|bootstrap|run|table|clean] [--pythons "3.13 3.14t"] [--pin CPU]
#
# Subcommands (default: all):
#   all        bootstrap + run
#   bootstrap  create worktrees + venvs, build every (commit x python) config
#   run        run the 9-config matrix, print + save the table
#   table      regenerate the table from existing results/ logs (no rebuild/rerun)
#   clean      remove the worktrees, venvs, caches, and results
#
# Env overrides:
#   BENCH_WORK   scratch dir (default: <repo>/venv/bench-work, gitignored)
#   BASE_REF     "before" commit  (default: 88e066e, the PR #593 base)
#   HEAD_REF     "after"  commit  (default: current HEAD)
#   PYTHONS      space-separated interpreters (default: "3.12 3.13 3.14t")
#   FORCE_BUILD  set to 1 to rebuild even if the venv already imports tvm_ffi
#
# Linux only (the memory probe reads /proc/self/statm).

set -uo pipefail  # NOT -e: this is a matrix orchestrator -- attempt every config and
                  # report a summary rather than dying on the first build/run hiccup.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PATCH="$SCRIPT_DIR/toggle_tying.patch"

WORK="${BENCH_WORK:-$REPO/venv/bench-work}"
BASE_REF="${BASE_REF:-88e066e}"
HEAD_REF="${HEAD_REF:-$(git -C "$REPO" rev-parse HEAD)}"
PYTHONS="${PYTHONS:-3.12 3.13 3.14t}"
FORCE_BUILD="${FORCE_BUILD:-0}"
RES="$WORK/results"
PIN=()

log() { echo "[bench] $*"; }
pytag() { echo "${1//./}"; }  # 3.14t -> 314t, 3.13 -> 313

usage() { sed -n '/^# One-command/,/^# Linux only/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# ---------------------------------------------------------------------------- bootstrap

add_worktree() {  # $1=path  $2=ref
  if [[ -d "$1" ]]; then
    log "worktree exists: $1 (skip)"
  else
    log "git worktree add --detach $1 $2"
    git -C "$REPO" worktree add --detach "$1" "$2" || { log "ERROR: worktree add failed"; return 1; }
  fi
}

apply_toggle() {  # $1=worktree -- apply the tying toggle idempotently (head tree only)
  if git -C "$1" apply --reverse --check "$PATCH" 2>/dev/null; then
    log "toggle already applied in $(basename "$1") (skip)"
  elif git -C "$1" apply --check "$PATCH" 2>/dev/null; then
    git -C "$1" apply "$PATCH" && log "toggle applied in $(basename "$1")"
  else
    log "ERROR: toggle_tying.patch does not apply to $1 (HEAD_REF base.pxi drifted?)"
    return 1
  fi
}

make_venv() {  # $1=path  $2=py
  if [[ -d "$1" ]]; then
    log "venv exists: $1 (skip)"
  else
    log "uv venv --python $2 $1"
    uv venv --python "$2" "$1" >/dev/null || { log "ERROR: uv venv failed for $2"; return 1; }
  fi
}

build() {  # $1=venv_dir  $2=src_tree  $3=build_dir_name  $4=label
  local py="$1/bin/python"
  if [[ "$FORCE_BUILD" != "1" ]] && "$py" -c 'import tvm_ffi' >/dev/null 2>&1; then
    local so; so=$("$py" -c 'import tvm_ffi.core as c,os;print(os.path.basename(c.__file__))' 2>/dev/null)
    log "BUILD $4 already importable -> ${so:-?} (skip; FORCE_BUILD=1 to rebuild)"
    return 0
  fi
  log "BUILD $4 (src=$(basename "$2") build-dir=$3)"
  if ! uv pip install --python "$py" --reinstall \
       --config-settings=build-dir="$3" -e "$2" > "$RES/build-$4.log" 2>&1; then
    log "  build FAILED -> $RES/build-$4.log"
    tail -5 "$RES/build-$4.log" | sed 's/^/    /'
    return 1
  fi
  uv pip install --python "$py" ninja >/dev/null 2>&1 || true  # load_inline needs ninja at runtime
  local so; so=$("$py" -c 'import tvm_ffi.core as c,os;print(os.path.basename(c.__file__))' 2>/dev/null || echo IMPORT-FAILED)
  log "  build $4 OK -> $so"
}

bootstrap() {
  mkdir -p "$RES"
  add_worktree "$WORK/wt-main" "$BASE_REF" || return 1
  add_worktree "$WORK/wt-head" "$HEAD_REF" || return 1
  # Worktrees do not inherit submodule checkouts; dlpack + libbacktrace are build inputs.
  log "init submodules in worktrees"
  git -C "$WORK/wt-main" submodule update --init --recursive >/dev/null 2>&1 || log "  WARN: wt-main submodules"
  git -C "$WORK/wt-head" submodule update --init --recursive >/dev/null 2>&1 || log "  WARN: wt-head submodules"
  apply_toggle "$WORK/wt-head" || return 1  # head carries the TVM_FFI_DISABLE_TYING lever
  for py in $PYTHONS; do
    local tag; tag="$(pytag "$py")"
    make_venv "$WORK/venv-main-$py" "$py"
    make_venv "$WORK/venv-head-$py" "$py"
    build "$WORK/venv-main-$py" "$WORK/wt-main" "build-m$tag" "main-$py"
    build "$WORK/venv-head-$py" "$WORK/wt-head" "build-h$tag" "head-$py"
  done
}

# ---------------------------------------------------------------------------- run

run_one() {  # $1=label  $2=python  $3=disable_tying(0/1)
  local label="$1" py="$2" disable="$3"
  if [[ ! -x "$py" ]]; then
    log ">>> $label SKIP (no interpreter at $py -- run bootstrap first)"
    return
  fi
  local cache="$WORK/cache-$label"; mkdir -p "$cache"
  # Prepend the venv bin so load_inline finds this venv's ninja (not otherwise on PATH).
  local envv=(env "TVM_FFI_CACHE_DIR=$cache" "PATH=$(dirname "$py"):$PATH")
  [[ "$disable" == "1" ]] && envv+=("TVM_FFI_DISABLE_TYING=1")
  log ">>> $label"
  "${PIN[@]}" "${envv[@]}" "$py" "$SCRIPT_DIR/bench_pyobject_tying.py" > "$RES/$label.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    log "    FAILED (exit=$rc) -- see $RES/$label.log"
    tail -5 "$RES/$label.log" | sed 's/^/      /'
    return
  fi
  # Ground-truth gate: the runtime probe must match the config (headon ties, others don't).
  local want="False"; [[ "$label" == headon-* ]] && want="True"
  local got; got=$(awk -F'\t' '$1=="CONFIG" && $2=="tying_on"{print $3}' "$RES/$label.log")
  if [[ "$got" != "$want" ]]; then
    log "    WARNING: tying_on=$got but expected $want for $label -- config mismatch!"
  else
    log "    tying_on=$got (ok)"
  fi
}

# Pick any built interpreter to run the stdlib-only table generator.
table_python() {
  local py
  for py in $PYTHONS; do
    [[ -x "$WORK/venv-head-$py/bin/python" ]] && { echo "$WORK/venv-head-$py/bin/python"; return; }
    [[ -x "$WORK/venv-main-$py/bin/python" ]] && { echo "$WORK/venv-main-$py/bin/python"; return; }
  done
  command -v python3
}

table() {
  local tp; tp="$(table_python)"
  [[ -z "$tp" ]] && { log "no python available to render the table"; return 1; }
  "$tp" "$SCRIPT_DIR/make_table.py" "$RES" | tee "$RES/SUMMARY_TABLE.txt"
}

run() {
  mkdir -p "$RES"
  for py in $PYTHONS; do
    run_one "main-$py"    "$WORK/venv-main-$py/bin/python" 0
    run_one "headoff-$py" "$WORK/venv-head-$py/bin/python" 1
    run_one "headon-$py"  "$WORK/venv-head-$py/bin/python" 0
  done
  echo
  echo "===== COMPARISON TABLE ====="
  table
}

# ---------------------------------------------------------------------------- clean

clean() {
  local wt
  for wt in wt-main wt-head; do
    if [[ -d "$WORK/$wt" ]]; then
      log "git worktree remove --force $WORK/$wt"
      git -C "$REPO" worktree remove --force "$WORK/$wt" 2>/dev/null || rm -rf "$WORK/$wt"
    fi
  done
  git -C "$REPO" worktree prune 2>/dev/null || true
  log "rm -rf venvs / caches / results under $WORK"
  rm -rf "$WORK"/venv-* "$WORK"/cache-* "$RES"
  rmdir "$WORK" 2>/dev/null || true
  log "clean done"
}

# ---------------------------------------------------------------------------- main

CMD="all"
while [[ $# -gt 0 ]]; do
  case "$1" in
    all|bootstrap|run|table|clean) CMD="$1"; shift;;
    --pythons) PYTHONS="$2"; shift 2;;
    --pin) PIN=(taskset -c "$2"); log "pinning runs to CPU $2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage; exit 1;;
  esac
done

log "repo=$REPO"
log "work=$WORK"
log "base=$BASE_REF head=$HEAD_REF pythons=[$PYTHONS] cmd=$CMD"

case "$CMD" in
  all)       bootstrap && run;;
  bootstrap) bootstrap;;
  run)       run;;
  table)     table;;
  clean)     clean;;
esac
