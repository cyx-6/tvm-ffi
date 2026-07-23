#!/usr/bin/env bash
# Shared env for the from-source LLVM build. Sourced by 10/20/30.
#
# Build everything with the manylinux gcc-toolset so every artifact stays under
# the manylinux GLIBCXX/CXXABI floor -- the reason we build LLVM ourselves
# instead of using conda-forge's newer-GCC binaries.
set -euo pipefail

export LLVM_PREFIX="${LLVM_PREFIX:-/opt/llvm}"
export LLVM_VERSION="${LLVM_VERSION:-22.1.0}"
export BUILD="${BUILD:-/tmp/llvm-build}"          # local disk; never a network FS

# Source trees (provided by actions/checkout; required).
: "${LLVM_SRC_DIR:?set to the checked-out llvm-project dir}"
: "${ZLIB_SRC_DIR:?set to the checked-out zlib dir}"
: "${ZSTD_SRC_DIR:?set to the checked-out zstd dir}"

# Newest gcc-toolset present in the image (manylinux_2_28 ships gcc-toolset-14).
_gcc_enable="$(ls -d /opt/rh/gcc-toolset-*/enable 2>/dev/null | sort -V | tail -1 || true)"
[ -n "$_gcc_enable" ] && source "$_gcc_enable"
export CC=gcc CXX=g++
export NPROC="${NPROC:-$(nproc)}"

echo "[env] prefix=$LLVM_PREFIX llvm=$LLVM_VERSION nproc=$NPROC gcc=$(gcc -dumpversion)"
