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
_gcc_enable="$(find /opt/rh -maxdepth 2 -path '*/gcc-toolset-*/enable' 2>/dev/null | sort -V | tail -1)"
# shellcheck source=/dev/null
[ -n "$_gcc_enable" ] && source "$_gcc_enable"
export CC=gcc CXX=g++
export NPROC="${NPROC:-$(nproc)}"

echo "[env] prefix=$LLVM_PREFIX llvm=$LLVM_VERSION nproc=$NPROC gcc=$(gcc -dumpversion)"
