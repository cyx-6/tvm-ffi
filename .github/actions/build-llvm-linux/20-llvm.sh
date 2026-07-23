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

# Minimal LLVM into $LLVM_PREFIX: only what the orcjit addon links
# (Core/OrcJIT/Support + native target). No clang, no non-native targets.
set -euo pipefail
# shellcheck disable=SC1091  # sourced at runtime from the action dir
. "$(dirname "$0")/00-env.sh"

# Native target only: X86 on x86_64 runners, AArch64 on arm runners.
case "$(uname -m)" in
  x86_64)  LLVM_TARGET=X86 ;;
  aarch64) LLVM_TARGET=AArch64 ;;
  *) echo "[llvm] unsupported arch $(uname -m)"; exit 1 ;;
esac

# RTTI ON is REQUIRED: the addon subclasses polymorphic LLVM classes
# (jitlink::JITLinkMemoryManager); without it the .so fails to load on a missing
# _ZTIN4llvm... typeinfo symbol. Static + PIC so the libs fold into the addon .so.
cmake -G Ninja "$LLVM_SRC_DIR/llvm" -B "$BUILD/llvm" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$LLVM_PREFIX" \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGET" \
  -DBUILD_SHARED_LIBS=OFF -DLLVM_ENABLE_PIC=ON -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_ZLIB=FORCE_ON -DZLIB_LIBRARY="$LLVM_PREFIX/lib/libz.a" \
  -DZLIB_INCLUDE_DIR="$LLVM_PREFIX/include" \
  -DLLVM_ENABLE_ZSTD=FORCE_ON -Dzstd_LIBRARY="$LLVM_PREFIX/lib/libzstd.a" \
  -Dzstd_INCLUDE_DIR="$LLVM_PREFIX/include" \
  -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_LIBEDIT=OFF \
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_DOCS=OFF \
  -DLLVM_ENABLE_ASSERTIONS=OFF

cmake --build "$BUILD/llvm" --target install

"$LLVM_PREFIX/bin/llvm-config" --version
echo "[llvm] targets: $("$LLVM_PREFIX/bin/llvm-config" --targets-built)"
echo "[llvm] done"
