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

# Static, -fPIC zlib + zstd into $LLVM_PREFIX. The addon links libz.a / libzstd.a
# statically from the prefix, so the wheel carries no external zlib/zstd dep.
# Both built with cmake+ninja (already required by LLVM) -- no make/configure.
set -euo pipefail
# shellcheck disable=SC1091  # sourced at runtime from the action dir
. "$(dirname "$0")/00-env.sh"

build_static () {   # <name> <src-dir> <extra cmake args...>
  local name="$1" src="$2"; shift 2
  echo "[deps] $name"
  cmake -G Ninja "$src" -B "$BUILD/$name" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$LLVM_PREFIX" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_C_COMPILER="$CC" "$@"
  cmake --build "$BUILD/$name" --target install
}

build_static zlib "$ZLIB_SRC_DIR" -DZLIB_BUILD_EXAMPLES=OFF

build_static zstd "$ZSTD_SRC_DIR/build/cmake" \
  -DZSTD_BUILD_STATIC=ON -DZSTD_BUILD_SHARED=OFF \
  -DZSTD_BUILD_PROGRAMS=OFF -DZSTD_BUILD_TESTS=OFF -DZSTD_LEGACY_SUPPORT=OFF

# zstd may install to lib64; the addon looks in lib. Normalize.
[ -f "$LLVM_PREFIX/lib64/libzstd.a" ] && cp -a "$LLVM_PREFIX/lib64/libzstd.a" "$LLVM_PREFIX/lib/libzstd.a"

ls -l "$LLVM_PREFIX/lib/libz.a" "$LLVM_PREFIX/lib/libzstd.a"
echo "[deps] done"
