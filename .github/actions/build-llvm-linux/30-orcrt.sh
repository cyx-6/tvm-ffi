#!/usr/bin/env bash
# Build only liborc_rt from compiler-rt and place it where the addon's GLOB
# (lib/clang/<ver>/lib/liborc_rt*.a) finds it. liborc_rt drives the JIT's
# ELFNixPlatform and is embedded into the addon .so at build time.
set -euo pipefail
. "$(dirname "$0")/00-env.sh"

# GCC host: compiler-rt can't infer the triple, so pass LLVM's own default.
TRIPLE="$("$LLVM_PREFIX/bin/llvm-config" --host-target)"

cmake -G Ninja "$LLVM_SRC_DIR/compiler-rt" -B "$BUILD/orcrt" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$LLVM_PREFIX" \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_C_COMPILER_TARGET="$TRIPLE" -DCMAKE_CXX_COMPILER_TARGET="$TRIPLE" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DLLVM_CONFIG_PATH="$LLVM_PREFIX/bin/llvm-config" \
  -DLLVM_CMAKE_DIR="$LLVM_PREFIX/lib/cmake/llvm" \
  -DCOMPILER_RT_STANDALONE_BUILD=ON -DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
  -DCOMPILER_RT_INCLUDE_TESTS=OFF \
  -DCOMPILER_RT_BUILD_ORC=ON -DCOMPILER_RT_BUILD_BUILTINS=OFF \
  -DCOMPILER_RT_BUILD_SANITIZERS=OFF -DCOMPILER_RT_BUILD_XRAY=OFF \
  -DCOMPILER_RT_BUILD_LIBFUZZER=OFF -DCOMPILER_RT_BUILD_PROFILE=OFF \
  -DCOMPILER_RT_BUILD_MEMPROF=OFF -DCOMPILER_RT_BUILD_CTX_PROFILE=OFF

cmake --build "$BUILD/orcrt" --target install

# A standalone GCC build installs to $prefix/lib/linux; mirror into the clang
# resource-dir layout the addon GLOBs.
built="$(find "$LLVM_PREFIX/lib" "$BUILD/orcrt" -name 'liborc_rt*.a' 2>/dev/null | head -1)"
[ -n "$built" ] || { echo "[orcrt] ERROR: liborc_rt not built"; exit 1; }
dest="$LLVM_PREFIX/lib/clang/${LLVM_VERSION%%.*}/lib/linux"
mkdir -p "$dest" && cp -a "$built" "$dest/"
echo "[orcrt] placed $dest/$(basename "$built")"
echo "[orcrt] done"
