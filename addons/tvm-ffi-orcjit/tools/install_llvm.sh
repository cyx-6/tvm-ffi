#!/bin/bash
# Install LLVM from conda-forge using micromamba.
# Usage: bash tools/install_llvm.sh [version]
#   version defaults to LLVM_VERSION env var, then 22.1.0
set -ex

LLVM_VERSION="${LLVM_VERSION:-${1:-22.1.0}}"
PREFIX="${LLVM_PREFIX:-/opt/llvm}"

# Detect micromamba platform
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)   PLATFORM="linux-64" ;;
  Linux-aarch64)  PLATFORM="linux-aarch64" ;;
  Darwin-x86_64)  PLATFORM="osx-64" ;;
  Darwin-arm64)   PLATFORM="osx-arm64" ;;
  *)              echo "Unsupported: $(uname -s)-$(uname -m)"; exit 1 ;;
esac

# Install micromamba (retry up to 3 times for transient CDN failures)
for i in 1 2 3; do
  curl -Ls "https://micro.mamba.pm/api/micromamba/${PLATFORM}/latest" \
    | tar -xvj -C /usr/local bin/micromamba && break
  echo "micromamba download attempt $i failed, retrying..."
  sleep 5
done

# Install LLVM, clang (for compiling test objects), compiler-rt (for liborc_rt),
# and zlib (static PIC lib from conda-forge).
/usr/local/bin/micromamba create -p "${PREFIX}" -c conda-forge \
  "llvmdev=${LLVM_VERSION}" "clangdev=${LLVM_VERSION}" "compiler-rt=${LLVM_VERSION}" \
  zlib \
  -y

# Build static PIC libzstd.a from source.
# conda-forge's zstd package only ships the shared library, but we need static
# linking so the wheel is self-contained (no runtime zstd dependency).
ZSTD_VERSION="1.5.7"
curl -sL "https://github.com/facebook/zstd/releases/download/v${ZSTD_VERSION}/zstd-${ZSTD_VERSION}.tar.gz" \
  | tar -xz
cmake -S "zstd-${ZSTD_VERSION}/build/cmake" -B _zstd_build \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DZSTD_BUILD_SHARED=OFF -DZSTD_BUILD_STATIC=ON \
  -DZSTD_BUILD_PROGRAMS=OFF
cmake --build _zstd_build --target install -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
rm -rf "zstd-${ZSTD_VERSION}" _zstd_build
