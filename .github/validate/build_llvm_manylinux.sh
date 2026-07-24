#!/usr/bin/env bash
# Build a minimal static LLVM (+ zlib/zstd + liborc_rt) from source inside a
# manylinux image and pack the install prefix into a compressed tarball.
#
# The output feeds apache/tvm-ffi's orcjit addon, whose wheel must stay under the
# manylinux GLIBCXX/CXXABI floor. conda-forge's LLVM is built with a newer GCC
# whose libstdc++ carries GLIBCXX_3.4.29+ symbols that break that floor, forcing a
# -static-libstdc++ workaround. Building here with the manylinux gcc-toolset keeps
# every artifact under GLIBCXX_3.4.22 / CXXABI_1.3.11, so auditwheel tags the addon
# manylinux_2_27/2_28 even with libstdc++ linked dynamically.
#
# Run inside quay.io/pypa/manylinux_2_28_<arch> (via `docker run`), e.g.:
#   bash scripts/build_llvm_manylinux.sh \
#     --arch x86_64 --llvm-version 22.1.0 \
#     --llvm-src /work/llvm-project --zlib-src /work/zlib --zstd-src /work/zstd \
#     --out /out/llvm-22.1.0-linux-x86_64.tar.zst
#
# The build flags below (static + PIC + RTTI, native target only, zlib/zstd
# FORCE_ON from the prefix) are the ABI contract the consumer's CMake and
# auditwheel depend on -- do not relax them.
set -euo pipefail

# ---- args ----
ARCH=""
LLVM_VERSION="22.1.0"
LLVM_SRC_DIR=""
ZLIB_SRC_DIR=""
ZSTD_SRC_DIR=""
PREFIX="/opt/llvm"
BUILD="/tmp/llvm-build"
OUT=""

usage() {
  cat >&2 <<'EOF'
usage: build_llvm_manylinux.sh --arch <x86_64|aarch64> [options]
  --arch <x86_64|aarch64>   target arch (also selects the LLVM native target)   [required]
  --llvm-version <v>        LLVM release, e.g. 22.1.0                            [default 22.1.0]
  --llvm-src <dir>          checked-out llvm-project (tag llvmorg-<v>)           [required]
  --zlib-src <dir>          checked-out madler/zlib (tag v<zlib>)               [required]
  --zstd-src <dir>          checked-out facebook/zstd (tag v<zstd>)             [required]
  --prefix <dir>            install prefix                                       [default /opt/llvm]
  --build <dir>            scratch build dir (local disk, never a network FS)    [default /tmp/llvm-build]
  --out <tarball>           write llvm-<v>-linux-<arch>.tar.zst here             [required]
EOF
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --arch)         ARCH="$2"; shift 2 ;;
    --llvm-version) LLVM_VERSION="$2"; shift 2 ;;
    --llvm-src)     LLVM_SRC_DIR="$2"; shift 2 ;;
    --zlib-src)     ZLIB_SRC_DIR="$2"; shift 2 ;;
    --zstd-src)     ZSTD_SRC_DIR="$2"; shift 2 ;;
    --prefix)       PREFIX="$2"; shift 2 ;;
    --build)        BUILD="$2"; shift 2 ;;
    --out)          OUT="$2"; shift 2 ;;
    -h|--help)      usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
done

[ -n "$ARCH" ] || { echo "error: --arch is required" >&2; usage; }
[ -n "$OUT" ]  || { echo "error: --out is required" >&2; usage; }
: "${LLVM_SRC_DIR:?set --llvm-src to the checked-out llvm-project dir}"
: "${ZLIB_SRC_DIR:?set --zlib-src to the checked-out zlib dir}"
: "${ZSTD_SRC_DIR:?set --zstd-src to the checked-out zstd dir}"

case "$ARCH" in
  x86_64)  LLVM_TARGET=X86 ;;
  aarch64) LLVM_TARGET=AArch64 ;;
  *) echo "error: unsupported --arch $ARCH" >&2; exit 1 ;;
esac

# ---- toolchain: newest gcc-toolset in the image (manylinux_2_28 -> 14) ----
_gcc_enable="$(find /opt/rh -maxdepth 2 -path '*/gcc-toolset-*/enable' 2>/dev/null | sort -V | tail -1)"
# shellcheck source=/dev/null
[ -n "$_gcc_enable" ] && source "$_gcc_enable"
export CC=gcc CXX=g++
NPROC="${NPROC:-$(nproc)}"
command -v ninja >/dev/null || pipx install ninja || pip install ninja
export PATH="$HOME/.local/bin:$PATH"

echo "[env] arch=$ARCH prefix=$PREFIX llvm=$LLVM_VERSION nproc=$NPROC gcc=$(gcc -dumpversion) target=$LLVM_TARGET"
mkdir -p "$PREFIX" "$BUILD"

# ---- deps: static, -fPIC zlib + zstd into the prefix ----
# The addon links libz.a / libzstd.a statically from the prefix, so the wheel
# carries no external zlib/zstd dep. cmake+ninja only -- no make/configure.
build_static () {   # <name> <src-dir> <extra cmake args...>
  local name="$1" src="$2"; shift 2
  echo "[deps] $name"
  cmake -G Ninja "$src" -B "$BUILD/$name" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_C_COMPILER="$CC" "$@"
  cmake --build "$BUILD/$name" --target install
}

build_static zlib "$ZLIB_SRC_DIR" -DZLIB_BUILD_EXAMPLES=OFF

build_static zstd "$ZSTD_SRC_DIR/build/cmake" \
  -DZSTD_BUILD_STATIC=ON -DZSTD_BUILD_SHARED=OFF \
  -DZSTD_BUILD_PROGRAMS=OFF -DZSTD_BUILD_TESTS=OFF -DZSTD_LEGACY_SUPPORT=OFF

# zstd may install to lib64; the addon looks in lib. Normalize.
[ -f "$PREFIX/lib64/libzstd.a" ] && cp -a "$PREFIX/lib64/libzstd.a" "$PREFIX/lib/libzstd.a"
ls -l "$PREFIX/lib/libz.a" "$PREFIX/lib/libzstd.a"

# ---- LLVM: only what the orcjit addon links (Core/OrcJIT/Support + native) ----
# RTTI ON is REQUIRED: the addon subclasses polymorphic LLVM classes
# (jitlink::JITLinkMemoryManager); without it the .so fails to load on a missing
# _ZTIN4llvm... typeinfo symbol. Static + PIC so the libs fold into the addon .so.
echo "[llvm] configure"
cmake -G Ninja "$LLVM_SRC_DIR/llvm" -B "$BUILD/llvm" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGET" \
  -DBUILD_SHARED_LIBS=OFF -DLLVM_ENABLE_PIC=ON -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_ZLIB=FORCE_ON -DZLIB_LIBRARY="$PREFIX/lib/libz.a" \
  -DZLIB_INCLUDE_DIR="$PREFIX/include" \
  -DLLVM_ENABLE_ZSTD=FORCE_ON -Dzstd_LIBRARY="$PREFIX/lib/libzstd.a" \
  -Dzstd_INCLUDE_DIR="$PREFIX/include" \
  -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_LIBEDIT=OFF \
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_DOCS=OFF \
  -DLLVM_ENABLE_ASSERTIONS=OFF

echo "[llvm] build + install"
cmake --build "$BUILD/llvm" --target install
"$PREFIX/bin/llvm-config" --version
echo "[llvm] targets: $("$PREFIX/bin/llvm-config" --targets-built)"

# ---- orc_rt: only liborc_rt from compiler-rt ----
# liborc_rt drives the JIT's ELFNixPlatform and is embedded into the addon .so at
# build time. Place it where the addon's GLOB (lib/clang/<ver>/lib/liborc_rt*.a)
# finds it. GCC host: compiler-rt can't infer the triple, so pass LLVM's default.
TRIPLE="$("$PREFIX/bin/llvm-config" --host-target)"
echo "[orcrt] configure (triple=$TRIPLE)"
cmake -G Ninja "$LLVM_SRC_DIR/compiler-rt" -B "$BUILD/orcrt" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_C_COMPILER_TARGET="$TRIPLE" -DCMAKE_CXX_COMPILER_TARGET="$TRIPLE" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DLLVM_CONFIG_PATH="$PREFIX/bin/llvm-config" \
  -DLLVM_CMAKE_DIR="$PREFIX/lib/cmake/llvm" \
  -DCOMPILER_RT_STANDALONE_BUILD=ON -DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
  -DCOMPILER_RT_INCLUDE_TESTS=OFF \
  -DCOMPILER_RT_BUILD_ORC=ON -DCOMPILER_RT_BUILD_BUILTINS=OFF \
  -DCOMPILER_RT_BUILD_SANITIZERS=OFF -DCOMPILER_RT_BUILD_XRAY=OFF \
  -DCOMPILER_RT_BUILD_LIBFUZZER=OFF -DCOMPILER_RT_BUILD_PROFILE=OFF \
  -DCOMPILER_RT_BUILD_MEMPROF=OFF -DCOMPILER_RT_BUILD_CTX_PROFILE=OFF

echo "[orcrt] build + install"
cmake --build "$BUILD/orcrt" --target install

# A standalone GCC build installs to $prefix/lib/linux; mirror into the clang
# resource-dir layout the addon GLOBs.
built="$(find "$PREFIX/lib" "$BUILD/orcrt" -name 'liborc_rt*.a' 2>/dev/null | head -1)"
[ -n "$built" ] || { echo "[orcrt] ERROR: liborc_rt not built"; exit 1; }
dest="$PREFIX/lib/clang/${LLVM_VERSION%%.*}/lib/linux"
mkdir -p "$dest" && cp -a "$built" "$dest/"
echo "[orcrt] placed $dest/$(basename "$built")"

# ---- pack: archive the prefix CONTENTS at top level ----
# Consumer extracts with `tar -C /opt/llvm`, no --strip-components.
mkdir -p "$(dirname "$OUT")"
echo "[pack] $OUT (zstd -19)"
ZSTD_CLEVEL="${ZSTD_CLEVEL:-19}" tar --zstd -cf "$OUT" -C "$PREFIX" .
ls -l "$OUT"
echo "[done] wrote $OUT"
