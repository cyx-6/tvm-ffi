cmake_minimum_required(VERSION 3.20)
project(
  tvm_ffi_orcjit
  VERSION 0.1.1
  LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- LLVM_PREFIX handling ----
# scikit-build-core overrides CMAKE_PREFIX_PATH via its init cache, so we read LLVM_PREFIX from the
# environment instead.
if (DEFINED ENV{LLVM_PREFIX})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{LLVM_PREFIX}")
  if (EXISTS "$ENV{LLVM_PREFIX}/Library")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{LLVM_PREFIX}/Library")
  endif ()
endif ()

# ---- Find packages ----
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR}")

find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED
)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE tvm_ffi_ROOT
)
find_package(tvm_ffi CONFIG REQUIRED)

# ---- Build shared library ----
add_library(
  tvm_ffi_orcjit SHARED
  src/ffi/orcjit_session.cc
  src/ffi/orcjit_dylib.cc
  src/ffi/orcjit_memory_manager.cc
  src/ffi/orcjit_slab.cc
  src/ffi/llvm_patches/gotpcrelx_fix.cc
  src/ffi/llvm_patches/init_fini_plugin.cc
  src/ffi/llvm_patches/macho_cxa_atexit_shim.cc
  src/ffi/llvm_patches/win_coff_pdata_strip.cc
  src/ffi/llvm_patches/win_dll_import_generator.cc
)
set_target_properties(
  tvm_ffi_orcjit PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
)
target_include_directories(
  tvm_ffi_orcjit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${LLVM_INCLUDE_DIRS}
)
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
target_compile_definitions(tvm_ffi_orcjit PRIVATE ${LLVM_DEFINITIONS_LIST})

# ---- Static LLVM linking via llvm-config ----
set(_llvm_config_hints ${LLVM_TOOLS_BINARY_DIR} "${LLVM_DIR}/../../../bin")
if (DEFINED ENV{LLVM_PREFIX})
  list(APPEND _llvm_config_hints "$ENV{LLVM_PREFIX}/bin" "$ENV{LLVM_PREFIX}/Library/bin")
endif ()
find_program(
  LLVM_CONFIG_EXE
  NAMES llvm-config-${LLVM_VERSION_MAJOR} llvm-config
  HINTS ${_llvm_config_hints}
)
if (NOT LLVM_CONFIG_EXE)
  message(FATAL_ERROR "llvm-config not found")
endif ()

execute_process(
  COMMAND ${LLVM_CONFIG_EXE} --link-static --libs Core OrcJIT Support native
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE _llvm_libs
)
execute_process(
  COMMAND ${LLVM_CONFIG_EXE} --link-static --ldflags
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE _llvm_ldflags
)
separate_arguments(_llvm_libs_list NATIVE_COMMAND "${_llvm_libs}")
separate_arguments(_llvm_ldflags_list NATIVE_COMMAND "${_llvm_ldflags}")

# ---- zlib (static, from LLVM prefix) ----
cmake_path(GET LLVM_DIR PARENT_PATH _llvm_prefix)
cmake_path(GET _llvm_prefix PARENT_PATH _llvm_prefix)
cmake_path(GET _llvm_prefix PARENT_PATH _llvm_prefix)
set(_lib_hints "${_llvm_prefix}/lib" "${_llvm_prefix}/lib64" "${_llvm_prefix}/Library/lib")

if (WIN32)
  find_library(
    ZLIB_STATIC
    NAMES zlibstatic z
    HINTS ${_lib_hints} REQUIRED
  )
else ()
  # DEMO: prefer the static archive, but fall back to the dynamic lib -- an el8/big-merge prefix
  # may ship only libz.so, and a hard REQUIRED on libz.a would abort configure on a packaging
  # detail rather than exercising the link/auditwheel path we want to measure.
  find_library(
    ZLIB_STATIC
    NAMES libz.a z
    HINTS ${_lib_hints}
  )
  if (NOT ZLIB_STATIC)
    set(ZLIB_STATIC z)
  endif ()
endif ()

# ---- zstd (static, from LLVM prefix) ----
if (WIN32)
  find_library(
    ZSTD_STATIC
    NAMES zstd_static zstd
    HINTS ${_lib_hints} REQUIRED
  )
else ()
  # DEMO: prefer the static archive, fall back to the dynamic lib (see zlib note above).
  find_library(
    ZSTD_STATIC
    NAMES libzstd.a zstd
    HINTS ${_lib_hints}
  )
  if (NOT ZSTD_STATIC)
    set(ZSTD_STATIC zstd)
  endif ()
endif ()

# ---- Link everything ----
target_link_libraries(
  tvm_ffi_orcjit PRIVATE tvm_ffi::header tvm_ffi::shared ${_llvm_ldflags_list} ${_llvm_libs_list}
                         ${ZLIB_STATIC} ${ZSTD_STATIC}
)

# LLVM system libs (ntdll/psapi on Windows, rt/dl on Linux), minus zlib/zstd (linked above).
execute_process(
  COMMAND ${LLVM_CONFIG_EXE} --system-libs
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE _sys_libs
  RESULT_VARIABLE _sys_libs_rc
)
if (_sys_libs_rc EQUAL 0 AND _sys_libs)
  separate_arguments(_sys_libs_list NATIVE_COMMAND "${_sys_libs}")
  list(FILTER _sys_libs_list EXCLUDE REGEX "(zstd|^-lz$|^z\\.lib$|xml2)")
  target_link_libraries(tvm_ffi_orcjit PRIVATE ${_sys_libs_list})
endif ()

# ---- Platform-specific fixups ----
if (APPLE)
  add_custom_command(
    TARGET tvm_ffi_orcjit
    POST_BUILD
    COMMAND install_name_tool -change @rpath/libc++.1.dylib /usr/lib/libc++.1.dylib
            $<TARGET_FILE:tvm_ffi_orcjit>
    COMMENT "Fixing libc++ rpath to use system library"
  )
elseif (UNIX AND NOT WIN32)
  # DEMO (conda-forge LLVM 22.1.0, unchanged from main): libstdc++ is kept DYNAMIC -- no
  # -static-libstdc++.  Hypothesis under test: although conda LLVM is built by conda gcc>=14, the
  # LLVM static archives + this addon only *reference* GLIBCXX symbols up to ~3.4.30 (measured on
  # libLLVM.so), which is <= the GLIBCXX_3.4.33 that the manylinux_2_28 build image's gcc-toolset-14
  # libstdc++ provides.  Static-vs-dynamic does not change which symbol versions are imported (that
  # is code-determined), so a dynamic libstdc++ should still pass auditwheel -- and it lets JIT'd C++
  # resolve its C++ runtime from the process libstdc++.so instead of a static copy baked in here.
  # If auditwheel rejects a >3.4.33 symbol, this hypothesis is wrong and static-libstdc++ must return.
endif ()

# Hide symbols pulled from static archives so they are not re-exported and cannot interpose with a
# host process's own copies (e.g. PyTorch's bundled LLVM).
#
# DEMO: with libstdc++ DYNAMIC (see above), the clean `--exclude-libs,ALL` is safe: it localizes
# every static archive (LLVM/zlib/zstd) while the C++ runtime the JIT needs is served by the dynamic
# libstdc++.so, not by hidden archive symbols.  This replaces the by-name exclude list that the
# static-libstdc++ build required (where ALL would have hidden the baked-in C++ runtime).
if (CMAKE_SYSTEM_NAME MATCHES "Linux|Android|FreeBSD|NetBSD|OpenBSD" AND CMAKE_CXX_COMPILER_ID
                                                                         MATCHES "GNU|Clang"
)
  target_link_options(tvm_ffi_orcjit PRIVATE "-Wl,--exclude-libs,ALL")
endif ()

# ---- Find and embed liborc_rt (Linux/ELF only) ----
# The archive bytes are baked into the .so's .rodata (see src/ffi/orc_rt_embed.S.in) and handed to
# ExecutorNativePlatform as a zero-copy MemoryBuffer at runtime, so the wheel ships one artifact
# with no on-disk path lookup. Platform notes:
#
# * Linux: liborc_rt drives ExecutorNativePlatform → ELFNixPlatform.
# * macOS: MachOPlatform is skipped to sidestep the compact-unwind delta bug (see
#   src/ffi/orcjit_session.cc); liborc_rt is unused.
# * Windows: COFFPlatform is not hooked up; liborc_rt is unused.
set(ORC_RT_PATH "")
if (NOT APPLE AND NOT WIN32)
  if (DEFINED ENV{ORC_RT_PATH})
    set(ORC_RT_PATH "$ENV{ORC_RT_PATH}")
  else ()
    # DEMO: also search lib64 -- an el8 x86_64 RPM may place the clang resource dir under
    # /usr/lib64/clang rather than /usr/lib/clang. GLOB_RECURSE descends into the lib/<triple>
    # or lib/linux subdir where compiler-rt ships liborc_rt.
    file(GLOB_RECURSE _orc_rt_candidates
         "${_llvm_prefix}/lib/clang/*/lib/liborc_rt*.a"
         "${_llvm_prefix}/lib64/clang/*/lib/liborc_rt*.a"
    )
    if (_orc_rt_candidates)
      list(GET _orc_rt_candidates 0 ORC_RT_PATH)
    endif ()
  endif ()
  if (NOT ORC_RT_PATH)
    message(WARNING "Could not find liborc_rt. ORC runtime features will be disabled.")
  endif ()
endif ()

if (ORC_RT_PATH)
  message(STATUS "Embedding liborc_rt: ${ORC_RT_PATH}")
  enable_language(ASM)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/orc_rt_embed.S.in"
    "${CMAKE_CURRENT_BINARY_DIR}/orc_rt_embed.S" @ONLY
  )
  target_sources(tvm_ffi_orcjit PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/orc_rt_embed.S")
  target_compile_definitions(tvm_ffi_orcjit PRIVATE TVM_FFI_ORCJIT_EMBED_ORC_RT=1)
  # Re-run .incbin if the archive itself changes (path is stable across builds).
  set_property(
    SOURCE "${CMAKE_CURRENT_BINARY_DIR}/orc_rt_embed.S" PROPERTY OBJECT_DEPENDS "${ORC_RT_PATH}"
  )
endif ()

# ---- Install ----
install(
  TARGETS tvm_ffi_orcjit
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib
)
