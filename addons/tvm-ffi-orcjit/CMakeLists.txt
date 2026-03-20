cmake_minimum_required(VERSION 3.20)
project(
  tvm_ffi_orcjit
  VERSION 0.1.0
  LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Append CMAKE_PREFIX_PATH from environment. scikit-build-core overrides
# CMAKE_PREFIX_PATH via its CMake init cache, so any value from the user's
# environment would be lost. Re-appending ensures find_package(LLVM) works
# regardless of where LLVM is installed.
if(DEFINED ENV{CMAKE_PREFIX_PATH})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")
endif()

# Find dependencies
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Find tvm-ffi package using the same method as quickstart example
find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED
)

# Get the cmake directory for tvm-ffi
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE tvm_ffi_ROOT
)
message(STATUS "tvm_ffi_ROOT: ${tvm_ffi_ROOT}")

find_package(tvm_ffi CONFIG REQUIRED)

# Source files
set(SOURCES src/ffi/orcjit_session.cc src/ffi/orcjit_dylib.cc)

# Build shared library
add_library(tvm_ffi_orcjit SHARED ${SOURCES})

target_include_directories(
  tvm_ffi_orcjit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${LLVM_INCLUDE_DIRS}
)

# Compile definitions
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
target_compile_definitions(tvm_ffi_orcjit PRIVATE ${LLVM_DEFINITIONS_LIST})

# ---- Static LLVM Linking via llvm-config ----
# Allow overriding via env: LLVM_CONFIG_PATH
if(DEFINED ENV{LLVM_CONFIG_PATH})
  set(LLVM_CONFIG_EXE "$ENV{LLVM_CONFIG_PATH}")
else()
  find_program(
    LLVM_CONFIG_EXE
    NAMES llvm-config-${LLVM_VERSION_MAJOR} llvm-config
    HINTS ${LLVM_TOOLS_BINARY_DIR} "${LLVM_DIR}/../../../bin"
  )
endif()
if(NOT LLVM_CONFIG_EXE)
  message(FATAL_ERROR "llvm-config not found, required for static LLVM linking")
endif()
message(STATUS "Using llvm-config: ${LLVM_CONFIG_EXE}")

execute_process(
  COMMAND ${LLVM_CONFIG_EXE} --link-static --libs Core OrcJIT Support native
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE _llvm_static_libs
)
execute_process(
  COMMAND ${LLVM_CONFIG_EXE} --link-static --ldflags
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE _llvm_static_ldflags
)
message(STATUS "LLVM static libs: ${_llvm_static_libs}")
message(STATUS "LLVM static ldflags: ${_llvm_static_ldflags}")

separate_arguments(_llvm_static_libs_list NATIVE_COMMAND "${_llvm_static_libs}")
separate_arguments(_llvm_static_ldflags_list NATIVE_COMMAND "${_llvm_static_ldflags}")

# Find static zlib and zstd from the LLVM prefix to avoid dynamic @rpath
# dependencies (macOS) and to enable manylinux_2_28 compatibility (Linux).
cmake_path(GET LLVM_DIR PARENT_PATH _llvm_prefix)
cmake_path(GET _llvm_prefix PARENT_PATH _llvm_prefix)
cmake_path(GET _llvm_prefix PARENT_PATH _llvm_prefix)
if(WIN32)
  find_library(ZLIB_STATIC NAMES zlibstatic z HINTS "${_llvm_prefix}/lib" REQUIRED)
  find_library(ZSTD_STATIC NAMES zstd_static zstd HINTS "${_llvm_prefix}/lib" REQUIRED)
else()
  find_library(ZLIB_STATIC NAMES libz.a HINTS "${_llvm_prefix}/lib" REQUIRED)
  find_library(ZSTD_STATIC NAMES libzstd.a HINTS "${_llvm_prefix}/lib" "${_llvm_prefix}/lib64" REQUIRED)
endif()
message(STATUS "Static zlib: ${ZLIB_STATIC}")
message(STATUS "Static zstd: ${ZSTD_STATIC}")

target_link_libraries(
  tvm_ffi_orcjit
  PRIVATE tvm_ffi::header tvm_ffi::shared ${_llvm_static_ldflags_list}
          ${_llvm_static_libs_list} ${ZLIB_STATIC} ${ZSTD_STATIC}
)
# Link LLVM system library dependencies (e.g. ntdll, psapi on Windows; rt, dl on Linux).
# We use llvm-config --system-libs but filter out zlib/zstd since we link our own static copies.
execute_process(
  COMMAND ${LLVM_CONFIG_EXE} --system-libs
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE _llvm_system_libs
  RESULT_VARIABLE _llvm_system_libs_rc
)
if(_llvm_system_libs_rc EQUAL 0 AND _llvm_system_libs)
  separate_arguments(_llvm_system_libs_list NATIVE_COMMAND "${_llvm_system_libs}")
  # Linux: -lz, -lzstd; Windows: z.lib, zstd.dll.lib, xml2.lib (conda-forge dep)
  list(FILTER _llvm_system_libs_list EXCLUDE REGEX "(zstd|^-lz$|^z\\.lib$|xml2)")
  message(STATUS "LLVM system libs: ${_llvm_system_libs_list}")
  target_link_libraries(tvm_ffi_orcjit PRIVATE ${_llvm_system_libs_list})
endif()

if(APPLE)
  # Fix @rpath/libc++.1.dylib (conda-forge) → /usr/lib/libc++.1.dylib (system)
  add_custom_command(TARGET tvm_ffi_orcjit POST_BUILD
    COMMAND install_name_tool -change @rpath/libc++.1.dylib /usr/lib/libc++.1.dylib $<TARGET_FILE:tvm_ffi_orcjit>
    COMMENT "Fixing libc++ rpath to use system library"
  )
elseif(UNIX AND NOT WIN32)
  # Statically link libstdc++ and libgcc to avoid GLIBCXX version requirements
  # on the target system, enabling manylinux_2_28 support.
  target_link_options(tvm_ffi_orcjit PRIVATE -static-libstdc++ -static-libgcc)
endif()

# ---- Find and bundle liborc_rt ----
# Search for liborc_rt under the LLVM prefix (provided by compiler-rt package).
# No clang binary is needed — we search the lib/clang/*/lib/ tree directly.
set(ORC_RT_PATH "")

if(DEFINED ENV{ORC_RT_PATH})
  # Allow explicit override via environment variable
  set(ORC_RT_PATH "$ENV{ORC_RT_PATH}")
else()
  if(WIN32)
    file(GLOB_RECURSE _orc_rt_candidates "${_llvm_prefix}/lib/clang/*/lib/orc_rt*.lib")
    if(NOT _orc_rt_candidates)
      file(GLOB_RECURSE _orc_rt_candidates "${_llvm_prefix}/Library/lib/clang/*/lib/orc_rt*.lib")
    endif()
  else()
    file(GLOB_RECURSE _orc_rt_candidates "${_llvm_prefix}/lib/clang/*/lib/liborc_rt*.a")
  endif()
  if(_orc_rt_candidates)
    list(GET _orc_rt_candidates 0 ORC_RT_PATH)
  endif()
endif()

if(NOT ORC_RT_PATH)
  message(WARNING "Could not find liborc_rt. "
    "ORC runtime features will be disabled. "
    "Ensure clang runtime libraries are installed (e.g. libclang-rt-dev).")
endif()

if(ORC_RT_PATH)
  message(STATUS "Found liborc_rt: ${ORC_RT_PATH}")
  # Copy into build dir for editable installs (where _lib_dir resolves to CMAKE_BINARY_DIR)
  file(COPY "${ORC_RT_PATH}" DESTINATION "${CMAKE_BINARY_DIR}")
  install(FILES "${ORC_RT_PATH}" DESTINATION lib)
endif()

# Installation rules
install(
  TARGETS tvm_ffi_orcjit
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib
)
