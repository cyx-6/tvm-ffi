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

# Option to use static LLVM linking (default ON for self-contained wheel)
option(TVM_FFI_ORCJIT_STATIC_LLVM "Statically link LLVM libraries" ON)

# Avoid LLVM debug symbol linking issues
add_definitions(-DNDEBUG=1)

# Find LLVM 18+ (required for modern ORC JIT APIs)
include(GNUInstallDirs)
find_package(LLVM REQUIRED CONFIG)
if(LLVM_VERSION_MAJOR LESS 18)
  message(FATAL_ERROR "LLVM 18+ required, found ${LLVM_PACKAGE_VERSION}")
endif()
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM library dir: ${LLVM_LIBRARY_DIR}")

# Get LLVM's cmake dir for finding its bundled packages (like zstd)
list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")

# Find tvm-ffi package
find_package(Python COMPONENTS Interpreter REQUIRED)

# Add conda/Python environment prefix to CMAKE_PREFIX_PATH for finding zstd/zlib
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.prefix)"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYTHON_ENV_PREFIX
)
list(APPEND CMAKE_PREFIX_PATH "${PYTHON_ENV_PREFIX}")
message(STATUS "Python env prefix: ${PYTHON_ENV_PREFIX}")
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE tvm_ffi_ROOT
)
find_package(tvm_ffi CONFIG REQUIRED)

# Find clang for ORC runtime
find_program(CLANG_EXECUTABLE clang HINTS ${LLVM_TOOLS_BINARY_DIR})
if(CLANG_EXECUTABLE)
  execute_process(
    COMMAND ${CLANG_EXECUTABLE} --print-runtime-dir
    OUTPUT_VARIABLE CLANG_RUNTIME_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()

# Find llvm-config for querying system libs
find_program(LLVM_CONFIG_EXE llvm-config HINTS ${LLVM_TOOLS_BINARY_DIR})

# Source files
set(SOURCES src/ffi/orcjit_session.cc src/ffi/orcjit_dylib.cc)

# Build shared library
add_library(tvm_ffi_orcjit SHARED ${SOURCES})

target_include_directories(tvm_ffi_orcjit SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_include_directories(tvm_ffi_orcjit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(tvm_ffi_orcjit PRIVATE ${LLVM_DEFINITIONS})

# LLVM is built without RTTI
if(NOT MSVC)
  set_source_files_properties(${SOURCES} PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()

if(TVM_FFI_ORCJIT_STATIC_LLVM)
  message(STATUS "Using static LLVM linking (for self-contained wheel)")

  if(NOT LLVM_CONFIG_EXE)
    message(FATAL_ERROR "llvm-config not found, required for static linking")
  endif()

  # Use llvm-config --link-static --libfiles to get static .a paths directly
  # This follows TVM's pattern and is more reliable than CMake targets
  execute_process(
    COMMAND ${LLVM_CONFIG_EXE} --link-static --libfiles core orcjit support native
    OUTPUT_VARIABLE LLVM_LIBFILES_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE LLVM_CONFIG_EXE_RESULT
    ERROR_QUIET
  )
  if(NOT "${LLVM_CONFIG_EXE_RESULT}" STREQUAL "0")
    message(FATAL_ERROR "Failed to get LLVM static libraries from llvm-config: ${LLVM_CONFIG_EXE_RESULT}")
  endif()

  # Convert space-separated paths to list
  separate_arguments(LLVM_STATIC_LIBS NATIVE_COMMAND "${LLVM_LIBFILES_RAW}")
  message(STATUS "LLVM static libraries: ${LLVM_STATIC_LIBS}")

  # Query system libraries from llvm-config
  execute_process(
    COMMAND ${LLVM_CONFIG_EXE} --system-libs
    OUTPUT_VARIABLE LLVM_SYSTEM_LIBS_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # Parse system libs and use proper CMake targets where available
  set(LLVM_SYSTEM_LIBS "")
  separate_arguments(LLVM_SYSTEM_LIBS_LIST NATIVE_COMMAND "${LLVM_SYSTEM_LIBS_RAW}")
  foreach(lib ${LLVM_SYSTEM_LIBS_LIST})
    if("${lib}" STREQUAL "-lz")
      find_package(ZLIB REQUIRED)
      list(APPEND LLVM_SYSTEM_LIBS ZLIB::ZLIB)
    elseif("${lib}" STREQUAL "-lzstd")
      find_package(zstd REQUIRED)
      if(TARGET zstd::libzstd_static)
        list(APPEND LLVM_SYSTEM_LIBS zstd::libzstd_static)
      else()
        list(APPEND LLVM_SYSTEM_LIBS zstd::libzstd_shared)
      endif()
    elseif("${lib}" STREQUAL "-lm")
      list(APPEND LLVM_SYSTEM_LIBS m)
    elseif("${lib}" STREQUAL "-lpthread")
      find_package(Threads REQUIRED)
      list(APPEND LLVM_SYSTEM_LIBS Threads::Threads)
    elseif("${lib}" STREQUAL "-ldl")
      list(APPEND LLVM_SYSTEM_LIBS ${CMAKE_DL_LIBS})
    elseif("${lib}" STREQUAL "-lrt")
      list(APPEND LLVM_SYSTEM_LIBS rt)
    elseif(NOT "${lib}" STREQUAL "")
      list(APPEND LLVM_SYSTEM_LIBS "${lib}")
    endif()
  endforeach()

  # LLVM 18+ always requires zstd and zlib, even if llvm-config doesn't report them
  # This is needed for compression support in LLVM bitcode
  find_package(zstd QUIET)
  if(zstd_FOUND AND NOT "zstd::libzstd_static" IN_LIST LLVM_SYSTEM_LIBS AND NOT "zstd::libzstd_shared" IN_LIST LLVM_SYSTEM_LIBS)
    if(TARGET zstd::libzstd_static)
      list(APPEND LLVM_SYSTEM_LIBS zstd::libzstd_static)
      message(STATUS "Adding zstd (static) - required by LLVM")
    elseif(TARGET zstd::libzstd_shared)
      list(APPEND LLVM_SYSTEM_LIBS zstd::libzstd_shared)
      message(STATUS "Adding zstd (shared) - required by LLVM")
    endif()
  endif()

  find_package(ZLIB QUIET)
  if(ZLIB_FOUND AND NOT "ZLIB::ZLIB" IN_LIST LLVM_SYSTEM_LIBS)
    list(APPEND LLVM_SYSTEM_LIBS ZLIB::ZLIB)
    message(STATUS "Adding zlib - required by LLVM")
  endif()

  message(STATUS "LLVM system libs: ${LLVM_SYSTEM_LIBS}")

  # Link with whole-archive for static initializers
  if(APPLE)
    # macOS: -force_load per library
    set(LLVM_LINK_LIBS "")
    foreach(lib ${LLVM_STATIC_LIBS})
      list(APPEND LLVM_LINK_LIBS "-Wl,-force_load,${lib}")
    endforeach()
    target_link_libraries(tvm_ffi_orcjit PRIVATE
      tvm_ffi::header tvm_ffi::shared
      ${LLVM_LINK_LIBS}
      ${LLVM_SYSTEM_LIBS}
    )
  else()
    # Linux: --whole-archive wrapper
    target_link_libraries(tvm_ffi_orcjit PRIVATE
      tvm_ffi::header tvm_ffi::shared
      -Wl,--whole-archive ${LLVM_STATIC_LIBS} -Wl,--no-whole-archive
      ${LLVM_SYSTEM_LIBS}
    )
  endif()
else()
  message(STATUS "Using shared LLVM linking")

  # For shared linking, prefer the monolithic LLVM library
  llvm_map_components_to_libnames(LLVM_LIBS core orcjit support native)
  list(FIND LLVM_LIBS "LLVM" _llvm_dynlib_index)
  if(_llvm_dynlib_index GREATER -1)
    set(LLVM_LIBS LLVM)
    message(STATUS "Using dynamic LLVM library")
  endif()

  target_link_libraries(tvm_ffi_orcjit PRIVATE tvm_ffi::header tvm_ffi::shared ${LLVM_LIBS})
endif()

# Find ORC runtime library for installation
set(ORC_RT_LIB "")
if(CLANG_RUNTIME_DIR AND EXISTS "${CLANG_RUNTIME_DIR}")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(ORC_RT_ARCH "x86_64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ORC_RT_ARCH "aarch64")
  else()
    set(ORC_RT_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  if(APPLE)
    set(ORC_RT_FILENAME "liborc_rt_osx.a")
  else()
    set(ORC_RT_FILENAME "liborc_rt-${ORC_RT_ARCH}.a")
  endif()

  if(EXISTS "${CLANG_RUNTIME_DIR}/${ORC_RT_FILENAME}")
    set(ORC_RT_LIB "${CLANG_RUNTIME_DIR}/${ORC_RT_FILENAME}")
    message(STATUS "Found ORC runtime: ${ORC_RT_LIB}")
  endif()
endif()

# Installation
install(TARGETS tvm_ffi_orcjit LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)
if(ORC_RT_LIB)
  install(FILES ${ORC_RT_LIB} DESTINATION lib)
endif()
