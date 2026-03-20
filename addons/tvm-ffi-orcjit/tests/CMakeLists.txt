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

cmake_minimum_required(VERSION 3.18)
project(tvm_ffi_orcjit_tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  add_compile_options(-mno-outline-atomics)
endif()

# Find tvm-ffi package
find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE tvm_ffi_ROOT
)
find_package(tvm_ffi CONFIG REQUIRED)

# Helper: build a single source file into an object and install as <name>.o
function(add_test_object name source)
  get_filename_component(_ext "${source}" EXT)
  if(_ext STREQUAL ".c")
    set(_lang C)
  else()
    set(_lang CXX)
  endif()
  add_library(${name}_obj OBJECT "${source}")
  target_link_libraries(${name}_obj PRIVATE tvm_ffi::header)
  if(_lang STREQUAL "C" AND MSVC)
    target_compile_options(${name}_obj PRIVATE /O2)
  else()
    target_compile_options(${name}_obj PRIVATE -O2)
  endif()
  install(
    FILES $<TARGET_OBJECTS:${name}_obj>
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
    RENAME ${name}.o
  )
endfunction()

# C++ object files — skip on Windows (C-only strategy for ORC JIT on Windows)
if(NOT WIN32)
  add_test_object(test_funcs          sources/test_funcs.cc)
  add_test_object(test_funcs2         sources/test_funcs2.cc)
  add_test_object(test_funcs_conflict sources/test_funcs_conflict.cc)
  add_test_object(test_ctor_dtor      sources/test_ctor_dtor.cc)
  add_test_object(test_call_global    sources/test_call_global.cc)
endif()

# Pure C object files — built on all platforms (no C++ runtime deps)
enable_language(C)
add_test_object(test_funcs_c          sources/test_funcs_c.c)
add_test_object(test_funcs2_c         sources/test_funcs2_c.c)
add_test_object(test_funcs_conflict_c sources/test_funcs_conflict_c.c)
add_test_object(test_call_global_c    sources/test_call_global_c.c)

# CUDA object files — optional
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
  add_test_object(test_funcs_cuda sources/test_funcs_cuda.cu)
endif()
