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
project(tvm_ffi_orcjit_example)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Run `python -m tvm_ffi.config --cmakedir` to find tvm-ffi package
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

# C++ object file — skip on Windows (C-only strategy for ORC JIT on Windows)
if(NOT WIN32)
  add_library(add_obj OBJECT add.cc)
  target_link_libraries(add_obj PRIVATE tvm_ffi::header)
  target_compile_options(add_obj PRIVATE -O2)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(add_obj PRIVATE -mno-outline-atomics)
  endif()
  add_custom_target(
    copy_cpp_obj ALL
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:add_obj> ${CMAKE_CURRENT_SOURCE_DIR}/add.o
    DEPENDS add_obj
    COMMENT "Copying add.o to example directory"
  )
endif()

# Pure C object file — built on all platforms
enable_language(C)
add_library(add_c_obj OBJECT add_c.c)
target_link_libraries(add_c_obj PRIVATE tvm_ffi::header)
if(MSVC)
  target_compile_options(add_c_obj PRIVATE /O2)
else()
  target_compile_options(add_c_obj PRIVATE -O2)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(add_c_obj PRIVATE -mno-outline-atomics)
  endif()
endif()
add_custom_target(
  copy_c_obj ALL
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:add_c_obj> ${CMAKE_CURRENT_SOURCE_DIR}/add_c.o
  DEPENDS add_c_obj
  COMMENT "Copying add_c.o to example directory"
)
