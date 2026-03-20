<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Quick Start Example

This example demonstrates the basic usage of tvm-ffi-orcjit to compile functions and load them dynamically at runtime. Both C++ and pure C variants are included.

## What's Included

- `add.cc` - C++ source file with math functions exported via `TVM_FFI_DLL_EXPORT_TYPED_FUNC`
- `add_c.c` - Pure C source file with the same functions exported via `TVMFFISafeCallType` ABI
- `run.py` - Python script that loads and calls the compiled functions
- `CMakeLists.txt` - CMake configuration to compile both variants

## Prerequisites

- Python 3.8+
- CMake 3.18+
- C/C++ compiler (g++, clang++, or MSVC)
- TVM-FFI and tvm-ffi-orcjit packages

## Steps

### 1. Compile the object files

```bash
cmake -B build
cmake --build build
```

This produces `add.o` (C++) and `add_c.o` (pure C). On Windows, only the C variant is built.

### 2. Run the Python loader

```bash
# C++ variant (default) — Linux/macOS only
python run.py

# Pure C variant — works on all platforms including Windows
python run.py --lang c
```

## How It Works

### C++ Variant (`add.cc`)

Functions are exported using the `TVM_FFI_DLL_EXPORT_TYPED_FUNC` macro, which provides automatic type marshaling. Supports `std::string` and other C++ types.

### C Variant (`add_c.c`)

Functions are exported using the `TVMFFISafeCallType` ABI directly (`__tvm_ffi_<name>` symbol prefix). No C++ runtime dependencies — works on all platforms including Windows ORC JIT.

### Python Side (`run.py`)

- `ExecutionSession()` creates an ORC JIT execution session
- `session.create_library()` creates a dynamic library (JITDylib)
- `lib.add()` loads the `.o` file into the JIT
- `lib.get_function()` looks up symbols in the JIT-compiled code
- Functions are called like normal Python functions
