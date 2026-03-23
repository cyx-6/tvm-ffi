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

# TVM-FFI OrcJIT

A Python package that enables dynamic loading of compiled object files (`.o`)
using LLVM ORC JIT v2, providing a flexible JIT execution environment for
TVM-FFI exported functions.

## Features

- **JIT Execution**: Load and execute compiled object files at runtime using LLVM's ORC JIT v2
- **Multiple Libraries**: Create separate dynamic libraries with independent symbol namespaces
- **Incremental Loading**: Add multiple object files to the same library incrementally
- **Symbol Isolation**: Different libraries can define the same symbol without conflicts
- **Init/Fini Support**: Handles static constructors/destructors across ELF (`.init_array`/`.ctors`), Mach-O (`__mod_init_func`), and COFF (`.CRT$XC*`/`.CRT$XT*`)
- **Cross-Platform**: Linux (x86_64, aarch64), macOS (arm64), Windows (AMD64)
- **Multi-Compiler**: Tested with LLVM Clang, GCC, Apple Clang, MSVC, and clang-cl
- **TVM-FFI Integration**: Seamlessly works with TVM-FFI's stable C ABI
- **Python API**: Simple Pythonic interface for JIT compilation and execution

## Supported Platforms and Compilers

Object files compiled with any of the following compiler/platform combinations
can be loaded and executed by the ORC JIT:

| Platform | Compilers | C | C++ |
|----------|-----------|:-:|:---:|
| Linux (x86_64, aarch64) | LLVM Clang, GCC | yes | yes |
| macOS (arm64) | LLVM Clang, Apple Clang | yes | yes |
| Windows (AMD64) | LLVM Clang, MSVC, clang-cl | yes | no |

Windows is C-only across all compilers. C++ objects compiled with
`TVM_FFI_DLL_EXPORT_TYPED_FUNC` use `try`/`catch` (via `TVM_FFI_SAFE_CALL_BEGIN/END`),
which requires Itanium exception ABI symbols (`__cxa_begin_catch`,
`__gxx_personality_v0`, etc.) that the MSVC-built host process cannot provide.
Pure C objects using the `TVMFFISafeCallType` ABI work on all platforms.

## Installation

### Prerequisites

- Python 3.10+
- CMake 3.18+
- LLVM 22+ (with ORC JIT support)
- C/C++ compiler with C++17 support

### Install from PyPI

```bash
pip install apache-tvm-ffi
pip install tvm-ffi-orcjit
```

### Development Installation

```bash
git clone --recursive https://github.com/cyx-6/tvm-ffi.git
cd tvm-ffi/addons/tvm-ffi-orcjit
pip install -e .
```

## Usage

### Basic Example

```python
from tvm_ffi_orcjit import ExecutionSession

# Create an execution session
session = ExecutionSession()

# Create a dynamic library
lib = session.create_library()

# Load an object file
lib.add("example.o")

# Get and call a function
add_func = lib.get_function("add")
result = add_func(1, 2)
print(f"Result: {result}")  # Output: Result: 3
```

### Multiple Libraries with Symbol Isolation

```python
session = ExecutionSession()

lib1 = session.create_library("lib1")
lib2 = session.create_library("lib2")

lib1.add("implementation_v1.o")
lib2.add("implementation_v2.o")

add_v1 = lib1.get_function("add")
add_v2 = lib2.get_function("add")

print(add_v1(5, 3))  # Uses implementation from lib1
print(add_v2(5, 3))  # Uses implementation from lib2
```

### Cross-Library Linking

```python
session = ExecutionSession()

base_lib = session.create_library("base")
base_lib.add("math_ops.o")

caller_lib = session.create_library("caller")
caller_lib.set_link_order(base_lib)  # Can resolve symbols from base_lib
caller_lib.add("caller.o")

result = caller_lib.get_function("call_math")(10, 20)
```

## Writing Functions for OrcJIT

### C++ (Linux/macOS)

```cpp
#include <tvm/ffi/function.h>

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add, [](int a, int b) {
    return a + b;
});
```

Compile: `clang++ -std=c++17 -fPIC -O2 -c -o example.o example.cc`

### Pure C (all platforms including Windows)

```c
#include <tvm/ffi/c_api.h>

TVM_FFI_DLL_EXPORT int __tvm_ffi_add(
    void* self, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result) {
  result->type_index = kTVMFFIInt;
  result->v_int64 = args[0].v_int64 + args[1].v_int64;
  return 0;
}
```

Compile: `clang -O2 -c -o example.o example.c`

## How It Works

- **LLJIT**: Built on LLVM's ORC JIT v2 with `ObjectLinkingLayer` (JITLink) for
  all platforms.
- **InitFiniPlugin**: Custom `ObjectLinkingLayer::Plugin` that collects function
  pointers from init/fini sections (ELF `.init_array`/`.ctors`/`.fini_array`/`.dtors`,
  Mach-O `__mod_init_func`/`__mod_term_func`, COFF `.CRT$XC*`/`.CRT$XT*`) and
  runs them in priority order at symbol lookup / library teardown.
- **DLL Import Stubs** (Windows): Custom `DefinitionGenerator` that resolves host
  process symbols from all loaded DLLs and creates `__imp_*` pointer stubs in
  JIT memory, keeping all fixups within PCRel32 range.
- **SEH Stripping** (Windows): `ObjectTransformLayer` strips `.pdata`/`.xdata`
  relocations from COFF objects before JITLink graph building, working around a
  JITLink limitation with COMDAT section symbols.

## Project Structure

```text
tvm-ffi-orcjit/
├── CMakeLists.txt              # Build configuration
├── pyproject.toml              # Python package metadata
├── src/ffi/
│   ├── orcjit_session.cc       # ExecutionSession (LLJIT setup, plugins)
│   ├── orcjit_session.h
│   ├── orcjit_dylib.cc         # DynamicLibrary (object loading, symbol lookup)
│   ├── orcjit_dylib.h
│   └── orcjit_utils.h          # LLVM error handling utilities
├── python/tvm_ffi_orcjit/
│   ├── __init__.py             # Module exports and library loading
│   ├── session.py              # Python ExecutionSession wrapper
│   └── dylib.py                # Python DynamicLibrary wrapper
├── tests/                      # See tests/README.md
├── examples/quick-start/       # Complete example with CMake
└── tools/
    ├── install_llvm.sh         # LLVM installation helper (Linux/macOS)
    └── install_llvm.ps1        # LLVM installation helper (Windows)
```

## CI

Runs on Linux (x86_64, aarch64), macOS (arm64), Windows (AMD64) via
`cibuildwheel`. Each platform builds test objects with multiple compilers
and runs the full test suite. See `.github/workflows/orcjit-build.yml`.

## Troubleshooting

### "Cannot find global function" error

The shared library wasn't loaded. Reinstall: `pip install --force-reinstall tvm-ffi-orcjit`

### "Duplicate definition of symbol" error

Use separate libraries for different implementations of the same symbol.

### "Symbol not found" error

Ensure functions are exported with TVM-FFI macros (`TVM_FFI_DLL_EXPORT_TYPED_FUNC`
for C++, or `__tvm_ffi_` prefix for C).

### Relocation errors on Windows

MSVC/clang-cl objects must be compiled with `/GS-` to disable buffer security
checks (`__security_cookie`) which are CRT symbols the JIT cannot resolve.

### LLVM version mismatch

The package requires LLVM 22+. Set `LLVM_DIR` if CMake can't find it:

```bash
export LLVM_DIR=/path/to/llvm/lib/cmake/llvm
```

## License

Apache License 2.0
