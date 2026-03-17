/**
 * Standalone LLVM ORC JIT debug test for Windows.
 *
 * This program isolates the symbol resolution issue from tvm-ffi.
 * It creates an LLJIT, loads a COFF object file, and reports exactly
 * what symbols fail to resolve and why.
 *
 * Build (on Windows with conda LLVM):
 *   clang++ -std=c++17 -O2 debug_win_jit.cpp -o debug_win_jit.exe \
 *     $(llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native)
 *
 * Or via CMake (see debug_win_jit_CMakeLists.txt).
 */

#include <llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdio>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#endif

using namespace llvm;
using namespace llvm::orc;

// Custom error-reporting definition generator that logs everything
class DebugDefinitionGenerator : public DefinitionGenerator {
public:
  Error tryToGenerate(LookupState &LS, LookupKind K,
                      JITDylib &JD,
                      JITDylibLookupFlags JDLookupFlags,
                      const SymbolLookupSet &LookupSet) override {
    fprintf(stderr, "\n[DebugGen] tryToGenerate called with %zu symbols:\n",
            LookupSet.size());

    SymbolMap NewSymbols;
    for (auto &[Name, Flags] : LookupSet) {
      StringRef NameStr = *Name;
      fprintf(stderr, "  [DebugGen] Looking up: '%s'\n", NameStr.str().c_str());

#ifdef _WIN32
      // Try the __imp_ stripping approach
      if (NameStr.starts_with("__imp_")) {
        std::string RealName = NameStr.drop_front(6).str();
        fprintf(stderr, "    -> __imp_ prefix detected, real name: '%s'\n",
                RealName.c_str());

        // Try each runtime DLL
        static const char *kDLLs[] = {
            "vcruntime140.dll", "vcruntime140_1.dll", "ucrtbase.dll",
            "msvcp140.dll",
        };
        for (const char *dll : kDLLs) {
          HMODULE hMod = LoadLibraryA(dll);
          if (!hMod) {
            fprintf(stderr, "    -> LoadLibraryA('%s') FAILED\n", dll);
            continue;
          }
          FARPROC addr = GetProcAddress(hMod, RealName.c_str());
          fprintf(stderr, "    -> GetProcAddress(%s, '%s') = %p\n", dll,
                  RealName.c_str(), (void *)addr);
          if (addr) {
            // Create a stub: allocate a pointer that holds the function address
            auto *stub = new uint64_t(reinterpret_cast<uint64_t>(addr));
            NewSymbols[Name] = {ExecutorAddr::fromPtr(stub),
                                JITSymbolFlags::Exported};
            fprintf(stderr, "    -> RESOLVED via __imp_ stub\n");
            break;
          }
        }
      } else {
        // Direct symbol lookup
        // Try runtime DLLs first
        static const char *kDLLs[] = {
            "vcruntime140.dll", "vcruntime140_1.dll", "ucrtbase.dll",
            "msvcp140.dll",
        };
        for (const char *dll : kDLLs) {
          HMODULE hMod = LoadLibraryA(dll);
          if (!hMod)
            continue;
          FARPROC addr = GetProcAddress(hMod, NameStr.str().c_str());
          if (addr) {
            fprintf(stderr, "    -> Found in %s at %p\n", dll, (void *)addr);
            NewSymbols[Name] = {ExecutorAddr::fromPtr(addr),
                                JITSymbolFlags::Exported};
            break;
          }
        }

        if (NewSymbols.find(Name) == NewSymbols.end()) {
          // Try all loaded modules
          HMODULE hMods[1024];
          DWORD cbNeeded;
          if (EnumProcessModules(GetCurrentProcess(), hMods, sizeof(hMods),
                                 &cbNeeded)) {
            DWORD count = cbNeeded / sizeof(HMODULE);
            if (count > 1024)
              count = 1024;
            for (DWORD i = 0; i < count; ++i) {
              FARPROC addr =
                  GetProcAddress(hMods[i], NameStr.str().c_str());
              if (addr) {
                char modName[260];
                GetModuleFileNameA(hMods[i], modName, sizeof(modName));
                fprintf(stderr, "    -> Found in module '%s' at %p\n", modName,
                        (void *)addr);
                NewSymbols[Name] = {ExecutorAddr::fromPtr(addr),
                                    JITSymbolFlags::Exported};
                break;
              }
            }
          }
        }

        if (NewSymbols.find(Name) == NewSymbols.end()) {
          // Try LLVM's DynamicLibrary
          void *addr =
              sys::DynamicLibrary::SearchForAddressOfSymbol(NameStr.str());
          if (addr) {
            fprintf(stderr, "    -> Found via LLVM SearchForAddressOfSymbol: %p\n",
                    addr);
            NewSymbols[Name] = {ExecutorAddr::fromPtr(addr),
                                JITSymbolFlags::Exported};
          } else {
            fprintf(stderr, "    -> NOT FOUND ANYWHERE\n");
          }
        }
      }
#endif
    }

    fprintf(stderr, "[DebugGen] Resolved %zu of %zu symbols\n",
            NewSymbols.size(), LookupSet.size());

    if (!NewSymbols.empty())
      return JD.define(absoluteSymbols(std::move(NewSymbols)));
    return Error::success();
  }
};

int main(int argc, char *argv[]) {
  fprintf(stderr, "=== LLVM ORC JIT Debug Test ===\n");

  // Initialize LLVM
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();

#ifdef _WIN32
  // Pre-load runtime DLLs
  fprintf(stderr, "\n--- Pre-loading runtime DLLs ---\n");
  for (const char *dll : {"vcruntime140.dll", "ucrtbase.dll", "msvcp140.dll"}) {
    std::string err;
    bool failed = sys::DynamicLibrary::LoadLibraryPermanently(dll, &err);
    fprintf(stderr, "  LoadLibraryPermanently(%s): %s\n", dll,
            failed ? err.c_str() : "OK");
  }

  // Test direct GetProcAddress
  fprintf(stderr, "\n--- Direct GetProcAddress test ---\n");
  HMODULE hVcrt = LoadLibraryA("vcruntime140.dll");
  fprintf(stderr, "  vcruntime140.dll handle: %p\n", (void *)hVcrt);
  if (hVcrt) {
    const char *test_syms[] = {
        "_Init_thread_header", "_Init_thread_footer",
        "_Init_thread_abort",  "??_7type_info@@6B@",
        "??3@YAXPEAX_K@Z",
    };
    for (const char *sym : test_syms) {
      FARPROC addr = GetProcAddress(hVcrt, sym);
      fprintf(stderr, "  GetProcAddress(vcrt, \"%s\") = %p\n", sym,
              (void *)addr);
    }
  }

  // Test LLVM's SearchForAddressOfSymbol
  fprintf(stderr, "\n--- LLVM SearchForAddressOfSymbol test ---\n");
  {
    const char *test_syms[] = {
        "_Init_thread_header", "_Init_thread_footer",
        "_Init_thread_abort",  "??_7type_info@@6B@",
        "??3@YAXPEAX_K@Z",
    };
    for (const char *sym : test_syms) {
      void *addr = sys::DynamicLibrary::SearchForAddressOfSymbol(sym);
      fprintf(stderr, "  SearchForAddressOfSymbol(\"%s\") = %p\n", sym, addr);
    }
  }
#endif

  // Create LLJIT with JITLink on Windows
  fprintf(stderr, "\n--- Creating LLJIT ---\n");
  auto builder = LLJITBuilder();

#ifdef _WIN32
  builder.setObjectLinkingLayerCreator(
      [](ExecutionSession &ES)
          -> Expected<std::unique_ptr<ObjectLayer>> {
        return std::make_unique<ObjectLinkingLayer>(ES);
      });
#endif

  auto jit_or_err = builder.create();
  if (!jit_or_err) {
    fprintf(stderr, "  FAILED to create LLJIT: ");
    logAllUnhandledErrors(jit_or_err.takeError(), errs());
    return 1;
  }
  auto &jit = *jit_or_err;
  fprintf(stderr, "  LLJIT created successfully\n");

  // Check ProcessSymbols JITDylib
  auto PSG = jit->getProcessSymbolsJITDylib();
  fprintf(stderr, "  ProcessSymbols JITDylib: %s\n",
          PSG ? "exists" : "NULL");

  // Add our debug generator
  if (PSG) {
    PSG->addGenerator(std::make_unique<DebugDefinitionGenerator>());
    fprintf(stderr, "  Added DebugDefinitionGenerator to ProcessSymbols\n");
  }

  // Try to load and resolve a COFF object file
  if (argc > 1) {
    const char *obj_path = argv[1];
    fprintf(stderr, "\n--- Loading object file: %s ---\n", obj_path);

    auto buf_or_err = MemoryBuffer::getFile(obj_path);
    if (!buf_or_err) {
      fprintf(stderr, "  Failed to read: %s\n",
              buf_or_err.getError().message().c_str());
      return 1;
    }

    // Create a JITDylib for the object
    auto &JD =
        jit->getExecutionSession().createBareJITDylib("test_dylib");

    // Add default link order
    for (auto &kv : jit->defaultLinkOrder()) {
      JD.addToLinkOrder(*kv.first, kv.second);
    }
    fprintf(stderr, "  Created JITDylib 'test_dylib' with default link order\n");

    // Add the object file
    if (auto err = jit->addObjectFile(JD, std::move(*buf_or_err))) {
      fprintf(stderr, "  Failed to add object file: ");
      logAllUnhandledErrors(std::move(err), errs());
      fprintf(stderr, "\n");
    } else {
      fprintf(stderr, "  Object file added successfully\n");
    }

    // Register an error handler to capture details
    jit->getExecutionSession().setErrorReporter([](Error Err) {
      fprintf(stderr, "\n[ERROR REPORTER] ");
      logAllUnhandledErrors(std::move(Err), errs());
      fprintf(stderr, "\n");
    });

    // Try to look up a symbol (this triggers materialization)
    fprintf(stderr, "\n--- Looking up symbol ---\n");
    // Try a few common symbol names
    for (const char *sym : {"test_add", "__tvm_ffi_test_add", "main",
                            "__tvm_ffi_main"}) {
      fprintf(stderr, "  Trying '%s'...\n", sym);
      auto addr_or_err = jit->lookup(JD, sym);
      if (addr_or_err) {
        fprintf(stderr, "  -> Found at %p\n",
                addr_or_err->toPtr<void *>());
      } else {
        fprintf(stderr, "  -> ");
        logAllUnhandledErrors(addr_or_err.takeError(), errs());
        fprintf(stderr, "\n");
      }
    }
  } else {
    fprintf(stderr,
            "\nNo object file specified. Usage: %s <path-to-test.o>\n",
            argv[0]);
    fprintf(stderr, "  Will skip COFF object loading test.\n");
  }

  fprintf(stderr, "\n=== Debug test complete ===\n");
  return 0;
}
