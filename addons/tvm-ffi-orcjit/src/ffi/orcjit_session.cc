/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file orcjit_session.cc
 * \brief LLVM ORC JIT ExecutionSession implementation
 */

#include "orcjit_session.h"

#include <llvm/ExecutionEngine/JITLink/JITLink.h>
#include <llvm/ExecutionEngine/JITLink/x86_64.h>
#include <llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ObjectTransformLayer.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/TargetParser/SubtargetFeature.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstddef>
#include <cstring>

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

#include "orcjit_dylib.h"
#include "orcjit_utils.h"

namespace tvm {
namespace ffi {
namespace orcjit {

// Initialize LLVM native target (only once)
struct LLVMInitializer {
  LLVMInitializer() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  }
};

static LLVMInitializer llvm_initializer;

class InitFiniPlugin : public llvm::orc::ObjectLinkingLayer::Plugin {
  ORCJITExecutionSession session_;

 public:
  explicit InitFiniPlugin(ORCJITExecutionSession session) : session_(std::move(session)) {}

  void modifyPassConfig(llvm::orc::MaterializationResponsibility& MR, llvm::jitlink::LinkGraph& G,
                        llvm::jitlink::PassConfiguration& Config) override {
    auto& jit_dylib = MR.getTargetJITDylib();
    // Mark all init/fini section blocks and their edge targets as live
    // so they survive dead-stripping.
    Config.PrePrunePasses.emplace_back([](llvm::jitlink::LinkGraph& G) {
      for (auto& Section : G.sections()) {
        auto section_name = Section.getName();
        // ELF: .init_array*, .fini_array*, .ctors*, .dtors*
        // Mach-O: __DATA,__mod_init_func, __DATA,__mod_term_func
        // COFF: .CRT$XC* (ctors), .CRT$XT* (dtors)
        if (section_name.starts_with(".init_array") || section_name.starts_with(".fini_array") ||
            section_name.starts_with(".ctors") || section_name.starts_with(".dtors") ||
            section_name == "__DATA,__mod_init_func" ||
            section_name == "__DATA,__mod_term_func" ||
            section_name.starts_with(".CRT$XC") || section_name.starts_with(".CRT$XT")) {
          for (auto* Block : Section.blocks()) {
            bool has_live_sym = false;
            for (auto* Sym : G.defined_symbols()) {
              if (&Sym->getBlock() == Block) {
                Sym->setLive(true);
                has_live_sym = true;
              }
            }
            // MSVC may emit .CRT$XC* blocks with data but no symbol table
            // entries (static variables in __declspec(allocate) sections).
            // Add an anonymous symbol so the block survives dead-stripping.
            if (!has_live_sym) {
              G.addAnonymousSymbol(*Block, 0, Block->getSize(), false, true).setLive(true);
            }
            for (auto& Edge : Block->edges()) {
              Edge.getTarget().setLive(true);
            }
          }
        }
      }
      return llvm::Error::success();
    });
#ifdef _WIN32
    // Without COFFPlatform, __ImageBase (used by IMAGE_REL_AMD64_ADDR32NB /
    // Pointer32NB relocations) defaults to 0. This causes all Pointer32 fixups
    // to overflow since JIT addresses don't fit in 32 bits.
    //
    // Fix: set __ImageBase to the lowest block address in the graph after
    // allocation. This makes all intra-graph Pointer32NB offsets small.
    //
    // Also strip .pdata/.xdata edges: SEH unwind data references external
    // handlers (e.g., __CxxFrameHandler3) in DLLs that may be >4GB from
    // __ImageBase. Since we don't call RtlAddFunctionTable, SEH data is
    // unused anyway.
    Config.PostAllocationPasses.emplace_back([](llvm::jitlink::LinkGraph& G) {
      // Set __ImageBase to the lowest allocated block address.
      auto ImageBaseName = G.intern("__ImageBase");
      llvm::jitlink::Symbol* ImageBase = nullptr;
      for (auto* Sym : G.external_symbols()) {
        if (Sym->getName() == ImageBaseName) {
          ImageBase = Sym;
          break;
        }
      }
      if (ImageBase) {
        llvm::orc::ExecutorAddr BaseAddr;
        for (auto* B : G.blocks()) {
          if (!BaseAddr || B->getAddress() < BaseAddr) {
            BaseAddr = B->getAddress();
          }
        }
        ImageBase->getAddressable().setAddress(BaseAddr);
      }
      // Strip .pdata/.xdata edges: external handlers may be >4GB from
      // __ImageBase, and we don't register SEH data anyway.
      for (auto& Sec : G.sections()) {
        if (Sec.getName() == ".pdata" || Sec.getName().starts_with(".xdata")) {
          for (auto* B : Sec.blocks()) {
            while (!B->edges_empty()) {
              B->removeEdge(B->edges().begin());
            }
          }
        }
      }
      return llvm::Error::success();
    });
#endif
    // After fixups, read resolved function pointers from all init/fini data sections.
    // Handles ELF (.init_array, .ctors, .fini_array, .dtors),
    // Mach-O (__DATA,__mod_init_func, __DATA,__mod_term_func),
    // and COFF (.CRT$XC*, .CRT$XT*) section conventions.
    Config.PostFixupPasses.emplace_back([this, &jit_dylib](llvm::jitlink::LinkGraph& G) {
      using Entry = ORCJITExecutionSessionObj::InitFiniEntry;
      for (auto& Sec : G.sections()) {
        auto section_name = Sec.getName();

        // --- ELF sections ---
        bool is_init_array = section_name.starts_with(".init_array");
        bool is_ctors = section_name.starts_with(".ctors");
        bool is_fini_array = section_name.starts_with(".fini_array");
        bool is_dtors = section_name.starts_with(".dtors");
        // --- Mach-O sections ---
        bool is_mod_init = (section_name == "__DATA,__mod_init_func");
        bool is_mod_term = (section_name == "__DATA,__mod_term_func");
        // --- COFF sections ---
        bool is_crt_xc = section_name.starts_with(".CRT$XC");
        bool is_crt_xt = section_name.starts_with(".CRT$XT");

        if (!is_init_array && !is_ctors && !is_fini_array && !is_dtors && !is_mod_init &&
            !is_mod_term && !is_crt_xc && !is_crt_xt)
          continue;

        int priority = 0;
        Entry::Section sec;
        bool is_init;
        // ELF default priority for sections without a numeric suffix is 65535.
        // Lower priority numbers run first for .init_array; .fini_array and .ctors
        // negate so that higher-numbered entries run first (reverse order).
        if (is_init_array) {
          if (section_name.consume_front(".init_array.")) {
            section_name.getAsInteger(10, priority);
          } else {
            priority = 65535;
          }
          sec = Entry::Section::kInitArray;
          is_init = true;
        } else if (is_ctors) {
          if (section_name.consume_front(".ctors.") && !section_name.getAsInteger(10, priority)) {
            priority = -priority;
          }
          sec = Entry::Section::kCtors;
          is_init = true;
        } else if (is_fini_array) {
          if (section_name.consume_front(".fini_array.") &&
              !section_name.getAsInteger(10, priority)) {
            priority = -priority;
          } else {
            priority = -65535;
          }
          sec = Entry::Section::kFiniArray;
          is_init = false;
        } else if (is_dtors) {
          if (section_name.consume_front(".dtors.")) {
            section_name.getAsInteger(10, priority);
          }
          sec = Entry::Section::kDtors;
          is_init = false;
        } else if (is_mod_init) {
          // Mach-O __mod_init_func: no priority system, treated as init_array
          sec = Entry::Section::kInitArray;
          is_init = true;
        } else if (is_mod_term) {
          // Mach-O __mod_term_func: no priority system, treated as fini_array
          sec = Entry::Section::kFiniArray;
          is_init = false;
        } else if (is_crt_xc) {
          // COFF .CRT$XC[suffix]: C++ constructors, sorted alphabetically by suffix.
          // Convert suffix to integer priority that preserves alphabetical ordering.
          // E.g., .CRT$XCA → 'A'*100000=6500000, .CRT$XCU → 'U'*100000=8500000,
          //        .CRT$XCT00200 → 'T'*100000+200=8400200
          sec = Entry::Section::kInitArray;
          is_init = true;
          auto suffix = section_name.substr(7);  // after ".CRT$XC"
          if (!suffix.empty()) {
            priority = static_cast<int>(suffix[0]) * 100000;
            if (suffix.size() > 1) {
              int num = 0;
              suffix.substr(1).getAsInteger(10, num);
              priority += num;
            }
          }
        } else {
          // COFF .CRT$XT[suffix]: C++ destructors, same suffix-to-priority scheme.
          sec = Entry::Section::kFiniArray;
          is_init = false;
          auto suffix = section_name.substr(7);  // after ".CRT$XT"
          if (!suffix.empty()) {
            priority = static_cast<int>(suffix[0]) * 100000;
            if (suffix.size() > 1) {
              int num = 0;
              suffix.substr(1).getAsInteger(10, num);
              priority += num;
            }
          }
        }

        for (auto* Block : Sec.blocks()) {
          auto Content = Block->getContent();
          size_t PtrSize = G.getPointerSize();
          for (size_t Offset = 0; Offset + PtrSize <= Content.size(); Offset += PtrSize) {
            uint64_t FnAddr = 0;
            memcpy(&FnAddr, Content.data() + Offset, PtrSize);
            if (FnAddr != 0) {
              Entry entry{llvm::orc::ExecutorAddr(FnAddr), sec, priority};
              if (is_init) {
                session_->AddPendingInitializer(&jit_dylib, entry);
              } else {
                session_->AddPendingDeinitializer(&jit_dylib, entry);
              }
            }
          }
        }
      }
      return llvm::Error::success();
    });
  }

  llvm::Error notifyFailed(llvm::orc::MaterializationResponsibility& MR) override {
    return llvm::Error::success();
  }

  llvm::Error notifyRemovingResources(llvm::orc::JITDylib& JD, llvm::orc::ResourceKey K) override {
    return llvm::Error::success();
  }

  void notifyTransferringResources(llvm::orc::JITDylib& JD, llvm::orc::ResourceKey DstKey,
                                   llvm::orc::ResourceKey SrcKey) override {}
};

#ifdef _WIN32
/*!
 * \brief Custom definition generator for Windows DLL import symbols.
 *
 * On Windows with the MSVC ABI, COFF objects reference DLL-imported symbols
 * via __imp_XXX pointer stubs and direct calls. Without COFFPlatform (which
 * we skip due to MSVC CRT dependency issues), JITLink cannot resolve these.
 *
 * For each resolved symbol, this generator creates a JIT-allocated LinkGraph
 * containing:
 *   - __imp_XXX: a GOT-like pointer entry holding the real address
 *   - XXX: a PLT-like jump stub (`jmp [__imp_XXX]`) for direct calls
 *
 * By allocating stubs in JIT memory (rather than using absoluteSymbols at
 * host-process addresses), all PCRel32 fixups from JIT code reach safely.
 *
 * Symbol search order:
 *   1. Specific MSVC runtime DLLs (vcruntime140, ucrtbase, msvcp140)
 *   2. All loaded process modules (EnumProcessModules)
 *   3. LLVM's SearchForAddressOfSymbol
 */
class DLLImportDefinitionGenerator : public llvm::orc::DefinitionGenerator {
  llvm::orc::ExecutionSession& ES_;
  llvm::orc::ObjectLinkingLayer& L_;

  static void* FindInProcessModules(const std::string& Name) {
    // Try specific runtime DLLs first, then tvm_ffi.dll (loaded by Python),
    // then all process modules, then LLVM's search.
    static const char* kRuntimeDLLs[] = {
        "vcruntime140.dll", "vcruntime140_1.dll", "ucrtbase.dll", "msvcp140.dll",
    };
    for (const char* dll : kRuntimeDLLs) {
      if (HMODULE hMod = LoadLibraryA(dll)) {
        if (auto addr = GetProcAddress(hMod, Name.c_str())) {
          return reinterpret_cast<void*>(addr);
        }
      }
    }
    // Also check tvm_ffi.dll (host process symbol provider)
    if (HMODULE hTvmFfi = GetModuleHandleA("tvm_ffi.dll")) {
      if (auto addr = GetProcAddress(hTvmFfi, Name.c_str())) {
        return reinterpret_cast<void*>(addr);
      }
    }
    HMODULE hMods[1024];
    DWORD cbNeeded;
    if (EnumProcessModules(GetCurrentProcess(), hMods, sizeof(hMods), &cbNeeded)) {
      DWORD count = cbNeeded / sizeof(HMODULE);
      if (count > 1024) count = 1024;
      for (DWORD i = 0; i < count; ++i) {
        if (auto addr = GetProcAddress(hMods[i], Name.c_str())) {
          return reinterpret_cast<void*>(addr);
        }
      }
    }
    if (void* addr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(Name)) {
      return addr;
    }
    return nullptr;
  }

 public:
  DLLImportDefinitionGenerator(llvm::orc::ExecutionSession& ES, llvm::orc::ObjectLinkingLayer& L)
      : ES_(ES), L_(L) {}

  llvm::Error tryToGenerate(llvm::orc::LookupState& LS, llvm::orc::LookupKind K,
                            llvm::orc::JITDylib& JD,
                            llvm::orc::JITDylibLookupFlags JDLookupFlags,
                            const llvm::orc::SymbolLookupSet& LookupSet) override {
    // Step 1: Collect unique base names (strip __imp_ prefix) and resolve addresses.
    llvm::DenseMap<llvm::orc::SymbolStringPtr, llvm::orc::ExecutorAddr> Resolved;
    for (auto& [Name, Flags] : LookupSet) {
      llvm::StringRef NameStr = *Name;
      std::string BaseName =
          NameStr.starts_with("__imp_") ? NameStr.drop_front(6).str() : NameStr.str();
      if (BaseName == "__ImageBase") continue;
      auto InternedBase = ES_.intern(BaseName);
      if (Resolved.count(InternedBase)) continue;
      void* Addr = FindInProcessModules(BaseName);
      if (Addr) {
        Resolved[InternedBase] = llvm::orc::ExecutorAddr::fromPtr(Addr);
      }
    }
    if (Resolved.empty()) return llvm::Error::success();

    // Step 2: Build a LinkGraph with __imp_ pointers and PLT jump stubs.
    auto G = std::make_unique<llvm::jitlink::LinkGraph>(
        "<DLL_IMPORT_STUBS>", ES_.getSymbolStringPool(), ES_.getTargetTriple(),
        llvm::SubtargetFeatures(), llvm::jitlink::getGenericEdgeKindName);
    auto Prot = static_cast<llvm::orc::MemProt>(
        static_cast<unsigned>(llvm::orc::MemProt::Read) |
        static_cast<unsigned>(llvm::orc::MemProt::Exec));
    auto& Sec = G->createSection("__dll_stubs", Prot);

    for (auto& [InternedName, Addr] : Resolved) {
      // Absolute symbol at the real address (local to this graph)
      auto& Target = G->addAbsoluteSymbol(
          G->intern(("__real_" + *InternedName).str()), Addr, G->getPointerSize(),
          llvm::jitlink::Linkage::Strong, llvm::jitlink::Scope::Local, false);
      // __imp_XXX pointer (GOT-like entry)
      auto& Ptr = llvm::jitlink::x86_64::createAnonymousPointer(*G, Sec, &Target);
      Ptr.setName(G->intern(("__imp_" + *InternedName).str()));
      Ptr.setLinkage(llvm::jitlink::Linkage::Strong);
      Ptr.setScope(llvm::jitlink::Scope::Default);
      // XXX jump stub (PLT-like entry) for direct calls
      auto& StubBlock = llvm::jitlink::x86_64::createPointerJumpStubBlock(*G, Sec, Ptr);
      G->addDefinedSymbol(StubBlock, 0, *InternedName, StubBlock.getSize(),
                          llvm::jitlink::Linkage::Strong, llvm::jitlink::Scope::Default, true,
                          false);
    }
    return L_.add(JD, std::move(G));
  }
};
#endif  // _WIN32

ORCJITExecutionSessionObj::ORCJITExecutionSessionObj(const std::string& orc_rt_path)
    : jit_(nullptr) {
  // Helper: force JITLink's ObjectLinkingLayer on platforms where
  // the default RTDyldObjectLinkingLayer won't work.
  //
  // macOS: MachOPlatform (via ExecutorNativePlatform) requires ObjectLinkingLayer.
  //
  // Windows: LLJIT defaults to RTDyldObjectLinkingLayer for COFF x86_64
  // (see LLJIT.cpp, LLJITBuilderState::prepareForConstruction). We need
  // ObjectLinkingLayer because:
  //   1. Our InitFiniPlugin inherits ObjectLinkingLayer::Plugin — RTDyld has
  //      no plugin API, so the static_cast<ObjectLinkingLayer&> would crash.
  //   2. We skip the ORC runtime on Windows (COFFPlatform requires MSVC CRT
  //      symbols like _CxxThrowException, RTTI vtables, iostream objects that
  //      are not resolvable in the JIT), so we handle .CRT$XC*/.CRT$XT*
  //      init/fini sections ourselves via the plugin.
  //
  // Linux: LLJIT already defaults to ObjectLinkingLayer for ELF, no override needed.
  auto setup_builder = [](llvm::orc::LLJITBuilder& builder) {
#if defined(__APPLE__) || defined(_WIN32)
    builder.setObjectLinkingLayerCreator(
        [](llvm::orc::ExecutionSession& ES)
            -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
          return std::make_unique<llvm::orc::ObjectLinkingLayer>(ES);
        });
#endif
#ifdef _WIN32
    // Override ProcessSymbols setup to NOT add the default
    // EPCDynamicLibrarySearchGenerator. That generator resolves symbols to
    // absolute host-process addresses, which causes PCRel32 overflow when
    // JIT code calls into DLLs >2GB away. Our DLLImportDefinitionGenerator
    // (added after construction) wraps every resolved address in a
    // JIT-allocated PLT stub, keeping all fixups in range.
    builder.setProcessSymbolsJITDylibSetup(
        [](llvm::orc::LLJIT& J) -> llvm::Expected<llvm::orc::JITDylibSP> {
          return &J.getExecutionSession().createBareJITDylib("<Process Symbols>");
        });
#endif
    (void)builder;
  };

  if (!orc_rt_path.empty()) {
    auto builder = llvm::orc::LLJITBuilder();
    builder.setPlatformSetUp(llvm::orc::ExecutorNativePlatform(orc_rt_path));
    setup_builder(builder);
    jit_ = std::move(
        call_llvm(builder.create(), "Failed to create LLJIT with ORC runtime"));
  } else {
    auto builder = llvm::orc::LLJITBuilder();
    setup_builder(builder);
    jit_ = std::move(call_llvm(builder.create(), "Failed to create LLJIT"));
  }
#ifdef _WIN32
  // Strip .pdata/.xdata relocations from COFF objects before JITLink graph building.
  // clang-cl puts static functions in COMDAT sections, and .pdata SEH unwind data
  // has relocations targeting COMDAT leader symbols. JITLink's COFFLinkGraphBuilder
  // doesn't register COMDAT leaders in its symbol table when the second COMDAT symbol
  // is CLASS_STATIC (not CLASS_EXTERNAL), causing "Could not find symbol" errors.
  // We already strip .pdata/.xdata edges in a PostAllocationPass; this moves the
  // stripping earlier to prevent the graph builder error.
  // Strip .pdata/.xdata relocations using raw COFF binary manipulation.
  // We avoid llvm/Object/COFF.h because windows.h (included transitively by
  // LLJIT.h) defines IMAGE_* macros that conflict with LLVM's COFF enums.
  jit_->getObjTransformLayer().setTransform(
      [](std::unique_ptr<llvm::MemoryBuffer> Buf)
          -> llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> {
        const char* Data = Buf->getBufferStart();
        size_t Size = Buf->getBufferSize();
        if (Size < 20) return std::move(Buf);

        // Parse COFF header (regular or bigobj format)
        uint16_t w0, w1;
        std::memcpy(&w0, Data, 2);
        std::memcpy(&w1, Data + 2, 2);
        bool bigobj = (w0 == 0 && w1 == 0xFFFF);

        uint16_t machine;
        uint32_t num_sections, ptr_to_symtab, num_symbols;
        size_t sec_hdr_start, sym_entry_size;
        if (bigobj) {
          if (Size < 56) return std::move(Buf);
          std::memcpy(&machine, Data + 6, 2);
          std::memcpy(&num_sections, Data + 44, 4);
          std::memcpy(&ptr_to_symtab, Data + 48, 4);
          std::memcpy(&num_symbols, Data + 52, 4);
          sec_hdr_start = 56;
          sym_entry_size = 20;
        } else {
          machine = w0;
          uint16_t ns, opt_hdr_size;
          std::memcpy(&ns, Data + 2, 2);
          std::memcpy(&opt_hdr_size, Data + 16, 2);
          std::memcpy(&ptr_to_symtab, Data + 8, 4);
          std::memcpy(&num_symbols, Data + 12, 4);
          num_sections = ns;
          sec_hdr_start = 20 + opt_hdr_size;
          sym_entry_size = 18;
        }
        if (machine != 0x8664) return std::move(Buf);

        // String table follows the symbol table
        size_t strtab_start =
            ptr_to_symtab + static_cast<size_t>(num_symbols) * sym_entry_size;

        // Resolve a section name (inline 8-byte or "/offset" string table ref)
        constexpr size_t kSecHdrSize = 40;
        auto resolve_name = [&](size_t hdr_off) -> llvm::StringRef {
          const char* raw = Data + hdr_off;
          if (raw[0] == '/' && raw[1] >= '0' && raw[1] <= '9') {
            uint32_t offset = 0;
            for (int j = 1; j < 8 && raw[j] >= '0' && raw[j] <= '9'; ++j)
              offset = offset * 10 + (raw[j] - '0');
            size_t pos = strtab_start + offset;
            if (pos < Size) {
              size_t len = 0;
              while (pos + len < Size && Data[pos + len]) ++len;
              return {Data + pos, len};
            }
          }
          size_t len = 0;
          while (len < 8 && raw[len]) ++len;
          return {raw, len};
        };

        // Collect section header offsets needing relocation stripping
        llvm::SmallVector<size_t, 8> strip_offsets;
        for (uint32_t i = 0; i < num_sections; ++i) {
          size_t off = sec_hdr_start + i * kSecHdrSize;
          if (off + kSecHdrSize > Size) break;
          auto name = resolve_name(off);
          if (name.starts_with(".pdata") || name.starts_with(".xdata")) {
            uint16_t num_relocs;
            std::memcpy(&num_relocs, Data + off + 32, 2);
            if (num_relocs > 0) strip_offsets.push_back(off);
          }
        }
        if (strip_offsets.empty()) return std::move(Buf);

        // Create mutable copy, zero out PointerToRelocations and NumberOfRelocations
        llvm::SmallVector<char> MutableBuf(Data, Data + Size);
        for (auto off : strip_offsets) {
          std::memset(&MutableBuf[off + 24], 0, 4);  // PointerToRelocations
          std::memset(&MutableBuf[off + 32], 0, 2);  // NumberOfRelocations
        }
        return llvm::MemoryBuffer::getMemBufferCopy(
            llvm::StringRef(MutableBuf.data(), MutableBuf.size()),
            Buf->getBufferIdentifier());
      });
#endif
#if defined(__linux__) || defined(_WIN32)
  // Linux/Windows: use our custom InitFiniPlugin for init/fini section collection
  // and priority-ordered execution.
  // macOS: MachOPlatform handles init/fini natively via jit_->initialize()/deinitialize().
  auto& objlayer = jit_->getObjLinkingLayer();
  static_cast<llvm::orc::ObjectLinkingLayer&>(objlayer).addPlugin(
      std::make_unique<InitFiniPlugin>(GetRef<ORCJITExecutionSession>(this)));
#endif
#ifdef _WIN32
  // On Windows, the default process-symbol generator only searches the main
  // exe module via GetProcAddress(GetModuleHandle(NULL), ...). Add a
  // comprehensive generator that searches all loaded DLLs (vcruntime140,
  // ucrtbase, tvm_ffi, etc.) and creates __imp_* pointer stubs.
  if (auto PSG = jit_->getProcessSymbolsJITDylib()) {
    auto& ObjLayer =
        static_cast<llvm::orc::ObjectLinkingLayer&>(jit_->getObjLinkingLayer());
    PSG->addGenerator(std::make_unique<DLLImportDefinitionGenerator>(
        jit_->getExecutionSession(), ObjLayer));
  }
#endif
}

ORCJITExecutionSession::ORCJITExecutionSession(const std::string& orc_rt_path) {
  ObjectPtr<ORCJITExecutionSessionObj> obj = make_object<ORCJITExecutionSessionObj>(orc_rt_path);
  data_ = std::move(obj);
}

ORCJITDynamicLibrary ORCJITExecutionSessionObj::CreateDynamicLibrary(const String& name) {
  TVM_FFI_CHECK(jit_ != nullptr, InternalError) << "ExecutionSession not initialized";

  // Generate name if not provided
  String lib_name = name;
  if (lib_name.empty()) {
    std::ostringstream oss;
    oss << "dylib_" << dylib_counter_++;
    lib_name = oss.str();
  }

  llvm::orc::JITDylib& jit_dylib =
      call_llvm(jit_->getExecutionSession().createJITDylib(lib_name.c_str()));
  // Use the LLJIT's default link order (Main → Platform → ProcessSymbols).
  // This provides host process symbols via the ProcessSymbols JITDylib's generator,
  // while ensuring the platform's __cxa_atexit interposer (in PlatformJD) takes
  // precedence — so __cxa_atexit handlers are managed by the platform and can be
  // drained per-JITDylib via __lljit_run_atexits at teardown.
  for (auto& kv : jit_->defaultLinkOrder()) {
    jit_dylib.addToLinkOrder(*kv.first, kv.second);
  }

  auto dylib = ORCJITDynamicLibrary(make_object<ORCJITDynamicLibraryObj>(
      GetRef<ORCJITExecutionSession>(this), &jit_dylib, jit_.get(), lib_name));

  return dylib;
}

llvm::orc::ExecutionSession& ORCJITExecutionSessionObj::GetLLVMExecutionSession() {
  TVM_FFI_CHECK(jit_ != nullptr, InternalError) << "ExecutionSession not initialized";
  return jit_->getExecutionSession();
}

llvm::orc::LLJIT& ORCJITExecutionSessionObj::GetLLJIT() {
  TVM_FFI_CHECK(jit_ != nullptr, InternalError) << "ExecutionSession not initialized";
  return *jit_;
}

using CtorDtor = void (*)();

void ORCJITExecutionSessionObj::RunPendingInitializers(llvm::orc::JITDylib& jit_dylib) {
  auto it = pending_initializers_.find(&jit_dylib);
  if (it != pending_initializers_.end()) {
    llvm::sort(it->second, [](const InitFiniEntry& a, const InitFiniEntry& b) {
      if (a.section != b.section) return static_cast<int>(a.section) < static_cast<int>(b.section);
      return a.priority < b.priority;
    });
    for (const auto& entry : it->second) {
      entry.address.toPtr<CtorDtor>()();
    }
    pending_initializers_.erase(it);
  }
}

void ORCJITExecutionSessionObj::RunPendingDeinitializers(llvm::orc::JITDylib& jit_dylib) {
  auto it = pending_deinitializers_.find(&jit_dylib);
  if (it != pending_deinitializers_.end()) {
    llvm::sort(it->second, [](const InitFiniEntry& a, const InitFiniEntry& b) {
      if (a.section != b.section) return static_cast<int>(a.section) < static_cast<int>(b.section);
      return a.priority < b.priority;
    });
    for (const auto& entry : it->second) {
      entry.address.toPtr<CtorDtor>()();
    }
    pending_deinitializers_.erase(it);
  }
}

void ORCJITExecutionSessionObj::AddPendingInitializer(llvm::orc::JITDylib* jit_dylib,
                                                      const InitFiniEntry& entry) {
  pending_initializers_[jit_dylib].push_back(entry);
}

void ORCJITExecutionSessionObj::AddPendingDeinitializer(llvm::orc::JITDylib* jit_dylib,
                                                        const InitFiniEntry& entry) {
  pending_deinitializers_[jit_dylib].push_back(entry);
}

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm
