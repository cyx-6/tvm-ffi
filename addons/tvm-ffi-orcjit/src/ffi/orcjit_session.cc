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

#include <llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstddef>

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
            for (auto* Sym : G.defined_symbols()) {
              if (&Sym->getBlock() == Block) {
                Sym->setLive(true);
              }
            }
            for (auto& Edge : Block->edges()) {
              Edge.getTarget().setLive(true);
            }
          }
        }
      }
      return llvm::Error::success();
    });
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
        if (is_init_array) {
          if (section_name.consume_front(".init_array.")) {
            section_name.getAsInteger(10, priority);
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

ORCJITExecutionSessionObj::ORCJITExecutionSessionObj(const std::string& orc_rt_path)
    : jit_(nullptr) {
  if (!orc_rt_path.empty()) {
    jit_ = std::move(call_llvm(llvm::orc::LLJITBuilder()
                                   .setPlatformSetUp(llvm::orc::ExecutorNativePlatform(orc_rt_path))
                                   .create(),
                               "Failed to create LLJIT with ORC runtime"));
  } else {
    jit_ = std::move(call_llvm(llvm::orc::LLJITBuilder().create(), "Failed to create LLJIT"));
  }
  auto& objlayer = jit_->getObjLinkingLayer();
  static_cast<llvm::orc::ObjectLinkingLayer&>(objlayer).addPlugin(
      std::make_unique<InitFiniPlugin>(GetRef<ORCJITExecutionSession>(this)));
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
