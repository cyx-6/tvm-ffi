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
 * \file orcjit_arena_mm.cc
 * \brief Arena-based JITLinkMemoryManager implementation.
 *
 * Follows the InProcessMemoryManager pattern from LLVM but replaces
 * per-object mmap with bump allocation from a pre-reserved arena.
 */
#include "orcjit_arena_mm.h"

#include <llvm/ExecutionEngine/JITLink/JITLink.h>
#include <llvm/ExecutionEngine/Orc/Shared/AllocationActions.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Memory.h>

#include <algorithm>
#include <cstring>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace tvm {
namespace ffi {
namespace orcjit {

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::orc;

// ── Platform abstraction ────────────────────────────────────────────

#ifdef _WIN32

void* ArenaJITLinkMemoryManager::reserveVA(size_t size) {
  void* p = VirtualAlloc(NULL, size, MEM_RESERVE, PAGE_NOACCESS);
  return p;
}

void ArenaJITLinkMemoryManager::releaseVA(void* addr, size_t size) {
  VirtualFree(addr, 0, MEM_RELEASE);
}

void ArenaJITLinkMemoryManager::commitPages(void* addr, size_t size) {
  VirtualAlloc(addr, size, MEM_COMMIT, PAGE_READWRITE);
}

void ArenaJITLinkMemoryManager::decommitPages(void* addr, size_t size) {
  VirtualFree(addr, size, MEM_DECOMMIT);
}

void ArenaJITLinkMemoryManager::protectPages(void* addr, size_t size, MemProt Prot) {
  DWORD prot = PAGE_NOACCESS;
  bool has_exec = (Prot & MemProt::Exec) != MemProt::None;
  bool has_write = (Prot & MemProt::Write) != MemProt::None;
  bool has_read = (Prot & MemProt::Read) != MemProt::None;
  if (has_exec && has_write)
    prot = PAGE_EXECUTE_READWRITE;
  else if (has_exec && has_read)
    prot = PAGE_EXECUTE_READ;
  else if (has_exec)
    prot = PAGE_EXECUTE;
  else if (has_write)
    prot = PAGE_READWRITE;
  else if (has_read)
    prot = PAGE_READONLY;
  DWORD old;
  VirtualProtect(addr, size, prot, &old);
  if (has_exec) FlushInstructionCache(GetCurrentProcess(), addr, size);
}

#else  // Unix (Linux, macOS)

void* ArenaJITLinkMemoryManager::reserveVA(size_t size) {
  void* p = ::mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) return nullptr;
  return p;
}

void ArenaJITLinkMemoryManager::releaseVA(void* addr, size_t size) { ::munmap(addr, size); }

void ArenaJITLinkMemoryManager::commitPages(void* addr, size_t size) {
  ::mprotect(addr, size, PROT_READ | PROT_WRITE);
}

void ArenaJITLinkMemoryManager::decommitPages(void* addr, size_t size) {
  // Intentionally a no-op for arena pages.  The ORC runtime may still reference
  // deallocated JIT memory during session teardown (e.g., ELFNixPlatform
  // deinitializers run after some allocations are freed).  Decommitting
  // (MADV_DONTNEED or mprotect PROT_NONE) would cause segfaults or illegal
  // instructions during shutdown.
  //
  // Physical pages stay committed but are returned to the free list for reuse.
  // The arena destructor releases all VA and physical memory via munmap.
  (void)addr;
  (void)size;
}

void ArenaJITLinkMemoryManager::protectPages(void* addr, size_t size, MemProt Prot) {
  int prot = PROT_NONE;
  if ((Prot & MemProt::Read) != MemProt::None) prot |= PROT_READ;
  if ((Prot & MemProt::Write) != MemProt::None) prot |= PROT_WRITE;
  if ((Prot & MemProt::Exec) != MemProt::None) prot |= PROT_EXEC;
  ::mprotect(addr, size, prot);
  if ((Prot & MemProt::Exec) != MemProt::None) {
    sys::Memory::InvalidateInstructionCache(addr, size);
  }
}

#endif  // _WIN32

// ── ArenaInFlightAlloc ──────────────────────────────────────────────

class ArenaJITLinkMemoryManager::ArenaInFlightAlloc
    : public JITLinkMemoryManager::InFlightAlloc {
 public:
  ArenaInFlightAlloc(ArenaJITLinkMemoryManager& MM, LinkGraph& G, BasicLayout BL,
                     size_t arena_offset, size_t standard_size, size_t finalize_size)
      : MM(MM),
        G(&G),
        BL(std::move(BL)),
        arena_offset_(arena_offset),
        standard_size_(standard_size),
        finalize_size_(finalize_size) {}

  ~ArenaInFlightAlloc() override {
    assert(!G && "ArenaInFlightAlloc destroyed without finalize or abandon");
  }

  void finalize(OnFinalizedFunction OnFinalized) override {
    // Apply target protections for each segment.
    if (auto Err = applyProtections()) {
      OnFinalized(std::move(Err));
      return;
    }

    // Run finalization actions (e.g., register EH frames).
    auto DeallocActions = shared::runFinalizeActions(BL.graphAllocActions());
    if (!DeallocActions) {
      OnFinalized(DeallocActions.takeError());
      return;
    }

    // Decommit finalize-lifetime pages — they're no longer needed.
    if (finalize_size_ > 0) {
      MM.decommitPages(MM.arena_base_ + arena_offset_ + standard_size_, finalize_size_);
      MM.freeRegion(arena_offset_ + standard_size_, finalize_size_);
    }

#ifndef NDEBUG
    G = nullptr;
#endif

    // Create finalized allocation handle.
    auto* FA = new FinalizedAllocInfo{arena_offset_, standard_size_, std::move(*DeallocActions)};
    OnFinalized(FinalizedAlloc(ExecutorAddr::fromPtr(FA)));
  }

  void abandon(OnAbandonedFunction OnAbandoned) override {
    // Decommit and return entire allocation to free list.
    size_t total = standard_size_ + finalize_size_;
    if (total > 0) {
      MM.decommitPages(MM.arena_base_ + arena_offset_, total);
      MM.freeRegion(arena_offset_, total);
    }

#ifndef NDEBUG
    G = nullptr;
#endif

    OnAbandoned(Error::success());
  }

 private:
  Error applyProtections() {
    for (auto& KV : BL.segments()) {
      const auto& AG = KV.first;
      auto& Seg = KV.second;

      uint64_t SegSize = alignTo(Seg.ContentSize + Seg.ZeroFillSize, MM.page_size_);
      MM.protectPages(Seg.WorkingMem, SegSize, AG.getMemProt());
    }
    return Error::success();
  }

  ArenaJITLinkMemoryManager& MM;
  LinkGraph* G;
  BasicLayout BL;
  size_t arena_offset_;
  size_t standard_size_;
  size_t finalize_size_;
};

// ── ArenaJITLinkMemoryManager ───────────────────────────────────────

ArenaJITLinkMemoryManager::ArenaJITLinkMemoryManager(uint64_t page_size, size_t arena_capacity)
    : arena_base_(nullptr), arena_capacity_(arena_capacity), page_size_(page_size), bump_offset_(0) {
  arena_base_ = static_cast<char*>(reserveVA(arena_capacity_));
  if (!arena_base_) {
    // Fatal: cannot reserve arena. This should not happen on 64-bit systems.
    llvm::report_fatal_error("ArenaJITLinkMemoryManager: failed to reserve " +
                             Twine(arena_capacity_ / (1024 * 1024)) + " MB of virtual address space");
  }
}

ArenaJITLinkMemoryManager::~ArenaJITLinkMemoryManager() {
  if (arena_base_) {
    releaseVA(arena_base_, arena_capacity_);
  }
}

Expected<size_t> ArenaJITLinkMemoryManager::bumpAllocate(size_t size) {
  std::lock_guard<std::mutex> Lock(mu_);

  // Try free list first (best-fit).
  size_t best_idx = free_list_.size();
  size_t best_waste = std::numeric_limits<size_t>::max();
  for (size_t i = 0; i < free_list_.size(); ++i) {
    if (free_list_[i].size >= size && free_list_[i].size - size < best_waste) {
      best_idx = i;
      best_waste = free_list_[i].size - size;
      if (best_waste == 0) break;
    }
  }

  if (best_idx < free_list_.size()) {
    size_t offset = free_list_[best_idx].offset;
    if (free_list_[best_idx].size == size) {
      free_list_.erase(free_list_.begin() + best_idx);
    } else {
      free_list_[best_idx].offset += size;
      free_list_[best_idx].size -= size;
    }
    return offset;
  }

  // Bump allocate.
  if (bump_offset_ + size > arena_capacity_) {
    return make_error<StringError>(
        "ArenaJITLinkMemoryManager: arena exhausted (used " +
            formatv("{0:x}", bump_offset_) + " + requested " + formatv("{0:x}", size) +
            " > capacity " + formatv("{0:x}", arena_capacity_) + ")",
        inconvertibleErrorCode());
  }

  size_t offset = bump_offset_;
  bump_offset_ += size;
  return offset;
}

void ArenaJITLinkMemoryManager::freeRegion(size_t offset, size_t size) {
  if (size == 0) return;
  std::lock_guard<std::mutex> Lock(mu_);

  // Insert into free list in sorted order.
  auto it = std::lower_bound(free_list_.begin(), free_list_.end(), offset,
                              [](const FreeBlock& fb, size_t off) { return fb.offset < off; });
  it = free_list_.insert(it, FreeBlock{offset, size});

  // Coalesce with next.
  auto next = it + 1;
  if (next != free_list_.end() && it->offset + it->size == next->offset) {
    it->size += next->size;
    free_list_.erase(next);
  }

  // Coalesce with previous.
  if (it != free_list_.begin()) {
    auto prev = it - 1;
    if (prev->offset + prev->size == it->offset) {
      prev->size += it->size;
      free_list_.erase(it);
    }
  }
}

void ArenaJITLinkMemoryManager::allocate(const JITLinkDylib* JD, LinkGraph& G,
                                         OnAllocatedFunction OnAllocated) {
  BasicLayout BL(G);

  // Compute total sizes grouped by lifetime.
  auto SegsSizes = BL.getContiguousPageBasedLayoutSizes(page_size_);
  if (!SegsSizes) {
    OnAllocated(SegsSizes.takeError());
    return;
  }

  if (SegsSizes->total() > std::numeric_limits<size_t>::max()) {
    OnAllocated(make_error<llvm::jitlink::JITLinkError>(
        "Total requested size " + formatv("{0:x}", SegsSizes->total()) + " for graph " +
        G.getName() + " exceeds address space"));
    return;
  }

  auto TotalSize = static_cast<size_t>(SegsSizes->total());
  if (TotalSize == 0) {
    // Empty graph — return a no-op allocation.
    OnAllocated(std::make_unique<ArenaInFlightAlloc>(*this, G, std::move(BL), 0, 0, 0));
    return;
  }

  // Bump-allocate from arena.
  auto OffsetOrErr = bumpAllocate(TotalSize);
  if (!OffsetOrErr) {
    OnAllocated(OffsetOrErr.takeError());
    return;
  }
  size_t ArenaOffset = *OffsetOrErr;

  // Commit pages as read-write.
  commitPages(arena_base_ + ArenaOffset, TotalSize);

  // Zero-fill the region.
  std::memset(arena_base_ + ArenaOffset, 0, TotalSize);

  // Assign addresses to segments, partitioned by lifetime.
  auto StandardSegsSize = static_cast<size_t>(SegsSizes->StandardSegs);
  auto FinalizeSegsSize = static_cast<size_t>(SegsSizes->FinalizeSegs);

  auto NextStandardSegAddr = ExecutorAddr::fromPtr(arena_base_ + ArenaOffset);
  auto NextFinalizeSegAddr = ExecutorAddr::fromPtr(arena_base_ + ArenaOffset + StandardSegsSize);

  for (auto& KV : BL.segments()) {
    auto& AG = KV.first;
    auto& Seg = KV.second;

    auto& SegAddr = (AG.getMemLifetime() == MemLifetime::Standard) ? NextStandardSegAddr
                                                                   : NextFinalizeSegAddr;

    Seg.WorkingMem = SegAddr.toPtr<char*>();
    Seg.Addr = SegAddr;

    auto SegSize = alignTo(Seg.ContentSize + Seg.ZeroFillSize, page_size_);
    SegAddr += SegSize;
  }

  // Apply layout — copies content and assigns block addresses.
  if (auto Err = BL.apply()) {
    // On error: decommit and free the arena region.
    decommitPages(arena_base_ + ArenaOffset, TotalSize);
    freeRegion(ArenaOffset, TotalSize);
    OnAllocated(std::move(Err));
    return;
  }

  OnAllocated(std::make_unique<ArenaInFlightAlloc>(*this, G, std::move(BL), ArenaOffset,
                                                   StandardSegsSize, FinalizeSegsSize));
}

void ArenaJITLinkMemoryManager::deallocate(std::vector<FinalizedAlloc> Allocs,
                                           OnDeallocatedFunction OnDeallocated) {
  Error DeallocErr = Error::success();

  for (auto& Alloc : Allocs) {
    auto* FA = Alloc.release().toPtr<FinalizedAllocInfo*>();

    // Run deallocation actions in reverse order.
    while (!FA->DeallocActions.empty()) {
      if (auto Err = FA->DeallocActions.back().runWithSPSRetErrorMerged())
        DeallocErr = joinErrors(std::move(DeallocErr), std::move(Err));
      FA->DeallocActions.pop_back();
    }

    // Decommit and free arena region.
    if (FA->arena_size > 0) {
      decommitPages(arena_base_ + FA->arena_offset, FA->arena_size);
      freeRegion(FA->arena_offset, FA->arena_size);
    }

    delete FA;
  }

  OnDeallocated(std::move(DeallocErr));
}

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm
