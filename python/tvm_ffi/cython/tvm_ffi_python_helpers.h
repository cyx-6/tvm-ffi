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
/*
 * \file tvm_ffi_python_helpers.h
 * \brief C++ based helpers for the Python FFI call to optimize performance.
 */
#ifndef TVM_FFI_PYTHON_HELPERS_H_
#define TVM_FFI_PYTHON_HELPERS_H_

#include <Python.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/memory.h>

// Define here to avoid dependencies on non-c headers for now
#ifndef TVM_FFI_INLINE
#if defined(_MSC_VER)
#define TVM_FFI_INLINE [[msvc::forceinline]] inline
#else
#define TVM_FFI_INLINE [[gnu::always_inline]] inline
#endif
#endif

// Local mirror of TVM_FFI_COLD_CODE / TVM_FFI_PREDICT_* from
// <tvm/ffi/base_details.h>. The Cython helper deliberately avoids that header
// (keeps the include surface c-headers-only), so we duplicate the macro
// definitions here. Keep these in sync with base_details.h: same expansion on
// GCC/Clang, no-op on MSVC.
#ifndef TVM_FFI_COLD_CODE
#if defined(__GNUC__) || defined(__clang__)
#define TVM_FFI_COLD_CODE [[gnu::cold]]
#else
#define TVM_FFI_COLD_CODE
#endif
#endif

#ifndef TVM_FFI_PREDICT_FALSE
#if defined(__GNUC__) || defined(__clang__)
#define TVM_FFI_PREDICT_FALSE(cond) (__builtin_expect(static_cast<bool>(cond), 0))
#define TVM_FFI_PREDICT_TRUE(cond) (__builtin_expect(static_cast<bool>(cond), 1))
#else
#define TVM_FFI_PREDICT_FALSE(cond) (cond)
#define TVM_FFI_PREDICT_TRUE(cond) (cond)
#endif
#endif

// Managed-dict (`__slots__ = ("__dict__",)` without an explicit dictoffset)
// is a CPython 3.11+ feature. On 3.9/3.10 such types instead use a regular
// ``tp_dictoffset != 0``, which the inactive-eligibility check catches anyway,
// so defining the flag as 0 here yields the correct (no-op) behavior.
#ifndef Py_TPFLAGS_MANAGED_DICT
#define Py_TPFLAGS_MANAGED_DICT 0
#endif

#include <atomic>
#include <cassert>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

///--------------------------------------------------------------------------------
/// We deliberately designed the data structure and function to be C-style
//  prefixed with TVMFFIPy so they can be easily invoked through Cython.
///--------------------------------------------------------------------------------

//================================================================================
// PyObject-tying state machine.
//
// Ties one Python wrapper to one C++ chandle so that
//   - ``a.x is a.x`` while the wrapper is live;
//   - ``id(a.x)`` is stable across drop+refetch (when other C++ holders keep
//     the chandle alive);
//   - ``f(x) is x`` whenever an FFI function returns a chandle that already
//     has a canonical wrapper.
//
// Layout
// ------
// Every Object allocated through the registered Python allocator
// (`TVMFFIPyAllocate`) is preceded by a 16-byte ``PyCustomAllocHeader``:
//
//   malloc start
//   +-------------------+--------------------------+--------+
//   |   tagged_pyobj    | TVMFFIObjectAllocHeader  |   T    |
//   |   (offset 0..8)   |   delete_space (8..16)   |        |
//   +-------------------+--------------------------+--------+
//                                                  ^ ptr = malloc + 16
//
// ``tagged_pyobj`` is a tagged pointer to the canonical Python wrapper. The
// wrapper is >= 16-aligned, so the low 4 bits are free; two encode the state
// (see below) without growing the header past its fixed 16 bytes.
//
// States
// ------
// Bit 0 (Inactive) and bit 1 (InTransit) tag ``tagged_pyobj`` into four states:
//   Detached:  ``tagged_pyobj == NULL`` -- no wrapper bound to this chandle.
//   Active:    ``ptr, bits == 00`` -- the live canonical wrapper.
//   Inactive:  ``ptr | Inactive`` -- dead, untracked allocation cached for
//              address-stable revival (settled).
//   InTransit: ``ptr | Inactive | InTransit`` -- an Inactive allocation whose
//              dealloc handshake is still in flight (see below). The Inactive
//              bit stays set, so ``TVMFFIPyTagIsInactive`` matches it too.
// The InTransit bit is a general-purpose "transition in flight" marker; in this
// revision it is load-bearing only in the dealloc handshake.
//
// Invariants
// ----------
//   I1. When a PyObject goes out of scope (no Python var refers to it), its
//       +1 on chandle is always released (in ``__dealloc__`` ->
//       ``TVMFFIPyTpDealloc``).
//   I2. When a chandle is destroyed, its cached allocation (if any) is
//       reclaimed.
//   I3'. ``wrapper.chandle`` is only ever a real C++ object pointer or NULL,
//       never a sentinel. A non-NULL chandle owns +1, except inside the
//       wrapper's own dealloc window (where it is kept only as a header locator).
//   I4. Every ``PyObject*`` the Cython side passes to a helper here is a live
//       wrapper (tag bits 0); only this header sets or clears the tag bits.
//
// The dealloc handshake
// ---------------------
// When an eligible wrapper deallocs, its allocation is *cached Inactive* if the
// chandle outlives it, or *genuinely freed* if the wrapper held the last ref.
// ``TVMFFIPyTpDealloc`` cannot read the refcount to decide (an FFI ``DecRef`` may
// race it with the GIL released), so it pre-tags ``Inactive | InTransit`` and
// ``DecRef``s unconditionally. The InTransit bit, read by ``tp_free`` just after,
// records whether the chandle's Weak deleter (``TVMFFIPyDeleteSpace``) fired:
// still set => C++ alive, keep cached; cleared by the deleter => C++ dead, free.
//
// Where transitions happen
// ------------------------
// ``TVMFFIPyMakeRetObject`` (this header), behind ``make_ret_object``
// (object.pxi) -- owns the whole return-object transition in one frame:
//     Detached/Active/Inactive -> Active : fresh / cached / revived-in-place.
//
// ``TVMFFIPyTpDealloc`` (CObject.__dealloc__) -- runs when the wrapper's
// refcount hits 0, before the free:
//     Active   -> Inactive : eligible; tag Inactive | InTransit, DecRef (the
//                            handshake; ``tp_free`` / ``TVMFFIPyDeleteSpace``
//                            settle it).
//     Active   -> Detached : type not eligible; detach first, then DecRef.
//
// ``TVMFFIPyArgSetterObjectRValueRef_`` (function.pxi),
// ``__move_handle_from__`` (object.pxi):
//     Active   -> Detached : detach the binding before a move nulls the
//                            source chandle.
//
// ``TVMFFIPyDeleteSpace`` (Weak deleter) -- the chandle's weak count hit 0:
//     Inactive|InTransit   : in-flight dealloc; defer both frees to ``tp_free``.
//     Inactive (settled)   : reclaim the cached wrapper and free the C++ block.
//
// Slot install
// ------------
// ``tp_alloc`` / ``tp_free`` are NOT inherited by dynamic subtypes (CPython
// resets them per dynamic subtype), so each registered type needs its own
// install. ``_update_registry`` (object.pxi) -- the choke point every
// registered FFI type funnels through -- calls ``TVMFFIPyInstallTypeSlots``
// there, once per type.
//
// Shutdown guard
// --------------
// ``TVMFFIPyMarkPythonFinalizing`` is wired to atexit from Cython module
// init. After it fires, inactive cached allocations on still-live chandles are
// intentionally leaked (process exiting; OS reclaims) rather than reaching
// for ``PyGILState_Ensure`` on a teardown interpreter.
//
// Free-threaded builds (``Py_GIL_DISABLED``)
// ------------------------------------------
// Without the GIL the bare ``tagged_pyobj`` reads/writes above race -- the
// Active-hit borrowed read in particular is a use-after-free (``make_ret``
// reads the wrapper, then a concurrent ``__dealloc__`` frees it before the
// IncRef). The feature stays enabled on free-threaded builds; the word becomes a
// tiny spin-lock + claim marker that serializes every transition. The mechanics
// (all ``#ifdef Py_GIL_DISABLED``; the GIL build is unchanged):
//
//  * One extra tag bit: Locked (bit 2) is the spin-lock. ``TVMFFIPyLockWord`` /
//    ``TVMFFIPyUnlockWord`` acquire/release it via ``__atomic_*`` CAS / store. An
//    alloc/revive that must run with the lock released publishes the InTransit bit
//    (the same "transition in flight" marker the dealloc handshake uses) so peers
//    yield; the two producers never collide on one word (a make_ret claim holds a
//    strong chandle ref, which gates off the tp_free/delete_space that would set or
//    read InTransit for a dealloc on the same wrapper).
//  * The lock is held only across short, *park-free* word/header edits -- never an
//    allocation, DecRef, GC op, or blocking call -- so a lock holder can never be
//    frozen by the cyclic GC's stop-the-world, and spinners make bounded progress.
//    Every wait (``TVMFFIPyLockYield``) detaches the thread state, the same
//    cooperation PyMutex performs, so a concurrent stop-the-world is never starved.
//  * Active-hit revival uses ``PyUnstable_TryIncRef`` (inc-if-nonzero): it succeeds
//    iff the wrapper is not being collected, which closes the borrowed-read UAF.
//    ``PyUnstable_EnableTryIncRef`` is armed at every Active publish (in
//    ``make_ret``'s alloc/revive and ``TVMFFIPyRebindPyObject``), before the
//    release-store, so a racing reader never TryIncRef's an un-armed wrapper.
//  * Memory safety of the Active hit rests on: ``make_ret`` holds a strong ref on
//    the chandle, which pins the C++ block; that in turn gates off the only paths
//    that free a cached wrapper's memory (``TVMFFIPyDeleteSpace`` needs weak->0;
//    ``tp_free`` under InTransit defers), so a wrapper read from the word is always
//    mapped and ``TryIncRef`` -- never a plain ``Py_INCREF`` on FT -- is safe.
//  * ``tp_dealloc`` has priority: ``make_ret`` yields on InTransit and revives only
//    a clean Inactive, so a settling dealloc is never overtaken.
//================================================================================

/*!
 * \brief Python-side derived header. ``base.delete_space`` sits at
 *        ``ptr - sizeof(TVMFFIObjectAllocHeader)`` so the generic C++
 *        deleter (which knows nothing about Python) can find it.
 */
struct PyCustomAllocHeader {
  PyObject* tagged_pyobj;
  TVMFFIObjectAllocHeader base;
};

static_assert(sizeof(PyCustomAllocHeader) == 16,
              "header must be 16 bytes so T at ptr = malloc + 16 is naturally "
              "aligned for alignof(T) up to alignof(max_align_t)");
static_assert(offsetof(PyCustomAllocHeader, base) ==
                  sizeof(PyCustomAllocHeader) - sizeof(TVMFFIObjectAllocHeader),
              "base must sit at ptr - sizeof(TVMFFIObjectAllocHeader) for the "
              "C++ deleter to find it");

TVM_FFI_INLINE PyCustomAllocHeader* TVMFFIPyHeader(void* ptr) {
  return reinterpret_cast<PyCustomAllocHeader*>(static_cast<char*>(ptr) -
                                                sizeof(PyCustomAllocHeader));
}

// Low-bit tags packed into ``tagged_pyobj`` (wrappers are >= 16-aligned, so the
// low 4 bits are free):
//   bit 0 (Inactive):  the bound wrapper is dead, its memory cached for revival.
//   bit 1 (InTransit): a transition on this binding is in flight. Two producers set
//                      it, and (by the invariant below) never concurrently on one
//                      word: the dealloc handshake (Active -> Inactive|InTransit,
//                      settled by tp_free/delete_space) and, on free-threaded builds,
//                      a make_ret alloc/revive holding the claim with the lock
//                      released (Inactive|InTransit over a revive, bare InTransit over
//                      a fresh alloc).
// On free-threaded builds one more bit turns the word into a spin-lock (see
// "Free-threaded builds" below); the GIL build never sets it:
//   bit 2 (Locked):  a thread holds the word's spin-lock (short, park-free section).
//   bit 3:           reserved / free for future use.
// ``TVMFFIPyRemoveTag`` masks every defined bit to recover the real wrapper pointer.
constexpr uintptr_t kPyCachedInactiveTagBit = 1;
constexpr uintptr_t kPyInTransitTagBit = 2;
#ifdef Py_GIL_DISABLED
constexpr uintptr_t kPyLockedTagBit = 4;
constexpr uintptr_t kPyTagBitMask = kPyCachedInactiveTagBit | kPyInTransitTagBit | kPyLockedTagBit;
#else
constexpr uintptr_t kPyTagBitMask = kPyCachedInactiveTagBit | kPyInTransitTagBit;
#endif

TVM_FFI_INLINE bool TVMFFIPyTagIsInactive(PyObject* tagged) {
  return (reinterpret_cast<uintptr_t>(tagged) & kPyCachedInactiveTagBit) != 0;
}
TVM_FFI_INLINE bool TVMFFIPyTagInTransit(PyObject* tagged) {
  return (reinterpret_cast<uintptr_t>(tagged) & kPyInTransitTagBit) != 0;
}
TVM_FFI_INLINE PyObject* TVMFFIPyRemoveTag(PyObject* tagged) {
  return reinterpret_cast<PyObject*>(reinterpret_cast<uintptr_t>(tagged) & ~kPyTagBitMask);
}
// Clear ONLY the InTransit bit (Inactive|InTransit -> Inactive, settled).
TVM_FFI_INLINE PyObject* TVMFFIPyTagClearInTransit(PyObject* tagged) {
  return reinterpret_cast<PyObject*>(reinterpret_cast<uintptr_t>(tagged) & ~kPyInTransitTagBit);
}

#ifdef Py_GIL_DISABLED
//---------------------------------------------------------------
// Free-threaded synchronization: a spin-lock encoded in ``tagged_pyobj``.
//
// The word is the single source of truth for a chandle's wrapper binding, and on
// free-threaded builds it is mutated concurrently. We serialize transitions with a
// per-word spin-lock (the Locked bit), CAS-acquired and store-released. Two rules
// keep it deadlock-free against the cyclic GC's stop-the-world:
//   1. The lock is held only across short, *park-free* sections -- pure word/header
//      memory ops, never an allocation, DecRef, GC op, or blocking call. So a lock
//      holder can never be frozen by GC mid-section, and spinners make bounded
//      progress.
//   2. Any wait (lock contention, or waiting out an InTransit transition) goes
//      through ``TVMFFIPyLockYield``, which detaches the thread state -- the
//      same cooperation PyMutex performs when it blocks -- so a concurrent
//      stop-the-world GC counts the waiter as paused and is never starved.
// Operations that must allocate/revive (which can reach a safepoint) do so with the
// lock *released*, publishing the InTransit bit first so others yield until the
// allocating thread re-locks and publishes the result.
//---------------------------------------------------------------

TVM_FFI_INLINE bool TVMFFIPyTagIsLocked(PyObject* tagged) {
  return (reinterpret_cast<uintptr_t>(tagged) & kPyLockedTagBit) != 0;
}

/*! \brief GC-safe back-off used by every wait below. Detaches the thread state so a
 *         concurrent stop-the-world GC treats this thread as paused; the matching
 *         re-attach blocks until any in-flight GC completes. Callers must hold an
 *         attached thread state (all our call sites do) and must NOT hold the word
 *         spin-lock across this. */
TVM_FFI_INLINE void TVMFFIPyLockYield() {
  PyThreadState* tstate = PyEval_SaveThread();
  PyEval_RestoreThread(tstate);
}

/*! \brief Acquire the per-word spin-lock. Returns the pre-lock state value (with the
 *         Locked bit cleared) that the caller now owns exclusively; the caller
 *         inspects it and must release via ``TVMFFIPyUnlockWord``. The Inactive /
 *         InTransit bits of the prior state are preserved in the return value, so
 *         the caller can decide whether to proceed or yield. */
TVM_FFI_INLINE PyObject* TVMFFIPyLockWord(PyCustomAllocHeader* h) {
  for (;;) {
    PyObject* cur = __atomic_load_n(&h->tagged_pyobj, __ATOMIC_RELAXED);
    if (!TVMFFIPyTagIsLocked(cur)) {
      PyObject* locked =
          reinterpret_cast<PyObject*>(reinterpret_cast<uintptr_t>(cur) | kPyLockedTagBit);
      // Acquire on success so the locked section happens-after the matching release.
      if (__atomic_compare_exchange_n(&h->tagged_pyobj, &cur, locked, /*weak=*/true,
                                      __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
        return cur;
      }
      // CAS failed (lost the race or spurious); ``cur`` reloaded -- retry without
      // yielding, the word was not Locked so contention is brief.
      continue;
    }
    TVMFFIPyLockYield();
  }
}

/*! \brief Release the per-word spin-lock, publishing ``new_state`` (which must have
 *         the Locked bit clear). Release-store so prior writes are visible to the
 *         next acquirer. */
TVM_FFI_INLINE void TVMFFIPyUnlockWord(PyCustomAllocHeader* h, PyObject* new_state) {
  __atomic_store_n(&h->tagged_pyobj, new_state, __ATOMIC_RELEASE);
}
#endif  // Py_GIL_DISABLED

/*!
 * \brief Per-thread vehicle carrying the inactive cached allocation address from
 *        ``make_ret_object`` (which knows the chandle) down into
 *        ``TVMFFIPyTpAlloc`` (which is handed only ``type`` and an item count).
 *
 * The slot's sole access primitive is a swap: store ``next``, return the prior
 * value. Taking the block is thus ``TVMFFIPyTLSReviveSlot(nullptr)`` --
 * read-and-clear in one step, with no separate clear to forget. Per-thread ->
 * free-threading safe.
 */
inline PyObject* TVMFFIPyTLSReviveSlot(PyObject* next) {
  static thread_local PyObject* slot = nullptr;
  std::swap(slot, next);
  return next;
}

/*! \brief Arm the cached allocation to be reused by the next ``tp_alloc`` on
 *         this thread. Called by ``make_ret_object`` immediately before
 *         ``cls.__new__``. */
extern "C" TVM_FFI_INLINE void TVMFFIPySetReviveBlock(PyObject* cached_alloc) {
  TVMFFIPyTLSReviveSlot(cached_alloc);
}

// Forward decl; defined below.
//
// NOTE: deliberately *not* TVM_FFI_INLINE. TVM_FFI_INLINE expands to
// [[gnu::always_inline]] which forbids taking the function's address as
// a stable, callable pointer — and we hand the address to the C++ side
// (stored in PyCustomAllocHeader::base.delete_space at allocate time).
inline void TVMFFIPyDeleteSpace(void* ptr);

// Atexit-driven shutdown guard. ``TVMFFIPyMarkPythonFinalizing`` flips
// the flag to false from an atexit hook registered in Cython module init;
// ``TVMFFIPyDeleteSpace`` reads it via ``TVMFFIPyIsPythonAlive`` before
// ``PyGILState_Ensure`` to avoid touching a teardown interpreter.
inline std::atomic<bool>& TVMFFIPyAliveFlagStorage() {
  static std::atomic<bool> flag{true};
  return flag;
}

extern "C" inline bool TVMFFIPyIsPythonAlive() noexcept {
  return TVMFFIPyAliveFlagStorage().load(std::memory_order_acquire);
}

extern "C" inline void TVMFFIPyMarkPythonFinalizing() noexcept {
  TVMFFIPyAliveFlagStorage().store(false, std::memory_order_release);
}

/*!
 * \brief True iff ``chandle`` was allocated through the Python custom
 *        allocator (full ``PyCustomAllocHeader`` ahead of it). False for
 *        allocations that came through libtvm_ffi's builtin default
 *        (only the base ``TVMFFIObjectAllocHeader``).
 *
 * Detection is by comparing ``base.delete_space`` against
 * ``TVMFFIPyDeleteSpace``: each frontend recognizes its own deleter
 * pointer, so multiple frontends can coexist without a flag bit on
 * ``TVMFFIObject``.
 */
TVM_FFI_INLINE bool TVMFFIPyIsCanonical(void* chandle) {
  if (chandle == nullptr) return false;
  TVMFFIObjectAllocHeader* base = reinterpret_cast<TVMFFIObjectAllocHeader*>(
      static_cast<char*>(chandle) - sizeof(TVMFFIObjectAllocHeader));
  return base->delete_space == &TVMFFIPyDeleteSpace;
}

//---------------------------------------------------------------
// State-machine helpers.
//---------------------------------------------------------------

extern "C" {

/*!
 * \brief Query the wrapper bound to ``chandle``. ``*out`` receives the
 *        *usable* (untagged) wrapper pointer; the return value says whether
 *        that wrapper is the live Active one.
 *
 * The (return, ``*out``) pair has four combinations; one cannot occur:
 *   (true,  non-NULL) : Active   -- the live canonical wrapper.
 *   (false, NULL)     : Detached / non-Python -- no wrapper bound.
 *   (false, non-NULL) : Inactive -- revivable cached allocation.
 *   (true,  NULL)     : impossible -- a true (Active) return always has a
 *                       real wrapper, so ``*out`` is non-NULL on every hit.
 *
 * \param chandle The chandle to query.
 * \param out Receives the usable (untagged) wrapper pointer, or NULL.
 * \return Whether the returned PyObject state is Active (the live canonical
 *         wrapper). False for Detached, Inactive, or non-Python chandles.
 */
TVM_FFI_INLINE bool TVMFFIPyTryGetAttachedPyObject(void* chandle, PyObject** out) {
  if (!TVMFFIPyIsCanonical(chandle)) {
    *out = nullptr;
    return false;
  }
  PyObject* tagged = TVMFFIPyHeader(chandle)->tagged_pyobj;
  *out = TVMFFIPyRemoveTag(tagged);
  return tagged != nullptr && !TVMFFIPyTagIsInactive(tagged);
}

}  // extern "C"

//---------------------------------------------------------------
// Forward declarations shared by SECTION A (make_ret) and the lifecycle sections.
//---------------------------------------------------------------

// Address of a CObject wrapper's ``chandle`` field, defined in object.pxi.
__PYX_EXTERN_C void** TVMFFICyObjectGetCHandlePtr(PyObject* ptr);

inline void TVMFFIPyTpFree(void* self);

#ifdef Py_GIL_DISABLED
// The free-threaded custom ``tp_dealloc`` slot, defined with the dealloc family below.
extern "C" inline void TVMFFIPyTpDeallocSlot(PyObject* self);
#endif

//---------------------------------------------------------------
// SECTION A -- alloc / revival / make_ret (HOT: per construction / per FFI return).
//
// The block birth (TVMFFIPyAllocate) and the whole Detached/Active/Inactive -> Active
// return transition (TVMFFIPyMakeRetObject, behind Cython's make_ret_object) plus the
// canonical-binding writer (TVMFFIPyRebindPyObject, used by move/construct/detach) and
// the wrapper allocator slot (TVMFFIPyTpAlloc) live here. Shared, build-agnostic members
// (Allocate, IsInactiveEligible, TpAlloc) sit above the one #ifdef/#else split; the FT
// arm carries the lock-synchronized writers (NewWrapper, Rebind, make_ret), the GIL arm
// the GIL-serialized ones (AttachPyObject, Rebind, make_ret).
//---------------------------------------------------------------

/*!
 * \brief Allocator entry registered with TVMFFISetCustomAllocator at
 *        Cython module init. Allocates ``sizeof(PyCustomAllocHeader) + size`` bytes
 *        with ``alignment``, zero-inits the header to the Detached
 *        state, wires ``base.delete_space = &TVMFFIPyDeleteSpace``, and
 *        returns the T location.
 *
 * Handler::New static_asserts ``alignof(T) <= alignof(max_align_t)``, so
 * the runtime ``alignment`` is bounded and ``base + sizeof(PyCustomAllocHeader)``
 * (= ``base + 16``) lands T naturally aligned for any T we allocate.
 */
inline void* TVMFFIPyAllocate(size_t size, size_t alignment, int32_t /*type_index*/,
                              void* /*context*/) {
  void* base_alloc =
      ::tvm::ffi::details::AlignedAllocRuntime(sizeof(PyCustomAllocHeader) + size, alignment);
  auto* h = static_cast<PyCustomAllocHeader*>(base_alloc);
  h->tagged_pyobj = nullptr;  // Detached
  h->base.delete_space = &TVMFFIPyDeleteSpace;
  return static_cast<char*>(base_alloc) + sizeof(PyCustomAllocHeader);
}

/*!
 * \brief Wrap a returned ``chandle`` into its canonical Python wrapper -- the
 *        whole Detached / Active / Inactive transition in one frame, behind
 *        Cython's ``make_ret_object``.
 *
 * The caller owns +1 (strong) on ``chandle``; ownership transfers to the
 * returned wrapper. Returns a new owned reference, or NULL with a Python error
 * set (the Cython side declares this ``object``, so NULL propagates as an
 * exception).
 *
 * \param chandle The returned object handle (caller owns +1 strong).
 * \param cls_type The wrapper class to instantiate (a ``PyTypeObject*``).
 * \return New owned wrapper reference, or NULL with a Python error set.
 */
#ifdef Py_GIL_DISABLED
/*! \brief Allocate (fresh) or revive (in place, at ``revive``'s address) a wrapper
 *         of type ``tp`` via ``tp_new``. Returns a new reference (refcount 1) or
 *         NULL with a Python error set. Used only by the free-threaded
 *         ``TVMFFIPyMakeRetObject`` (the GIL path keeps the original inline form). */
inline PyObject* TVMFFIPyNewWrapper(PyTypeObject* tp, PyObject* revive) {
  if (revive != nullptr) TVMFFIPySetReviveBlock(revive);
  PyObject* args = PyTuple_New(0);
  if (args == nullptr) {
    TVMFFIPySetReviveBlock(nullptr);  // disarm: tp_new will not run
    return nullptr;
  }
  PyObject* obj = tp->tp_new(tp, args, nullptr);
  Py_DECREF(args);
  TVMFFIPySetReviveBlock(nullptr);  // defensive: clear if tp_new bypassed tp_alloc
  return obj;
}

/*!
 * \brief Atomically rebind ``chandle``'s canonical PyObject to ``neo`` iff the current
 *        binding is exactly ``expect`` (Active(expect)) or Detached; otherwise leave the
 *        word untouched. ``neo == NULL`` clears the binding (Active -> Detached).
 *
 * One compare-and-rebind critical section covers every (re)binding the tie needs:
 *   - move:    ``Rebind(chandle, other, self)`` -- transfer canonical status other->self.
 *   - construct:``Rebind(chandle, NULL, self)`` -- attach self iff Detached.
 *   - detach:  ``Rebind(chandle, obj,  NULL)`` -- clear iff we are the Active binding.
 * No-op for non-canonical chandles, or when the word is otherwise bound/busy (``neo``
 * simply does not become canonical -- an identity-only outcome, never a safety issue).
 * (FT arm: arm TryIncRef before publishing Active under the word lock.)
 */
extern "C" TVM_FFI_INLINE void TVMFFIPyRebindPyObject(void* chandle, PyObject* expect,
                                                      PyObject* neo) {
  if (!TVMFFIPyIsCanonical(chandle)) return;
  PyCustomAllocHeader* h = TVMFFIPyHeader(chandle);
  // Arm TryIncRef before publishing Active, so a racing Active-hit make_ret can safely
  // TryIncRef ``neo``. Skip for a clearing rebind (neo == NULL) -- there is no wrapper
  // to arm and the publish is just Detached.
  if (neo != nullptr) PyUnstable_EnableTryIncRef(neo);
  PyObject* cur = TVMFFIPyLockWord(h);
  TVMFFIPyUnlockWord(h, (cur == expect || cur == nullptr) ? neo : cur);
}

// make_ret (free-threaded): a lock-loop over the four word states -- InTransit busy =>
// yield through a GC safepoint; Active => TryIncRef (authoritative, see below);
// Inactive => revive in place at the same address; Detached => fresh wrapper. Active is
// always published under the word spin-lock; alloc/revive run with the lock RELEASED,
// publishing the InTransit bit so peers yield instead of racing the allocation.
extern "C" inline PyObject* TVMFFIPyMakeRetObject(void* chandle, PyObject* cls_type) {
  PyTypeObject* tp = reinterpret_cast<PyTypeObject*>(cls_type);
  // Non-canonical chandle (no Python alloc header, e.g. a C++-static registry
  // object): never tied -- wrap fresh, transferring the caller's +1.
  if (!TVMFFIPyIsCanonical(chandle)) {
    PyObject* obj = TVMFFIPyNewWrapper(tp, nullptr);
    if (obj == nullptr) {
      TVMFFIObjectDecRef(chandle);
      return nullptr;
    }
    *TVMFFICyObjectGetCHandlePtr(obj) = chandle;
    return obj;
  }
  PyCustomAllocHeader* h = TVMFFIPyHeader(chandle);
  for (;;) {
    PyObject* cur = TVMFFIPyLockWord(h);
    // (1) A transition is in flight on this chandle -- either a dealloc handshake
    // settling, or another make_ret allocating / reviving; both publish InTransit.
    // Dealloc has priority: yield through a GC safepoint and retry.
    if (TVMFFIPyTagInTransit(cur)) {
      TVMFFIPyUnlockWord(h, cur);
      TVMFFIPyLockYield();
      continue;
    }
    // (2) Active hit: ``cur`` is the live canonical wrapper (no tag bits). TryIncRef
    // is the synchronizer -- and it is *authoritative* here because our custom
    // ``tp_dealloc`` (TVMFFIPyTpDeallocSlot) clears the Active binding BEFORE Cython's
    // resurrection bump runs. So a word still showing Active(cur) means cur's bump has
    // not happened: cur's refcount is truthful (>=1 => alive; ==0 => shared in
    // {0, MERGED}, both of which make TryIncRef fail). The doomed shared=MERGED-with-
    // count footprint never coexists with an Active(cur) word.
    if (cur != nullptr && !TVMFFIPyTagIsInactive(cur)) {
      if (PyUnstable_TryIncRef(cur)) {
        TVMFFIPyUnlockWord(h, cur);   // republish Active(cur) unchanged
        TVMFFIObjectDecRef(chandle);  // drop caller's redundant +1 (OUTSIDE the lock)
        return cur;
      }
      // ``cur`` reached refcount 0; its tp_dealloc is queued (blocked on our lock).
      // Release so dealloc can settle the word to Inactive, then retry to revive.
      TVMFFIPyUnlockWord(h, cur);
      TVMFFIPyLockYield();
      continue;
    }
    // (3) Inactive(W) clean: revive W in place at the same address (stable id()).
    if (cur != nullptr) {  // TagIsInactive(cur); InTransit already excluded
      PyObject* w = TVMFFIPyRemoveTag(cur);
      // Publish Inactive|InTransit (the claim) so others yield while we revive -- the
      // alloc must run with the lock RELEASED, as it can reach a GC safepoint. This is
      // the same word shape the dealloc handshake uses; it is unambiguous because a
      // tp_free/delete_space on ``w`` cannot run during the claim (we are its sole
      // referrer, reviving 0->1, and make_ret's strong chandle ref pins weak > 0).
      TVMFFIPyUnlockWord(
          h, reinterpret_cast<PyObject*>(reinterpret_cast<uintptr_t>(w) | kPyCachedInactiveTagBit |
                                         kPyInTransitTagBit));
      PyObject* obj = TVMFFIPyNewWrapper(tp, w);  // revives w in place
      if (obj == nullptr) {
        TVMFFIPyLockWord(h);  // restore Inactive(W), drop caller's +1
        TVMFFIPyUnlockWord(h, reinterpret_cast<PyObject*>(reinterpret_cast<uintptr_t>(w) |
                                                          kPyCachedInactiveTagBit));
        TVMFFIObjectDecRef(chandle);
        return nullptr;
      }
      *TVMFFICyObjectGetCHandlePtr(obj) = chandle;  // caller's +1 transfers to obj
      PyUnstable_EnableTryIncRef(obj);              // arm BEFORE publishing Active
      TVMFFIPyLockWord(h);
      TVMFFIPyUnlockWord(h, obj);  // publish Active(obj)
      return obj;
    }
    // (4) Detached (cur == NULL): fresh wrapper. Publish bare InTransit (the claim:
    // no wrapper pointer yet, so Inactive stays clear) so concurrent fresh-allocs on
    // this chandle dedup (they yield at (1), then Active-hit our wrapper). This is the
    // one word shape with InTransit set but Inactive clear; it is read only by (1)
    // (which tests InTransit alone) and cleared by the direct Active(obj) publish
    // below, never via TVMFFIPyTagClearInTransit.
    TVMFFIPyUnlockWord(h, reinterpret_cast<PyObject*>(kPyInTransitTagBit));
    PyObject* obj = TVMFFIPyNewWrapper(tp, nullptr);
    if (obj == nullptr) {
      TVMFFIPyLockWord(h);
      TVMFFIPyUnlockWord(h, nullptr);  // restore Detached
      TVMFFIObjectDecRef(chandle);
      return nullptr;
    }
    *TVMFFICyObjectGetCHandlePtr(obj) = chandle;  // caller's +1 transfers to obj
    PyUnstable_EnableTryIncRef(obj);
    TVMFFIPyLockWord(h);
    TVMFFIPyUnlockWord(h, obj);  // publish Active(obj)
    return obj;
  }
}
#else
/*!
 * \brief Bind ``obj`` to ``chandle`` as the canonical active PyObject (GIL build only).
 *        No-op for chandles without a Python alloc header. Used by the GIL ``make_ret``
 *        fresh path: it locks-and-publishes Active with no window for an allocation in
 *        between. (Deliberately absent on FT, where a fresh wrapper must be published
 *        through the InTransit-claim protocol or ``TVMFFIPyRebindPyObject``; a plain
 *        store would be a mis-synchronized publish, so any FT caller is a deliberate
 *        compile error.)
 */
TVM_FFI_INLINE void TVMFFIPyAttachPyObject(void* chandle, PyObject* obj) {
  if (!TVMFFIPyIsCanonical(chandle)) return;
  TVMFFIPyHeader(chandle)->tagged_pyobj = obj;
}

/*!
 * \brief Atomically rebind ``chandle``'s canonical PyObject to ``neo`` iff the current
 *        binding is exactly ``expect`` (Active(expect)) or Detached; otherwise leave the
 *        word untouched. ``neo == NULL`` clears the binding (Active -> Detached).
 *
 * One compare-and-rebind covers every (re)binding the tie needs (move / construct /
 * detach). On the GIL build the single-statement body is the whole critical section.
 */
extern "C" TVM_FFI_INLINE void TVMFFIPyRebindPyObject(void* chandle, PyObject* expect,
                                                      PyObject* neo) {
  if (!TVMFFIPyIsCanonical(chandle)) return;
  PyCustomAllocHeader* h = TVMFFIPyHeader(chandle);
  PyObject* cur = h->tagged_pyobj;
  if (cur == expect || cur == nullptr) {
    h->tagged_pyobj = neo;
  }
}

// make_ret (GIL): straight-line, GIL-serialized. Active hit => Py_INCREF the cached
// wrapper; Inactive => arm the revive block so tp_alloc reuses the cached allocation;
// Detached => fresh wrapper. No lock, no retry -- the GIL is the synchronizer.
extern "C" inline PyObject* TVMFFIPyMakeRetObject(void* chandle, PyObject* cls_type) {
  PyObject* attached;
  if (TVMFFIPyTryGetAttachedPyObject(chandle, &attached)) {
    // Active hit: return the cached wrapper; drop the caller's redundant +1.
    Py_INCREF(attached);
    TVMFFIObjectDecRef(chandle);
    return attached;
  }
  if (attached != nullptr) {
    // Inactive: revive the cached allocation in place (stable id()).
    TVMFFIPySetReviveBlock(attached);
  }
  PyObject* args = PyTuple_New(0);
  if (args == nullptr) {
    TVMFFIPySetReviveBlock(nullptr);  // disarm: tp_new will not run
    TVMFFIObjectDecRef(chandle);
    return nullptr;
  }
  PyTypeObject* tp = reinterpret_cast<PyTypeObject*>(cls_type);
  PyObject* obj = tp->tp_new(tp, args, nullptr);
  Py_DECREF(args);
  TVMFFIPySetReviveBlock(nullptr);  // defensive: clear if tp_new bypassed tp_alloc
  if (obj == nullptr) {
    TVMFFIObjectDecRef(chandle);
    return nullptr;
  }
  *TVMFFICyObjectGetCHandlePtr(obj) = chandle;
  TVMFFIPyAttachPyObject(chandle, obj);
  return obj;
}
#endif

//---------------------------------------------------------------
// PyObject-tying: the wrapper lifecycle (custom tp_alloc / tp_dealloc / tp_free
// and per-type slot installation).
//
// These run the dealloc handshake described in the top-of-file section header.
// They are kept in execution order -- eligibility & alloc, then the dealloc
// family (binding transition, the ``__dealloc__`` hook, and on free-threaded
// builds the pre-bump ``tp_dealloc`` slot + its routing), then free, then the
// per-type install -- coupled by the InTransit bit in ``tagged_pyobj`` that
// dealloc arms and free settles.
//---------------------------------------------------------------

/*!
 * \brief True iff a wrapper of ``wrapper``'s type may be cached & revived.
 *
 * Requirements (all must hold, else we genuinely free and lose only
 * stable-id-across-drop):
 *  - GC type: revival re-tracks and genuine free / reclaim use GC_Del.
 *  - our custom ``tp_free`` is installed: otherwise the generic free would
 *    reclaim the block while ``tagged_pyobj`` still points at it (UAF).
 *  - no instance ``__dict__`` (plain or managed): reusing a cached allocation whose dict
 *    region was cleared would need dict re-init we do not perform. Lean
 *    wrappers (the common, tested case) have ``tp_dictoffset == 0``.
 *  - no finalizer (``tp_finalize``, i.e. a Python ``__del__``): CPython sets
 *    a permanent GC-finalized bit the first time ``tp_finalize`` runs and
 *    never runs it again on that block. Reusing an inactive cached allocation would silently
 *    suppress ``__del__`` for every revived generation (the cached allocation's bit is
 *    already set and there is no public API to clear it). Excluding these
 *    types makes ``__del__`` fire correctly once per drop (genuine free each
 *    time); the only cost is no stable-id-across-drop.
 */
TVM_FFI_INLINE bool TVMFFIPyIsInactiveEligible(PyObject* wrapper) {
  PyTypeObject* tp = Py_TYPE(wrapper);
  if (!PyType_IS_GC(tp)) return false;
  if (tp->tp_free != &TVMFFIPyTpFree) return false;
  if (tp->tp_dictoffset != 0) return false;
  if ((tp->tp_flags & Py_TPFLAGS_MANAGED_DICT) != 0) return false;
  if (tp->tp_finalize != nullptr) return false;
  return true;
}

/*!
 * \brief Custom ``tp_alloc``. On the revival path (an inactive cached allocation was handed
 *        to this thread via ``TVMFFIPySetReviveBlock``) it revives the cached allocation
 *        in place — same address, so ``id()`` is stable — re-initializing it
 *        to match ``PyType_GenericAlloc``'s contract. Otherwise (miss) it
 *        forwards to ``PyType_GenericAlloc`` for a fresh, tracked object.
 *
 * Revive-path contract (must match what ``tp_new`` expects from
 * ``PyType_GenericAlloc``):
 *  1. zero the body ``[sizeof(PyObject), tp_basicsize)`` so ``__cinit__``
 *     sees clean fields;
 *  2. ``PyObject_Init`` -> ob_refcnt = 1, ob_type, INCREF(type);
 *  3. ``PyObject_GC_Track`` -> GenericAlloc returns a *tracked* object and
 *     ``tp_new`` does not track again, so the revive path must track.
 * (No stale GC-finalized bit to clear: the design uses no tp_finalize.)
 *
 * Fixed-size only: ``PyObject_Init`` resets ``ob_refcnt``/``ob_type`` but not
 * ``ob_size``. Every registered FFI wrapper is a fixed-size cdef class
 * (``tp_itemsize == 0``), asserted below; a future variable-sized type would
 * need ``PyObject_InitVar(.., nitems)`` here and a matching basicsize check.
 */
inline PyObject* TVMFFIPyTpAlloc(PyTypeObject* type, Py_ssize_t nitems) {
  // Take the revive block and leave the slot NULL in one step (per-thread).
  PyObject* blk = TVMFFIPyTLSReviveSlot(nullptr);
  if (blk != nullptr) {
    // REVIVAL: revive the inactive cached allocation at the same address. The body memset
    // below assumes ``type->tp_basicsize`` equals the cached allocation's original
    // basicsize. This holds because a chandle's ``type_index`` maps to one
    // stable wrapper class for the life of the process, and ``make_ret_object``
    // derives both the cached allocation (from the chandle) and ``type`` (= cls for that
    // same type_index) from the very same chandle on the revival path.
    assert(type->tp_itemsize == 0 &&
           "cache-&-revive supports only fixed-size wrappers; a variable-sized "
           "type needs PyObject_InitVar and a per-instance basicsize check");
    std::memset(reinterpret_cast<char*>(blk) + sizeof(PyObject), 0,
                static_cast<size_t>(type->tp_basicsize) - sizeof(PyObject));
    PyObject_Init(blk, type);
    PyObject_GC_Track(blk);
    return blk;
  }
  return PyType_GenericAlloc(type, nitems);  // MISS: fresh (already tracked)
}

//---------------------------------------------------------------
// SECTION B -- dealloc (HOT: per wrapper death).
//
// The cache-vs-free handshake across three slots: TVMFFIPyTpDealloc (the __dealloc__
// hook) opens it, TVMFFIPyTpFree settles it, and TVMFFIPyDeleteSpace (the C++ weak
// deleter) reclaims the block. On free-threaded builds the transition is hoisted
// PRE-bump into TVMFFIPyTpDeallocSlot (the custom tp_dealloc), backed by the
// NearestCdefBase walk + the original-thunk table; on the GIL build it runs from
// __dealloc__ post-bump (safe under the GIL). One #ifdef/#else split: the FT arm
// carries the lock-synchronized transition plus the slot machinery (no GIL twin);
// the GIL arm carries the GIL-serialized transition.
//---------------------------------------------------------------

#ifdef Py_GIL_DISABLED
// ===================== free-threaded arm =====================
/*! \brief The nearest non-heap (cdef) ancestor of ``tp``: the type CPython's
 *         ``subtype_dealloc`` base-walks to for a heap subtype, or ``tp`` itself for a
 *         cdef class. This is the single point where the tying ``tp_dealloc`` slot is
 *         installed (``TVMFFIPyInstallTypeSlots``) and where the original thunk is
 *         recorded (``TVMFFIPyFindOriginalDealloc``), so both must locate it the same
 *         way. Returns nullptr only for a type with no non-heap ancestor (not a
 *         ``CObject`` family type -- never reaches the tying paths). */
TVM_FFI_INLINE PyTypeObject* TVMFFIPyNearestCdefBase(PyTypeObject* tp) {
  for (PyTypeObject* t = tp; t != nullptr; t = t->tp_base) {
    if ((t->tp_flags & Py_TPFLAGS_HEAPTYPE) == 0) return t;
  }
  return nullptr;
}

/*!
 * \brief The dealloc binding transition: release the wrapper's +1 on chandle and
 *        open the cache-vs-free handshake for an eligible canonical wrapper.
 *
 *  Three paths:
 *   - Eligible canonical Active binding: Active -> Inactive | InTransit, keep the
 *     allocation cached, then DecRef (``tp_free`` settles the InTransit handshake).
 *   - Ineligible wrapper: detach the binding (Active -> Detached) BEFORE the DecRef,
 *     so a deleter firing inside it sees no stale Active back-pointer, then DecRef.
 *   - Non-canonical / not-our binding (cleared by an eager move, points at another
 *     wrapper or an inactive cached allocation, or a chandle with no Python alloc
 *     header): just null ``chandle`` and DecRef.
 *
 *  CALLED PRE-BUMP. On the GIL build this runs from ``__dealloc__``
 *  (``TVMFFIPyTpDealloc``). On the free-threaded build it MUST run before Cython's
 *  ``tp_dealloc`` resurrection bump (``Py_SET_REFCNT(o, +1)``): for a merge-path
 *  death that bump leaves the wrapper in a shared=MERGED state with a positive count,
 *  in which ``PyUnstable_TryIncRef`` *succeeds* even though Cython will unconditionally
 *  free the wrapper afterward. If the word still advertised Active(wrapper) during that
 *  window a concurrent ``make_ret`` Active-hit would revive a doomed reference (UAF).
 *  So on FT this is invoked from ``TVMFFIPyTpDeallocSlot`` (the custom ``tp_dealloc``),
 *  which clears the binding here and only then chains to Cython's thunk (= the bump).
 *  After that, Active(wrapper) in the word implies the bump has not run, so the
 *  wrapper's refcount is truthful and TryIncRef is authoritative.
 */
// tp_dealloc binding transition (free-threaded): lock the word, and if we are exactly
// the live Active binding (cur == wrapper) either cache it Inactive|InTransit (eligible)
// or detach it (ineligible); otherwise leave the word and take the shared genuine-free
// tail. The DecRef MUST run outside the lock -- it can fire the chandle deleter ->
// TVMFFIPyDeleteSpace, which re-locks the same (non-reentrant) word.
extern "C" TVM_FFI_INLINE void TVMFFIPyTpDeallocImpl(void** ptr_to_chandle, PyObject* wrapper) {
  void* chandle = *ptr_to_chandle;
  // Already released by an eager move (detached to NULL); nothing to do. NULL is
  // the only released state here -- the transit marker lives in ``tagged_pyobj``.
  if (chandle == nullptr) return;
  if (TVMFFIPyIsCanonical(chandle)) {
    PyCustomAllocHeader* h = TVMFFIPyHeader(chandle);
    PyObject* cur = TVMFFIPyLockWord(h);
    if (cur == wrapper) {
      if (TVMFFIPyIsInactiveEligible(wrapper)) {
        // Active -> Inactive | InTransit, then DecRef outside the lock.
        TVMFFIPyUnlockWord(
            h, reinterpret_cast<PyObject*>(reinterpret_cast<uintptr_t>(wrapper) |
                                           kPyCachedInactiveTagBit | kPyInTransitTagBit));
        TVMFFIObjectDecRef(chandle);
        return;
      }
      // Active -> Detached: not eligible. Publish NULL before the DecRef so a
      // deleter firing inside it sees no stale Active binding.
      TVMFFIPyUnlockWord(h, nullptr);
    } else {
      TVMFFIPyUnlockWord(h, cur);  // not our binding -- leave it, take the tail
    }
  }
  // Tail (not eligible / not ours / non-canonical): genuine free. Null wrapper.chandle
  // BEFORE DecRef so the deleter chain observes a consistent (chandle == NULL, no +1
  // owed) state.
  *ptr_to_chandle = nullptr;
  TVMFFIObjectDecRef(chandle);
}

// Free-threaded custom ``tp_dealloc`` (the pre-bump binding-clear hook).
//
// Cython's generated ``tp_dealloc`` bumps the wrapper's refcount
// (``Py_SET_REFCNT(o, +1)``) before running ``__dealloc__`` and then frees it
// unconditionally. For a merge-path death (a wrapper that crossed threads) that bump
// produces a ``shared = _Py_REF_SHARED(1, MERGED)`` footprint on which
// ``PyUnstable_TryIncRef`` SUCCEEDS -- so a concurrent ``make_ret`` Active-hit would
// revive a wrapper Cython is about to free (borrowed-ref UAF). The fix is to perform
// the Active->Inactive/Detached binding transition BEFORE that bump, in a custom
// ``tp_dealloc`` slot that then chains to Cython's original thunk.
//
// Slot routing across the type hierarchy (verified empirically):
//   * Non-heap cdef classes (CObject and its cdef subclasses Function/Tensor/Error/
//     OpaquePyObject/CContainerBase) each carry a Cython dealloc thunk. We replace that
//     slot with ``TVMFFIPyTpDeallocSlot`` and remember the original.
//   * Heap subtypes (dataclasses, Array, Map, ...) keep CPython's ``subtype_dealloc``,
//     which base-walks ``tp_base`` to the nearest non-``subtype_dealloc`` dealloc and
//     calls it -- i.e. it lands on our slot installed on the nearest cdef ancestor. So
//     we install the slot on that nearest non-heap ancestor (idempotently) and never on
//     heap subtypes, which avoids re-entry while still intercepting every death pre-bump.
//
// The original thunk is found via ``TVMFFIPyNearestCdefBase(Py_TYPE(self))`` -- the same
// type ``subtype_dealloc`` would have dispatched to, and where we recorded the original.
//---------------------------------------------------------------

// Append-only table mapping a wrapped cdef type to its original ``tp_dealloc``. Writes
// happen only at type registration (``TVMFFIPyInstallTypeSlots``); reads happen in the
// dealloc slot. Lock-free reads: an entry's fields are written before ``g_orig_count``
// is release-stored, and readers acquire-load the count, so a visible slot always has a
// fully-initialized original. Capacity is tiny (one per cdef CObject-family type).
struct TVMFFIPyOrigDealloc {
  PyTypeObject* tp;
  destructor orig;
};
inline constexpr int kPyMaxWrappedTypes = 64;
inline TVMFFIPyOrigDealloc g_orig_dealloc[kPyMaxWrappedTypes];
inline std::atomic<int> g_orig_count{0};

/*! \brief Find the recorded original ``tp_dealloc`` for ``tp``: look up its nearest cdef
 *         ancestor (where the slot was installed and the original recorded) in the table.
 *         Returns nullptr if none (should not happen for a type whose instances reach our
 *         slot -- that nearest ancestor is exactly what we wrapped). */
inline destructor TVMFFIPyFindOriginalDealloc(PyTypeObject* tp) {
  PyTypeObject* base = TVMFFIPyNearestCdefBase(tp);
  int n = g_orig_count.load(std::memory_order_acquire);
  for (int i = 0; i < n; ++i) {
    if (g_orig_dealloc[i].tp == base) return g_orig_dealloc[i].orig;
  }
  return nullptr;
}

/*! \brief Custom ``tp_dealloc`` (FT only): clear the wrapper<->chandle binding pre-bump,
 *         then chain to the original Cython thunk (which performs the bump, ``__dealloc__``,
 *         and ``tp_free``). */
extern "C" inline void TVMFFIPyTpDeallocSlot(PyObject* self) {
  TVMFFIPyTpDeallocImpl(TVMFFICyObjectGetCHandlePtr(self), self);
  destructor orig = TVMFFIPyFindOriginalDealloc(Py_TYPE(self));
  // ``orig`` is always found for instances that reach this slot (their type or a cdef
  // ancestor was wrapped). The guard is purely defensive.
  if (orig != nullptr) {
    orig(self);
  }
}

/*!
 * \brief ``__dealloc__`` hook, called by Cython's ``tp_dealloc`` thunk AFTER its
 *        resurrection bump.
 *
 *        GIL build: no custom ``tp_dealloc`` slot is installed, so this is the only
 *        site that runs the binding transition. Post-bump is safe under the GIL.
 *
 *        Free-threaded build: a NO-OP. The transition already ran PRE-bump in
 *        ``TVMFFIPyTpDeallocSlot``, the custom ``tp_dealloc`` installed on every cdef
 *        carrier. Coverage is total because the ``CObject`` cdef family is a closed,
 *        compile-time set -- ``CObject``, ``CContainerBase``, ``OpaquePyObject``,
 *        ``Error``, ``Tensor``, ``Function`` -- each wrapped at registration
 *        (``TVMFFIPyInstallTypeSlots``) or, for the registration-bypassing carrier
 *        ``OpaquePyObject``, at init (``TVMFFIPyInstallTypeSlotsForOpaque``); every
 *        heap subtype routes through ``subtype_dealloc`` to its nearest cdef ancestor's
 *        slot. So no FT ``CObject`` death reaches here with the transition still owed,
 *        and re-running it post-bump would double-DecRef and clobber the InTransit
 *        locator ``tp_free`` needs. (The table that backs the slot holds at most that
 *        handful of types against a capacity of ``kPyMaxWrappedTypes`` = 64, so it
 *        never overflows into an unwrapped-carrier degrade.)
 */
extern "C" TVM_FFI_INLINE void TVMFFIPyTpDealloc(void** ptr_to_chandle, PyObject* wrapper) {
  (void)ptr_to_chandle;
  (void)wrapper;
}

/*!
 * \brief Custom ``tp_free``, the second step of the dealloc handshake. The
 *        InTransit bit (read here) says whether the chandle's deleter fired
 *        during the ``TVMFFIPyTpDealloc`` DecRef: still set => keep the
 *        allocation cached Inactive; cleared => free the C++ block too. The
 *        ``chandle == NULL`` / non-canonical path is a plain genuine free,
 *        dispatching on GC-ness like CPython's default.
 */
// tp_free (free-threaded): for a canonical chandle, lock the word and read InTransit --
// set => the chandle outlived us, settle to Inactive and keep ``self`` cached; clear =>
// the deleter deferred the block free to us, free it. ``*chandle_ptr = nullptr`` MUST be
// written while still holding the lock, BEFORE UnlockWord publishes Inactive: otherwise a
// make_ret revive could grab the Inactive word and re-set ``self``'s chandle, only for
// this stale NULL store to clobber it (Active wrapper with chandle == NULL -> crash on
// the next field access). The lock also serializes against TVMFFIPyDeleteSpace on the
// same chandle.
inline void TVMFFIPyTpFree(void* self) {
  void** chandle_ptr = TVMFFICyObjectGetCHandlePtr(static_cast<PyObject*>(self));
  void* chandle = *chandle_ptr;
  if (chandle != nullptr && TVMFFIPyIsCanonical(chandle)) {
    PyCustomAllocHeader* h = TVMFFIPyHeader(chandle);  // header read BEFORE any free below
    PyObject* cur = TVMFFIPyLockWord(h);
    if (TVMFFIPyTagInTransit(cur)) {
      // Case 0: chandle outlived us -- settle to stable Inactive, keep ``self``
      // cached as the Inactive allocation (do NOT free it here).
      *chandle_ptr = nullptr;
      TVMFFIPyUnlockWord(h, TVMFFIPyTagClearInTransit(cur));
      return;
    }
    // Case 1: deleter fired and deferred the block free to us (its sole free).
    *chandle_ptr = nullptr;
    TVMFFIPyUnlockWord(h, cur);
    ::tvm::ffi::details::AlignedFree(static_cast<char*>(chandle) - sizeof(PyCustomAllocHeader));
  }
  PyObject* op = static_cast<PyObject*>(self);
  if (PyObject_IS_GC(op)) {
    PyObject_GC_Del(op);
  } else {
    PyObject_Free(op);
  }
}

/*!
 * \brief delete_space callback (installed by TVMFFIPyAllocate), invoked from the
 *        C++ Weak deleter when the chandle's block's weak count hits 0. On an
 *        Inactive binding it reads the InTransit bit: set => an in-flight
 *        ``tp_free`` will free the block, so defer (keep it readable); clear =>
 *        reclaim the cached wrapper and free the block here. Detached just frees
 *        the block. The InTransit read is GIL-ordered against ``tp_free``'s
 *        clear so a cross-thread last-ref never defers a freed block; during
 *        teardown the same-thread defer still holds and stray cases only leak.
 */
// delete_space (free-threaded): a LOCK-FREE fast path for the common Detached case --
// every C++-only object (allocated through the process-wide custom allocator but never
// surfaced to Python) dies here, so keep it off the word lock to avoid serializing
// non-Python destruction or risking a lock-order inversion against a C++ lock held during
// destruction. When the weak count hits zero there are no live refs, so the word can only
// be Detached or Inactive(|InTransit) -- never Active/Locked, and never a make_ret claim
// (a claim holds a strong chandle ref => weak > 0, so it cannot overlap this weak->0
// callback). The InTransit it may see is therefore unambiguously the dealloc handshake's.
// An Inactive binding reads InTransit under the lock: set => an in-flight tp_free will
// free the block (defer); clear => reclaim the cached wrapper and free here.
inline void TVMFFIPyDeleteSpace(void* ptr) {
  void* base_alloc = static_cast<char*>(ptr) - sizeof(PyCustomAllocHeader);
  auto* h = static_cast<PyCustomAllocHeader*>(base_alloc);
  PyObject* cur0 = __atomic_load_n(&h->tagged_pyobj, __ATOMIC_ACQUIRE);
  if (!TVMFFIPyTagIsInactive(cur0)) {  // Detached: lock-free free
    ::tvm::ffi::details::AlignedFree(base_alloc);
    return;
  }
  if (TVMFFIPyIsPythonAlive()) {
    PyGILState_STATE gstate = PyGILState_Ensure();
    if (TVMFFIPyIsPythonAlive()) {
      PyObject* cur = TVMFFIPyLockWord(h);
      if (TVMFFIPyTagInTransit(cur)) {
        // In-flight: defer both frees to tp_free; keep the block.
        TVMFFIPyUnlockWord(h, TVMFFIPyTagClearInTransit(cur));
        PyGILState_Release(gstate);
        return;
      }
      // Settled: detach, reclaim the cached wrapper (outside the word lock), then
      // free the block below.
      PyObject* wrapper = TVMFFIPyRemoveTag(cur);
      TVMFFIPyUnlockWord(h, nullptr);
      PyObject_GC_Del(wrapper);
      PyGILState_Release(gstate);
    } else {
      PyGILState_Release(gstate);
    }
  } else if (TVMFFIPyTagInTransit(cur0)) {
    // Teardown, same-thread in-flight: defer the block free (no thread state).
    __atomic_store_n(&h->tagged_pyobj, TVMFFIPyTagClearInTransit(cur0), __ATOMIC_RELEASE);
    return;
  }
  ::tvm::ffi::details::AlignedFree(base_alloc);
}

/*! \brief Record + replace the ``tp_dealloc`` of ``tp`` with the FT pre-bump slot,
 *         once. Idempotent: a type already carrying the slot, or one with no original
 *         to chain to, is left untouched. */
inline void TVMFFIPyWrapDealloc(PyTypeObject* tp) {
  destructor cur = tp->tp_dealloc;
  if (cur == &TVMFFIPyTpDeallocSlot || cur == nullptr) return;  // already wrapped / none
  // Append (tp, cur) before publishing the bumped count, so a concurrent reader that
  // sees the new count sees a fully-initialized entry. Registration is the sole writer
  // and is effectively serialized by the import path; a relaxed read of the count for
  // the duplicate check is fine (we only ever append).
  int n = g_orig_count.load(std::memory_order_relaxed);
  for (int i = 0; i < n; ++i) {
    if (g_orig_dealloc[i].tp == tp) return;  // already recorded
  }
  if (n >= kPyMaxWrappedTypes) return;  // table full: leave unwrapped (degrade safely)
  g_orig_dealloc[n].tp = tp;
  g_orig_dealloc[n].orig = cur;
  g_orig_count.store(n + 1, std::memory_order_release);
  tp->tp_dealloc = &TVMFFIPyTpDeallocSlot;
}
#else
// ========================= GIL arm =========================
/*!
 * \brief The dealloc binding transition: release the wrapper's +1 on chandle and
 *        open the cache-vs-free handshake for an eligible canonical wrapper.
 *
 *  Three paths:
 *   - Eligible canonical Active binding: Active -> Inactive | InTransit, keep the
 *     allocation cached, then DecRef (``tp_free`` settles the InTransit handshake).
 *   - Ineligible wrapper: detach the binding (Active -> Detached) BEFORE the DecRef,
 *     so a deleter firing inside it sees no stale Active back-pointer, then DecRef.
 *   - Non-canonical / not-our binding (cleared by an eager move, points at another
 *     wrapper or an inactive cached allocation, or a chandle with no Python alloc
 *     header): just null ``chandle`` and DecRef.
 *
 *  CALLED PRE-BUMP. On the GIL build this runs from ``__dealloc__``
 *  (``TVMFFIPyTpDealloc``). On the free-threaded build it MUST run before Cython's
 *  ``tp_dealloc`` resurrection bump (``Py_SET_REFCNT(o, +1)``): for a merge-path
 *  death that bump leaves the wrapper in a shared=MERGED state with a positive count,
 *  in which ``PyUnstable_TryIncRef`` *succeeds* even though Cython will unconditionally
 *  free the wrapper afterward. If the word still advertised Active(wrapper) during that
 *  window a concurrent ``make_ret`` Active-hit would revive a doomed reference (UAF).
 *  So on FT this is invoked from ``TVMFFIPyTpDeallocSlot`` (the custom ``tp_dealloc``),
 *  which clears the binding here and only then chains to Cython's thunk (= the bump).
 *  After that, Active(wrapper) in the word implies the bump has not run, so the
 *  wrapper's refcount is truthful and TryIncRef is authoritative.
 */
// tp_dealloc binding transition (GIL): same shape, GIL-serialized so the word is a plain
// field. Past the canonical guard ``chandle`` is a real object, so the binding (if ours)
// is the live Active wrapper -- a plain ``==`` is the canonical check.
extern "C" TVM_FFI_INLINE void TVMFFIPyTpDeallocImpl(void** ptr_to_chandle, PyObject* wrapper) {
  void* chandle = *ptr_to_chandle;
  if (chandle == nullptr) return;
  if (TVMFFIPyIsCanonical(chandle)) {
    PyCustomAllocHeader* h = TVMFFIPyHeader(chandle);
    if (h->tagged_pyobj == wrapper) {
      if (TVMFFIPyIsInactiveEligible(wrapper)) {
        // Active -> Inactive: tag Inactive | InTransit, then DecRef. ``chandle``
        // keeps pointing at the C++ block as a header locator; ``tp_free`` reads
        // the InTransit bit from there to settle the handshake.
        h->tagged_pyobj = reinterpret_cast<PyObject*>(reinterpret_cast<uintptr_t>(wrapper) |
                                                      kPyCachedInactiveTagBit | kPyInTransitTagBit);
        TVMFFIObjectDecRef(chandle);
        return;
      }
      // Active -> Detached: not eligible. Detach first so the chandle deleter
      // (which may fire inside the DecRef below) sees no stale Active binding.
      h->tagged_pyobj = nullptr;
    }
  }
  // Tail (not eligible / not ours / non-canonical): genuine free.
  *ptr_to_chandle = nullptr;
  TVMFFIObjectDecRef(chandle);
}

/*!
 * \brief ``__dealloc__`` hook, called by Cython's ``tp_dealloc`` thunk AFTER its
 *        resurrection bump.
 *
 *        GIL build: no custom ``tp_dealloc`` slot is installed, so this is the only
 *        site that runs the binding transition. Post-bump is safe under the GIL.
 *
 *        Free-threaded build: a NO-OP. The transition already ran PRE-bump in
 *        ``TVMFFIPyTpDeallocSlot``, the custom ``tp_dealloc`` installed on every cdef
 *        carrier. Coverage is total because the ``CObject`` cdef family is a closed,
 *        compile-time set -- ``CObject``, ``CContainerBase``, ``OpaquePyObject``,
 *        ``Error``, ``Tensor``, ``Function`` -- each wrapped at registration
 *        (``TVMFFIPyInstallTypeSlots``) or, for the registration-bypassing carrier
 *        ``OpaquePyObject``, at init (``TVMFFIPyInstallTypeSlotsForOpaque``); every
 *        heap subtype routes through ``subtype_dealloc`` to its nearest cdef ancestor's
 *        slot. So no FT ``CObject`` death reaches here with the transition still owed,
 *        and re-running it post-bump would double-DecRef and clobber the InTransit
 *        locator ``tp_free`` needs. (The table that backs the slot holds at most that
 *        handful of types against a capacity of ``kPyMaxWrappedTypes`` = 64, so it
 *        never overflows into an unwrapped-carrier degrade.)
 */
extern "C" TVM_FFI_INLINE void TVMFFIPyTpDealloc(void** ptr_to_chandle, PyObject* wrapper) {
  TVMFFIPyTpDeallocImpl(ptr_to_chandle, wrapper);
}

/*!
 * \brief Custom ``tp_free``, the second step of the dealloc handshake. The
 *        InTransit bit (read here) says whether the chandle's deleter fired
 *        during the ``TVMFFIPyTpDealloc`` DecRef: still set => keep the
 *        allocation cached Inactive; cleared => free the C++ block too. The
 *        ``chandle == NULL`` / non-canonical path is a plain genuine free,
 *        dispatching on GC-ness like CPython's default.
 */
// tp_free (GIL): same handshake, GIL-serialized so the word is a plain field.
inline void TVMFFIPyTpFree(void* self) {
  void** chandle_ptr = TVMFFICyObjectGetCHandlePtr(static_cast<PyObject*>(self));
  void* chandle = *chandle_ptr;
  if (chandle != nullptr && TVMFFIPyIsCanonical(chandle)) {
    PyCustomAllocHeader* h = TVMFFIPyHeader(chandle);  // header read BEFORE any free below
    if (TVMFFIPyTagInTransit(h->tagged_pyobj)) {
      // Case 0: chandle outlived us -- settle to stable Inactive, keep cached.
      h->tagged_pyobj = TVMFFIPyTagClearInTransit(h->tagged_pyobj);
      *chandle_ptr = nullptr;
      return;
    }
    // Case 1: deleter fired and deferred the block free to us (its sole free).
    *chandle_ptr = nullptr;
    ::tvm::ffi::details::AlignedFree(static_cast<char*>(chandle) - sizeof(PyCustomAllocHeader));
  }
  PyObject* op = static_cast<PyObject*>(self);
  if (PyObject_IS_GC(op)) {
    PyObject_GC_Del(op);
  } else {
    PyObject_Free(op);
  }
}

/*!
 * \brief delete_space callback (installed by TVMFFIPyAllocate), invoked from the
 *        C++ Weak deleter when the chandle's block's weak count hits 0. On an
 *        Inactive binding it reads the InTransit bit: set => an in-flight
 *        ``tp_free`` will free the block, so defer (keep it readable); clear =>
 *        reclaim the cached wrapper and free the block here. Detached just frees
 *        the block. The InTransit read is GIL-ordered against ``tp_free``'s
 *        clear so a cross-thread last-ref never defers a freed block; during
 *        teardown the same-thread defer still holds and stray cases only leak.
 */
// delete_space (GIL): same handshake, GIL-serialized. The InTransit read is GIL-ordered
// against tp_free's clear, so a last-ref never defers a freed block; during teardown the
// same-thread defer still holds and stray cases only leak.
inline void TVMFFIPyDeleteSpace(void* ptr) {
  void* base_alloc = static_cast<char*>(ptr) - sizeof(PyCustomAllocHeader);
  auto* h = static_cast<PyCustomAllocHeader*>(base_alloc);
  if (TVMFFIPyTagIsInactive(h->tagged_pyobj)) {
    if (TVMFFIPyIsPythonAlive()) {
      PyGILState_STATE gstate = PyGILState_Ensure();
      if (TVMFFIPyIsPythonAlive()) {
        if (TVMFFIPyTagInTransit(h->tagged_pyobj)) {
          // In-flight: defer both frees to tp_free; keep the block.
          h->tagged_pyobj = TVMFFIPyTagClearInTransit(h->tagged_pyobj);
          PyGILState_Release(gstate);
          return;
        }
        // Settled: reclaim the cached wrapper, then free the block below.
        PyObject_GC_Del(TVMFFIPyRemoveTag(h->tagged_pyobj));
      }
      PyGILState_Release(gstate);
    } else if (TVMFFIPyTagInTransit(h->tagged_pyobj)) {
      // Teardown, same-thread in-flight: defer the block free (no GIL needed).
      h->tagged_pyobj = TVMFFIPyTagClearInTransit(h->tagged_pyobj);
      return;
    }
  }
  ::tvm::ffi::details::AlignedFree(base_alloc);
}
#endif  // Py_GIL_DISABLED

//---------------------------------------------------------------
// SECTION C -- installation (COLD: once per registered type / once per process).
//
// ``TVMFFIPyInstallTypeSlots`` patches a registered type's ``tp_alloc`` / ``tp_free``
// (and, on FT, the pre-bump ``tp_dealloc`` slot). These slots are NOT inherited by
// dynamically created subtypes (CPython resets them per dynamic subtype), so every
// registered type is patched at its Cython registration choke point. Idempotent; no-op
// when ``type_obj`` is not a type object.
//
// ``TVMFFIPyInstallTypeSlotsForOpaque`` extends coverage to ``OpaquePyObject`` -- the one
// cdef carrier built via ``__new__`` that bypasses the registration funnel. On FT this
// makes slot coverage of the closed ``CObject`` cdef family total (the invariant that
// lets ``TVMFFIPyTpDealloc`` be an unconditional FT no-op). On the GIL build it is a NO-OP
// -- ``OpaquePyObject`` keeps its default slots, exactly as before, so the GIL build is
// unchanged. (The extra slots would be inert for it anyway: never bound Active in any
// word, so ``tp_alloc`` always misses to ``PyType_GenericAlloc`` and ``tp_free`` always
// takes the plain path; only the pre-bump dealloc routing matters.)
//
// ``TVMFFIPyRegisterDefaultAllocator`` (shared, below the split) installs the process-wide
// block allocator once at module init.
//---------------------------------------------------------------

#ifdef Py_GIL_DISABLED
extern "C" TVM_FFI_INLINE void TVMFFIPyInstallTypeSlots(PyObject* type_obj) {
  if (type_obj == nullptr || !PyType_Check(type_obj)) return;
  PyTypeObject* tp = reinterpret_cast<PyTypeObject*>(type_obj);
  tp->tp_alloc = &TVMFFIPyTpAlloc;
  tp->tp_free = &TVMFFIPyTpFree;
  // Install the pre-bump dealloc slot on the nearest non-heap (cdef) ancestor -- the
  // exact type ``subtype_dealloc`` base-walks to for a heap subtype, and ``tp`` itself
  // for a cdef class. Wrapping there intercepts every death pre-bump without re-entering
  // ``subtype_dealloc`` (which we never replace). Heap subtypes keep ``subtype_dealloc``.
  if (PyTypeObject* base = TVMFFIPyNearestCdefBase(tp)) {
    TVMFFIPyWrapDealloc(base);
  }
}
extern "C" TVM_FFI_INLINE void TVMFFIPyInstallTypeSlotsForOpaque(PyObject* type_obj) {
  TVMFFIPyInstallTypeSlots(type_obj);  // give the registration-bypass carrier the slots too
}
#else
extern "C" TVM_FFI_INLINE void TVMFFIPyInstallTypeSlots(PyObject* type_obj) {
  if (type_obj == nullptr || !PyType_Check(type_obj)) return;
  PyTypeObject* tp = reinterpret_cast<PyTypeObject*>(type_obj);
  tp->tp_alloc = &TVMFFIPyTpAlloc;
  tp->tp_free =
      &TVMFFIPyTpFree;  // no custom tp_dealloc slot: the bump UAF cannot occur under the GIL
}
extern "C" TVM_FFI_INLINE void TVMFFIPyInstallTypeSlotsForOpaque(PyObject* type_obj) {
  (void)type_obj;  // OpaquePyObject keeps its default slots on the GIL build
}
#endif

/*!
 * \brief Install ``TVMFFIPyAllocate`` as the process-wide custom allocator.
 *        Storage for the registered entry is a function-static so the
 *        address is process-stable.
 */
extern "C" TVM_FFI_INLINE int TVMFFIPyRegisterDefaultAllocator() {
  // Installed on both the GIL and free-threaded builds; on free-threaded builds the
  // header state machine is lock-synchronized (see "Free-threaded builds" above).
  static TVMFFICustomAllocator allocator{&TVMFFIPyAllocate, /*context=*/nullptr};
  return TVMFFISetCustomAllocator(&allocator);
}

//------------------------------------------------------------------------------------
// Helpers for Python thread-state attachment
//------------------------------------------------------------------------------------
//
// On classic builds, PyGILState_Ensure attaches the current thread and acquires the GIL.
// On free-threaded builds, there is no process-wide GIL to acquire, but CPython still
// requires an attached thread state before manipulating Python refcounts.
class TVMFFIPyWithAttachedThreadState {
 public:
  TVMFFIPyWithAttachedThreadState() noexcept { gstate_ = PyGILState_Ensure(); }
  ~TVMFFIPyWithAttachedThreadState() { PyGILState_Release(gstate_); }

 private:
  PyGILState_STATE gstate_;
};

/*!
 * \brief Closure state carried as the resource handle for an FFI function that
 *        wraps a Python callable and optional exchange api for tensor handling.
 *
 * Created by TVMFFIPyConvertPyCallback and released by
 * TVMFFIPyCallbackClosure::Deleter when the FFI function is destroyed.
 */
struct TVMFFIPyCallbackClosure {
  /*! \brief Strong reference to the Python callable. */
  PyObject* callable;
  /*! \brief Optional DLPack exchange API used when constructing Tensor returns. */
  const DLPackExchangeAPI* dlpack_exchange_api;
  /*!
   * \brief Deleter registered with TVMFFIFunctionCreate. Runs on FFI function destroy.
   *
   * Releases the closure's strong Python reference and frees the closure.
   */
  static void Deleter(void* context) noexcept {
    TVMFFIPyWithAttachedThreadState thread_state;
    auto* closure = static_cast<TVMFFIPyCallbackClosure*>(context);
    Py_DecRef(closure->callable);
    delete closure;
  }
};

/*!
 * \brief Thread-local call stack used by TVMFFIPyCallContext.
 */
class TVMFFIPyCallStack {
 public:
  /*! \brief The stack of arguments */
  std::vector<TVMFFIAny> args_stack;
  /*! \brief The top of the argument call stack currently */
  int64_t args_stack_top = 0;
  /*!
   * \brief The stack of extra temporary Python objects that may not fit into
   * one temp per argument budget, mainly used by value protocol.
   */
  std::vector<PyObject*> extra_temp_py_objects_stack;

  /*! \brief Constructor to initialize the call stack */
  TVMFFIPyCallStack() {
    // keep it 4K as default stack size so it is page aligned
    constexpr size_t kDefaultStackSize = 4096;
    // fit everything roughly 4K stack
    args_stack.resize(kDefaultStackSize / sizeof(TVMFFIAny));
    extra_temp_py_objects_stack.reserve(16);
  }
};

//---------------------------------------------------------------------------------------------
// Support for Python -> FFI function calls.
//---------------------------------------------------------------------------------------------
/*!
 * \brief Context for each ffi call to track the stream, device and temporary arguments.
 */
class TVMFFIPyCallContext {
 public:
  /*! \brief The workspace for the packed arguments */
  TVMFFIAny* packed_args = nullptr;
  /*! \brief Detected device type, if any */
  int device_type = -1;
  /*! \brief Detected device id, if any */
  int device_id = 0;
  /*! \brief Detected stream, if any */
  void* stream = nullptr;
  /*! \brief the DLPack exchange API, if any */
  const DLPackExchangeAPI* dlpack_c_exchange_api{nullptr};
  /*! \brief pointer to the call stack space */
  TVMFFIPyCallStack* call_stack = nullptr;
  /*! \brief the temporary arguments to be recycled */
  void** temp_ffi_objects = nullptr;
  /*! \brief the temporary arguments to be recycled */
  void** temp_py_objects = nullptr;
  /*! \brief the number of temporary arguments */
  int num_temp_ffi_objects = 0;
  /*! \brief the number of temporary arguments */
  int num_temp_py_objects = 0;

  /*! \brief RAII guard constructor to create a TVMFFIPyCallContext */
  TVMFFIPyCallContext(TVMFFIPyCallStack* call_stack, int64_t num_args) : call_stack(call_stack) {
    // In most cases, it will try to allocate from temp_stack,
    // then allocate from heap if the request goes beyond the stack size.
    static_assert(sizeof(TVMFFIAny) >= (sizeof(void*) * 2));
    static_assert(alignof(TVMFFIAny) % alignof(void*) == 0);
    old_args_stack_top_ = call_stack->args_stack_top;
    int64_t requested_count = num_args * 2;
    TVMFFIAny* stack_head = call_stack->args_stack.data() + call_stack->args_stack_top;
    if (call_stack->args_stack_top + requested_count >
        static_cast<int64_t>(call_stack->args_stack.size())) {
      // allocate from heap
      heap_ptr_ = new TVMFFIAny[requested_count];
      stack_head = heap_ptr_;
    } else {
      call_stack->args_stack_top += requested_count;
    }
    this->packed_args = stack_head;
    // by default we co-locate the temporary arguments with packed arguments
    // for better cache locality with one temp per argument budget.
    this->temp_ffi_objects = reinterpret_cast<void**>(stack_head + num_args);
    this->temp_py_objects = this->temp_ffi_objects + num_args;
    this->old_extra_temp_py_objects_stack_top_ = call_stack->extra_temp_py_objects_stack.size();
  }

  ~TVMFFIPyCallContext() {
    TVMFFIPyWithAttachedThreadState thread_state;
    try {
      // recycle the temporary arguments if any
      for (int i = 0; i < this->num_temp_ffi_objects; ++i) {
        TVMFFIObjectDecRef(this->temp_ffi_objects[i]);
      }
      for (int i = 0; i < this->num_temp_py_objects; ++i) {
        Py_DecRef(static_cast<PyObject*>(this->temp_py_objects[i]));
      }
      for (size_t i = old_extra_temp_py_objects_stack_top_;
           i < call_stack->extra_temp_py_objects_stack.size(); ++i) {
        Py_DecRef(static_cast<PyObject*>(call_stack->extra_temp_py_objects_stack[i]));
      }
      call_stack->extra_temp_py_objects_stack.resize(old_extra_temp_py_objects_stack_top_);
    } catch (const std::exception& ex) {
      // very rare, catch c++ exception and set python error
      PyErr_SetString(PyExc_RuntimeError, ex.what());
    }
    // now recycle the memory of the call stack
    if (heap_ptr_ == nullptr) {
      call_stack->args_stack_top = old_args_stack_top_;
    } else {
      delete[] heap_ptr_;
    }
  }

 private:
  /*! \brief the heap pointer */
  TVMFFIAny* heap_ptr_ = nullptr;
  /*! \brief the old stack top */
  size_t old_args_stack_top_;
  /*! \brief the begin index of the temporary Python objects stack */
  size_t old_extra_temp_py_objects_stack_top_;
};

/*! \brief Argument setter for a given python argument. */
struct TVMFFIPyArgSetter {
  /*!
   * \brief Function pointer to invoke the setter.
   * \param self Pointer to this, this should be TVMFFIPyArgSetter*
   * \param call_ctx The call context.
   * \param arg The python argument to be set
   * \param out The output argument.
   * \return 0 on success, -1 on failure. PyError should be set if -1 is returned.
   */
  int (*func)(TVMFFIPyArgSetter* self, TVMFFIPyCallContext* call_ctx, PyObject* arg,
              TVMFFIAny* out);
  /*!
   * \brief Optional DLPackExchangeAPI struct pointer.
   * This is the new struct-based approach that bundles all DLPack exchange functions.
   */
  const DLPackExchangeAPI* dlpack_c_exchange_api{nullptr};
  /*!
   * \brief Invoke the setter.
   * \param call_ctx The call context.
   * \param arg The python argument to be set
   * \param out The output argument.
   * \return 0 on success, -1 on failure. PyError should be set if -1 is returned.
   */
  TVM_FFI_INLINE int operator()(TVMFFIPyCallContext* call_ctx, PyObject* arg,
                                TVMFFIAny* out) const {
    return (*func)(const_cast<TVMFFIPyArgSetter*>(this), call_ctx, arg, out);
  }
};

//---------------------------------------------------------------------------------------------
// The following section contains predefined setters for common POD types
// They ar not meant to be used directly, but instead being registered to TVMFFIPyCallManager
//---------------------------------------------------------------------------------------------
int TVMFFIPyArgSetterFloat_(TVMFFIPyArgSetter*, TVMFFIPyCallContext*, PyObject* arg,
                            TVMFFIAny* out) noexcept {
  out->type_index = kTVMFFIFloat;
  // this function getsdispatched when type is already float, so no need to worry about error
  out->v_float64 = PyFloat_AsDouble(arg);
  return 0;
}

int TVMFFIPyArgSetterInt_(TVMFFIPyArgSetter*, TVMFFIPyCallContext*, PyObject* arg,
                          TVMFFIAny* out) noexcept {
  int overflow = 0;
  out->type_index = kTVMFFIInt;
  out->v_int64 = PyLong_AsLongLongAndOverflow(arg, &overflow);

  if (TVM_FFI_PREDICT_FALSE(overflow != 0)) {
    PyErr_SetString(PyExc_OverflowError, "Python int too large to convert to int64_t");
    return -1;
  }
  return 0;
}

int TVMFFIPyArgSetterBool_(TVMFFIPyArgSetter*, TVMFFIPyCallContext*, PyObject* arg,
                           TVMFFIAny* out) noexcept {
  out->type_index = kTVMFFIBool;
  // this function getsdispatched when type is already bool, so no need to worry about error
  out->v_int64 = PyLong_AsLong(arg);
  return 0;
}

int TVMFFIPyArgSetterNone_(TVMFFIPyArgSetter*, TVMFFIPyCallContext*, PyObject* arg,
                           TVMFFIAny* out) noexcept {
  out->type_index = kTVMFFINone;
  out->v_int64 = 0;
  return 0;
}

//---------------------------------------------------------------------------------------------
// Support for PyCallback function calls.
//---------------------------------------------------------------------------------------------

/*!
 * \brief Context for a C -> Python callback call.
 *
 * Owns a temporary PyObject* array that holds arguments converted from the
 * packed FFI call. Space is first taken from the thread-local args_stack on
 * TVMFFIPyCallStack; if insufficient, we fall back to the heap.
 *
 * Unlike TVMFFIPyCallContext::~TVMFFIPyCallContext, this destructor does NOT
 * attach a thread state — callers are expected to already hold one
 * (e.g. via TVMFFIPyWithAttachedThreadState at the top of the callback).
 *
 * The destructor also decrefs every PyObject* pushed into py_args[0 ..
 * num_active_py_args-1], tracking the pushed count via `num_active_py_args`.
 */
class TVMFFIPyCallbackContext {
 public:
  /*! \brief The temporary PyObject* slots for Python call arguments. */
  PyObject** py_args = nullptr;
  /*! \brief How many slots have a live reference and need decref on cleanup. */
  int32_t num_active_py_args = 0;
  /*! \brief Number of total argument slots allocated. */
  int32_t num_args = 0;

  TVMFFIPyCallbackContext(TVMFFIPyCallStack* call_stack, int32_t num_args)
      : num_args(num_args), call_stack_(call_stack) {
    static_assert(sizeof(TVMFFIAny) % sizeof(PyObject*) == 0);
    // slots needed in the unit of TVMFFIAny
    int64_t slots_needed =
        (static_cast<int64_t>(num_args) * sizeof(PyObject*) + sizeof(TVMFFIAny) - 1) /
        sizeof(TVMFFIAny);
    old_args_stack_top_ = call_stack->args_stack_top;
    if (call_stack->args_stack_top + slots_needed <=
        static_cast<int64_t>(call_stack->args_stack.size())) {
      py_args =
          reinterpret_cast<PyObject**>(call_stack->args_stack.data() + call_stack->args_stack_top);
      call_stack->args_stack_top += slots_needed;
    } else {
      heap_ptr_ = new PyObject*[num_args];
      py_args = heap_ptr_;
    }
  }

  ~TVMFFIPyCallbackContext() {
    // caller must already hold an attached thread state; do NOT re-attach.
    // we ensure that all the pyargs are not null
    for (int32_t i = 0; i < num_active_py_args; ++i) {
      Py_DecRef(py_args[i]);
    }
    if (heap_ptr_ == nullptr) {
      call_stack_->args_stack_top = old_args_stack_top_;
    } else {
      delete[] heap_ptr_;
    }
  }

 private:
  TVMFFIPyCallStack* call_stack_ = nullptr;
  int64_t old_args_stack_top_ = 0;
  PyObject** heap_ptr_ = nullptr;
};

/*!
 * \brief A callback arg setter entry registered to handle efficient callback argument conversion.
 */
struct TVMFFIPyCallbackArgSetter {
  /*!
   * \brief Callback type that converts a borrowed TVMFFIAny (AnyView) to a new-reference PyObject*.
   * \param handle Pointer to the TVMFFIPyCallbackArgSetter (for per-type state).
   * \param dlpack_exchange_api The DLPack exchange API (may be NULL).
   * \param arg The TVMFFIAny value to convert (borrowed; setter must inc if it transfers
   * ownership).
   * \param out Output: a new-reference PyObject*.
   * \return 0 on success, -1 on failure (PyErr set).
   */
  int (*func)(TVMFFIPyCallbackArgSetter* handle, const DLPackExchangeAPI* dlpack_exchange_api,
              const TVMFFIAny* arg, PyObject** out);
};

// common callback arg setters that can be quikcly implemented in C++ and used by cython factory
// note that PyErr is propagated back so we just need to return -1 on failure.
int TVMFFIPyCallbackArgSetterNone_(TVMFFIPyCallbackArgSetter*, const DLPackExchangeAPI*,
                                   const TVMFFIAny*, PyObject** out) noexcept {
  Py_IncRef(Py_None);
  *out = Py_None;
  return 0;
}

int TVMFFIPyCallbackArgSetterBool_(TVMFFIPyCallbackArgSetter*, const DLPackExchangeAPI*,
                                   const TVMFFIAny* arg, PyObject** out) noexcept {
  *out = PyBool_FromLong(static_cast<long>(arg->v_int64));
  return (*out != nullptr) ? 0 : -1;
}

int TVMFFIPyCallbackArgSetterInt_(TVMFFIPyCallbackArgSetter*, const DLPackExchangeAPI*,
                                  const TVMFFIAny* arg, PyObject** out) noexcept {
  *out = PyLong_FromLongLong(arg->v_int64);
  return (*out != nullptr) ? 0 : -1;
}

int TVMFFIPyCallbackArgSetterFloat_(TVMFFIPyCallbackArgSetter*, const DLPackExchangeAPI*,
                                    const TVMFFIAny* arg, PyObject** out) noexcept {
  *out = PyFloat_FromDouble(arg->v_float64);
  return (*out != nullptr) ? 0 : -1;
}

int TVMFFIPyCallbackArgSetterSmallStr_(TVMFFIPyCallbackArgSetter*, const DLPackExchangeAPI*,
                                       const TVMFFIAny* arg, PyObject** out) noexcept {
  TVMFFIByteArray ba = TVMFFISmallBytesGetContentByteArray(arg);
  *out = PyUnicode_DecodeUTF8(ba.data, static_cast<Py_ssize_t>(ba.size), nullptr);
  return (*out != nullptr) ? 0 : -1;
}

int TVMFFIPyCallbackArgSetterSmallBytes_(TVMFFIPyCallbackArgSetter*, const DLPackExchangeAPI*,
                                         const TVMFFIAny* arg, PyObject** out) noexcept {
  TVMFFIByteArray ba = TVMFFISmallBytesGetContentByteArray(arg);
  *out = PyBytes_FromStringAndSize(ba.data, static_cast<Py_ssize_t>(ba.size));
  return (*out != nullptr) ? 0 : -1;
}

///--------------------------------------------------------------------------------
/// Declaring functions defined in Cython to be invoked by the C++ implementation.
/// in all cases PyErr is propagated back so we just need to return -1 on failure.
///--------------------------------------------------------------------------------
/*
 * \brief Set the error raised from Python to the FFI side.
 * \param py_err The Python error to be set.
 * \return 0 on success, -1 on failure. PyError should be set if -1 is returned.
 */
__PYX_EXTERN_C int TVMFFICyErrorSetRaisedFromPyError(PyObject* py_err);
/*
 * \brief Create an argument setter for a given Python argument type.
 * \param arg The Python argument to be set.
 * \param out The output argument setter.
 * \return 0 on success, -1 on failure. PyError should be set if -1 is returned.
 */
__PYX_EXTERN_C int TVMFFICyArgSetterFactory(PyObject* arg, TVMFFIPyArgSetter* out);
/*
 * \brief Create a callback arg setter for a given type index.
 * \param type_index The type index of the argument.
 * \param out The output callback arg setter.
 * \return 0 on success, -1 on failure. PyError should be set if -1 is returned.
 */
__PYX_EXTERN_C int TVMFFICyCallbackArgSetterFactory(int32_t type_index,
                                                    TVMFFIPyCallbackArgSetter* out);
//---------------------------------------------------------------------------------------------
// The function call manager section
//---------------------------------------------------------------------------------------------

/*!
 * \brief A manager class that handles python ffi calls.
 */
class TVMFFIPyCallManager {
 public:
  /*!
   * \brief Get the thread local call manager.
   * \return The thread local call manager.
   */
  static TVMFFIPyCallManager* ThreadLocal() {
    static thread_local TVMFFIPyCallManager inst;
    return &inst;
  }
  /*!
   * \brief Call a function with a variable number of arguments
   * \param func_handle The handle of the function to call
   * \param py_arg_tuple The arguments to the function
   * \param result The result of the function
   * \param c_api_ret_code The return code of the C-call
   * \param release_gil Whether to release the GIL
   * \param optional_out_ctx_dlpack_api The DLPack exchange API to be used for the result
   * \return 0 on when there is no python error, -1 on python error
   * \note When an error happens on FFI side, we should return 0 and set c_api_ret_code
   */
  TVM_FFI_INLINE int FuncCall(void* func_handle, PyObject* py_arg_tuple, TVMFFIAny* result,
                              int* c_api_ret_code, bool release_gil,
                              const DLPackExchangeAPI** optional_out_ctx_dlpack_api) {
    int64_t num_args = PyTuple_Size(py_arg_tuple);
    if (TVM_FFI_PREDICT_FALSE(num_args == -1)) return -1;
    try {
      // allocate a call stack
      TVMFFIPyCallContext ctx(&call_stack_, num_args);
      // Iterate over the arguments and set them
      for (int64_t i = 0; i < num_args; ++i) {
        PyObject* py_arg = PyTuple_GetItem(py_arg_tuple, i);
        TVMFFIAny* c_arg = ctx.packed_args + i;
        if (TVM_FFI_PREDICT_FALSE(SetArgument(&ctx, py_arg, c_arg) != 0)) return -1;
      }
      TVMFFIStreamHandle prev_stream = nullptr;
      DLPackManagedTensorAllocator prev_tensor_allocator = nullptr;
      // setup stream context if needed
      if (ctx.device_type != -1) {
        c_api_ret_code[0] =
            TVMFFIEnvSetStream(ctx.device_type, ctx.device_id, ctx.stream, &prev_stream);
        // setting failed, directly return
        if (TVM_FFI_PREDICT_FALSE(c_api_ret_code[0] != 0)) return 0;
      }
      if (ctx.dlpack_c_exchange_api != nullptr &&
          ctx.dlpack_c_exchange_api->managed_tensor_allocator != nullptr) {
        c_api_ret_code[0] = TVMFFIEnvSetDLPackManagedTensorAllocator(
            ctx.dlpack_c_exchange_api->managed_tensor_allocator, 0, &prev_tensor_allocator);
        if (TVM_FFI_PREDICT_FALSE(c_api_ret_code[0] != 0)) return 0;
      }
      // call the function
      if (release_gil) {
        // release the GIL
        Py_BEGIN_ALLOW_THREADS;
        c_api_ret_code[0] = TVMFFIFunctionCall(func_handle, ctx.packed_args, num_args, result);
        Py_END_ALLOW_THREADS;
      } else {
        c_api_ret_code[0] = TVMFFIFunctionCall(func_handle, ctx.packed_args, num_args, result);
      }
      // restore the original stream
      if (ctx.device_type != -1 && prev_stream != ctx.stream) {
        // always try recover first, even if error happens
        if (TVM_FFI_PREDICT_FALSE(
                TVMFFIEnvSetStream(ctx.device_type, ctx.device_id, prev_stream, nullptr) != 0)) {
          // recover failed, set python error
          PyErr_SetString(PyExc_RuntimeError, "Failed to recover stream");
          return -1;
        }
      }

      if (ctx.dlpack_c_exchange_api != nullptr &&
          prev_tensor_allocator != ctx.dlpack_c_exchange_api->managed_tensor_allocator) {
        // note: we cannot set the error value to c_api_ret_code[0] here because it
        // will be overwritten by the error value from the function call
        if (TVM_FFI_PREDICT_FALSE(
                TVMFFIEnvSetDLPackManagedTensorAllocator(prev_tensor_allocator, 0, nullptr) != 0)) {
          PyErr_SetString(PyExc_RuntimeError, "Failed to recover DLPack managed tensor allocator");
          return -1;
        }
        // return error after
        if (TVM_FFI_PREDICT_FALSE(c_api_ret_code[0] != 0)) return 0;
      }
      if (optional_out_ctx_dlpack_api != nullptr && ctx.dlpack_c_exchange_api != nullptr) {
        *optional_out_ctx_dlpack_api = ctx.dlpack_c_exchange_api;
      }
      return 0;
    } catch (const std::exception& ex) {
      // very rare, catch c++ exception and set python error
      PyErr_SetString(PyExc_RuntimeError, ex.what());
      return -1;
    }
  }

  /*
   * \brief Call a constructor with a variable number of arguments
   *
   * This function is similar to FuncCall, but it will not set the
   * stream and tensor allocator, instead, it will synchronize the TVMFFIPyCallContext
   * with the parent context. This behavior is needed for nested conversion of arguments
   * where detected argument setting needs to be synchronized with final call.
   *
   * This function will also not release  the GIL since constructor call is usually cheap.
   *
   * \param func_handle The handle of the constructor to call
   * \param py_arg_tuple The arguments to the constructor
   * \param result The result of the constructor
   * \param c_api_ret_code The return code of the constructor
   * \param parent_ctx The parent call context to
   * \return 0 on success, -1 on failure
   */
  TVM_FFI_INLINE int ConstructorCall(void* func_handle, PyObject* py_arg_tuple, TVMFFIAny* result,
                                     int* c_api_ret_code, TVMFFIPyCallContext* parent_ctx) {
    int64_t num_args = PyTuple_Size(py_arg_tuple);
    if (TVM_FFI_PREDICT_FALSE(num_args == -1)) return -1;
    try {
      // allocate a call stack
      TVMFFIPyCallContext ctx(&call_stack_, num_args);
      // Iterate over the arguments and set them
      for (int64_t i = 0; i < num_args; ++i) {
        PyObject* py_arg = PyTuple_GetItem(py_arg_tuple, i);
        TVMFFIAny* c_arg = ctx.packed_args + i;
        if (TVM_FFI_PREDICT_FALSE(SetArgument(&ctx, py_arg, c_arg) != 0)) return -1;
      }
      c_api_ret_code[0] = TVMFFIFunctionCall(func_handle, ctx.packed_args, num_args, result);
      // propagate the call context to the parent context
      if (parent_ctx != nullptr) {
        // stream and current device information
        if (parent_ctx->device_type == -1) {
          parent_ctx->device_type = ctx.device_type;
          parent_ctx->device_id = ctx.device_id;
          parent_ctx->stream = ctx.stream;
        }
        // DLPack exchange API
        if (parent_ctx->dlpack_c_exchange_api == nullptr) {
          parent_ctx->dlpack_c_exchange_api = ctx.dlpack_c_exchange_api;
        }
      }
      return 0;
    } catch (const std::exception& ex) {
      // very rare, catch c++ exception and set python error
      PyErr_SetString(PyExc_RuntimeError, ex.what());
      return -1;
    }
  }

  TVM_FFI_INLINE int SetField(void* field_setter, int64_t field_flags, void* field_ptr,
                              PyObject* py_arg, int* c_api_ret_code) {
    try {
      TVMFFIPyCallContext ctx(&call_stack_, 1);
      TVMFFIAny* c_arg = ctx.packed_args;
      if (TVM_FFI_PREDICT_FALSE(SetArgument(&ctx, py_arg, c_arg) != 0)) return -1;
      if (!(field_flags & kTVMFFIFieldFlagBitSetterIsFunctionObj)) {
        auto setter = reinterpret_cast<TVMFFIFieldSetter>(field_setter);
        c_api_ret_code[0] = (*setter)(field_ptr, c_arg);
      } else {
        TVMFFIAny args[2]{};
        args[0].type_index = kTVMFFIOpaquePtr;
        args[0].v_ptr = field_ptr;
        args[1] = *c_arg;
        TVMFFIAny result{};
        result.type_index = kTVMFFINone;
        c_api_ret_code[0] =
            TVMFFIFunctionCall(static_cast<TVMFFIObjectHandle>(field_setter), args, 2, &result);
      }
      return 0;
    } catch (const std::exception& ex) {
      // very rare, catch c++ exception and set python error
      PyErr_SetString(PyExc_RuntimeError, ex.what());
      return -1;
    }
  }

  TVM_FFI_INLINE int PyObjectToFFIAny(PyObject* py_arg, TVMFFIAny* out, int* c_api_ret_code) {
    try {
      TVMFFIPyCallContext ctx(&call_stack_, 1);
      TVMFFIAny* c_arg = ctx.packed_args;
      if (TVM_FFI_PREDICT_FALSE(SetArgument(&ctx, py_arg, c_arg) != 0)) return -1;
      c_api_ret_code[0] = TVMFFIAnyViewToOwnedAny(c_arg, out);
      return 0;
    } catch (const std::exception& ex) {
      // very rare, catch c++ exception and set python error
      PyErr_SetString(PyExc_RuntimeError, ex.what());
      return -1;
    }
  }

  /*!
   * \brief Set an py_arg to out.
   * \param ctx The call context
   * \param py_arg The python argument to be set
   * \param out The output argument
   * \return 0 on success, -1 on failure
   */
  TVM_FFI_INLINE int SetArgument(TVMFFIPyCallContext* ctx, PyObject* py_arg, TVMFFIAny* out) {
    PyTypeObject* py_type = Py_TYPE(py_arg);
    // pre-zero the output argument, modulo the type index
    out->type_index = kTVMFFINone;
    out->zero_padding = 0;
    out->v_int64 = 0;
    // find the pre-cached setter
    // This class is thread-local, so we don't need to worry about race condition
    auto it = arg_dispatch_map_.find(py_type);
    if (TVM_FFI_PREDICT_TRUE(it != arg_dispatch_map_.end())) {
      TVMFFIPyArgSetter setter = it->second;
      // if error happens, propagate it back
      if (TVM_FFI_PREDICT_FALSE(setter(ctx, py_arg, out) != 0)) return -1;
    } else {
      // no dispatch found, query and create a new one.
      TVMFFIPyArgSetter setter;
      // propagate python error back
      if (TVM_FFI_PREDICT_FALSE(TVMFFICyArgSetterFactory(py_arg, &setter) != 0)) {
        return -1;
      }
      // update dispatch table
      arg_dispatch_map_.emplace(py_type, setter);
      if (TVM_FFI_PREDICT_FALSE(setter(ctx, py_arg, out) != 0)) return -1;
    }
    return 0;
  }

  /*!
   * \brief Get the size of the arg dispatch map
   * \return The size of the arg dispatch map
   */
  size_t GetArgDispatchMapSize() const { return arg_dispatch_map_.size(); }

  /*!
   * \brief Convert a borrowed TVMFFIAny (AnyView) into a new-reference PyObject*.
   * \param dlpack_exchange_api The DLPack exchange API (may be NULL).
   * \param arg The borrowed TVMFFIAny to convert.
   * \param py_arg The output PyObject*.
   * \return 0 on success, -1 on failure. PyError should be set if -1 is returned.
   */
  TVM_FFI_INLINE int SetPyCallbackArg(const DLPackExchangeAPI* dlpack_exchange_api,
                                      const TVMFFIAny* arg, PyObject** out) {
    size_t type_index = static_cast<size_t>(arg->type_index);
    // Mirror of SetArgument for the C++ -> Python callback path: each per-type
    // callback arg setter is responsible for its own refcount.
    // hot path: cached hit
    if (type_index < callback_arg_dispatch_table_.size() &&
        callback_arg_dispatch_table_[type_index].func != nullptr) {
      TVMFFIPyCallbackArgSetter* setter = &callback_arg_dispatch_table_[type_index];
      return setter->func(setter, dlpack_exchange_api, arg, out);
    }
    // cold path: grow and populate via factory
    if (type_index >= callback_arg_dispatch_table_.size()) {
      // initialize empty entries with nullptr
      callback_arg_dispatch_table_.resize(type_index + 1, TVMFFIPyCallbackArgSetter{nullptr});
    }
    TVMFFIPyCallbackArgSetter* setter = &callback_arg_dispatch_table_[type_index];
    if (TVMFFICyCallbackArgSetterFactory(static_cast<int32_t>(type_index), setter) != 0) {
      return -1;
    }
    return setter->func(setter, dlpack_exchange_api, arg, out);
  }

  /*!
   * \brief Python Callback function entry
   *
   * \param context The TVMFFIPyCallbackClosure* holding the Python callable
   *                and optional DLPack exchange API.
   * \param packed_args The packed FFI arguments.
   * \param num_args Number of arguments.
   * \param result Output FFI result.
   * \return 0 on success, -1 on error.
   */
  TVM_FFI_INLINE int PyCallback(void* context, const TVMFFIAny* packed_args, int32_t num_args,
                                TVMFFIAny* result) noexcept {
    TVMFFIPyWithAttachedThreadState thread_state;
    auto* closure = static_cast<TVMFFIPyCallbackClosure*>(context);
    // Wrap the body in try/catch so any C++ exception raised by the stack
    // allocator (TVMFFIPyCallbackContext / TVMFFIPyCallContext), dispatch
    // table resize in SetPyCallbackArg, or unordered_map::emplace in
    // SetArgument is converted into a PyErr + FFI error instead of
    // triggering std::terminate via the noexcept contract.
    try {
      TVMFFIPyCallbackContext cb_ctx(&call_stack_, num_args);
      // Step 1: Convert each packed arg (borrowed AnyView) to a PyObject*
      for (int32_t i = 0; i < num_args; ++i) {
        if (TVM_FFI_PREDICT_FALSE(SetPyCallbackArg(closure->dlpack_exchange_api, &packed_args[i],
                                                   &cb_ctx.py_args[i]) != 0)) {
          ForwardPyErrorToFFI();
          return -1;
        }
        // must set active arguments count to ensure correct recycling
        cb_ctx.num_active_py_args = i + 1;
      }
      // Step 2: Call the Python function via vectorcall. Wrap py_result in
      // a RAII guard so its +1 is released on every exit path, including
      // the C++ exception path (e.g., bad_alloc from ret_ctx construction
      // or SetArgument's emplace).
#if PY_VERSION_HEX >= 0x03090000
      PyObject* py_result_raw = PyObject_Vectorcall(closure->callable, cb_ctx.py_args,
                                                    static_cast<size_t>(num_args), nullptr);
#else
      // backward compatibility for Python 3.8
      PyObject* py_result_raw = _PyObject_Vectorcall(closure->callable, cb_ctx.py_args,
                                                     static_cast<size_t>(num_args), nullptr);
#endif
      struct PyResultGuard {
        PyObject* p;
        ~PyResultGuard() {
          if (p != nullptr) Py_DecRef(p);
        }
      } py_result{py_result_raw};
      if (py_result.p == Py_None) {
        // fast path for Py_None
        result->type_index = kTVMFFINone;
        result->zero_padding = 0;
        result->v_int64 = 0;
        return 0;
      } else if (py_result.p != nullptr) {
        // normal return
        // Use SetArgument on a temporary view slot, then promote to owned.
        // Note: SetArgument only borrows py_result's chandle into `view`; it
        // does NOT inc. py_result must stay alive until AFTER
        // TVMFFIAnyViewToOwnedAny has promoted the view to an owned ref,
        // otherwise dec'ing py_result first could free the underlying object
        // (e.g. if py_result owns the last ref to a freshly created tensor).
        // The guard's destructor runs AFTER the return value is computed.
        TVMFFIPyCallContext ret_ctx(&call_stack_, 1);
        TVMFFIAny* view = ret_ctx.packed_args;
        if (TVM_FFI_PREDICT_FALSE(SetArgument(&ret_ctx, py_result.p, view) != 0)) {
          ForwardPyErrorToFFI();
          return -1;
        }
        // TLS FFI error set on failure.
        return TVMFFIAnyViewToOwnedAny(view, result);
      } else {
        // vectorcall failed
        ForwardPyErrorToFFI();
        return -1;
      }
    } catch (const std::exception& ex) {
      // very rare, catch c++ exception and set python error
      PyErr_SetString(PyExc_RuntimeError, ex.what());
      ForwardPyErrorToFFI();
      return -1;
    }
  }

  /*!
   * \brief Fetch the current Python exception and forward it to
   *        TVMFFICyErrorSetRaisedFromPyError, then clear the Python error indicator.
   *
   * This helper correctly extracts the exception *value* (not just the type
   * returned by PyErr_Occurred()) so that set_last_ffi_error can access the
   * message and traceback.
   */
  TVM_FFI_COLD_CODE static void ForwardPyErrorToFFI() noexcept {
#if PY_VERSION_HEX >= 0x030C0000
    // Python 3.12+: PyErr_Fetch / PyErr_NormalizeException are deprecated.
    // PyErr_GetRaisedException returns an already-normalized exception
    // instance and clears the indicator. Traceback is attached as usual.
    PyObject* pvalue = PyErr_GetRaisedException();
    if (pvalue != nullptr) {
      TVMFFICyErrorSetRaisedFromPyError(pvalue);
      Py_DecRef(pvalue);
    }
#else
    // Python 3.9 - 3.11.
    PyObject *ptype, *pvalue, *ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
    if (ptraceback != nullptr) {
      PyException_SetTraceback(pvalue, ptraceback);
    }
    TVMFFICyErrorSetRaisedFromPyError(pvalue);
    Py_DecRef(ptype);
    Py_DecRef(pvalue);
    Py_DecRef(ptraceback);
#endif
  }

 private:
  TVMFFIPyCallManager() {
    static constexpr size_t kDefaultDispatchCapacity = 32;
    arg_dispatch_map_.reserve(kDefaultDispatchCapacity);
    // Pre-allocate callback arg dispatch table for static type indices
    static constexpr size_t kDefaultCallbackArgDispatchCapacity = 128;
    callback_arg_dispatch_table_.resize(kDefaultCallbackArgDispatchCapacity);
  }

  // internal arg dispatch map: type -> argument setter
  std::unordered_map<PyTypeObject*, TVMFFIPyArgSetter> arg_dispatch_map_;
  // call stack
  TVMFFIPyCallStack call_stack_;
  // callback arg setter dispatch table indexed by type_index (view-based path
  // used by PyCallback; see TVMFFIPyCallbackArgSetter docs above)
  std::vector<TVMFFIPyCallbackArgSetter> callback_arg_dispatch_table_;
};

/*!
 * \brief Call a function with a variable number of arguments
 * \param func_handle The handle of the function to call
 * \param py_arg_tuple The arguments to the function
 * \param result The result of the function
 * \param c_api_ret_code The return code of the function
 * \param release_gil Whether to release the GIL
 * \param out_ctx_dlpack_api The DLPack exchange API to be used for the result
 * \return 0 on success, nonzero on failure
 */
TVM_FFI_INLINE int TVMFFIPyFuncCall(void* func_handle, PyObject* py_arg_tuple, TVMFFIAny* result,
                                    int* c_api_ret_code, bool release_gil = true,
                                    const DLPackExchangeAPI** out_ctx_dlpack_api = nullptr) {
  return TVMFFIPyCallManager::ThreadLocal()->FuncCall(
      func_handle, py_arg_tuple, result, c_api_ret_code, release_gil, out_ctx_dlpack_api);
}

/*!
 * \brief Call a constructor function with a variable number of arguments
 *
 * This function is similar to TVMFFIPyFuncCall, but it will not set the
 * stream and tensor allocator. Instead, it will synchronize the TVMFFIPyCallContext
 * with the parent context. This behavior is needed for nested conversion of arguments
 * where detected argument settings need to be synchronized with the final call.
 *
 * This function will also not release the GIL since constructor call is usually cheap.
 *
 * \param func_handle The handle of the function to call
 * \param py_arg_tuple The arguments to the constructor
 * \param result The result of the constructor
 * \param c_api_ret_code The return code of the constructor
 * \param parent_ctx The parent call context
 * \param release_gil Whether to release the GIL
 * \param out_dlpack_exporter The DLPack exporter to be used for the result
 * \return 0 on success, nonzero on failure
 */
TVM_FFI_INLINE int TVMFFIPyConstructorCall(void* func_handle, PyObject* py_arg_tuple,
                                           TVMFFIAny* result, int* c_api_ret_code,
                                           TVMFFIPyCallContext* parent_ctx) {
  return TVMFFIPyCallManager::ThreadLocal()->ConstructorCall(func_handle, py_arg_tuple, result,
                                                             c_api_ret_code, parent_ctx);
}

/*!
 * \brief Set a field of a FFI object
 * \param field_setter The field setter (function pointer or FunctionObj handle)
 * \param field_flags The field flags (to dispatch between function pointer and FunctionObj)
 * \param field_ptr The pointer to the field
 * \param py_arg The python argument to be set
 * \param c_api_ret_code The return code of the function
 * \return 0 on success, nonzero on failure
 */
TVM_FFI_INLINE int TVMFFIPyCallFieldSetter(void* field_setter, int64_t field_flags, void* field_ptr,
                                           PyObject* py_arg, int* c_api_ret_code) {
  return TVMFFIPyCallManager::ThreadLocal()->SetField(field_setter, field_flags, field_ptr, py_arg,
                                                      c_api_ret_code);
}

/*!
 * \brief Set an python argument to a FFI Any using the generic dispatcher in call manager
 * \param ctx The call context
 * \param py_arg_tvm_ffi_value The python argument to be set using the __tvm_ffi_value__ protocol
 * \param out The output argument
 * \return 0 on success, nonzero on failure
 */
TVM_FFI_INLINE int TVMFFIPySetArgumentGenericDispatcher(TVMFFIPyCallContext* ctx,
                                                        PyObject* py_arg_tvm_ffi_value,
                                                        TVMFFIAny* out) {
  return TVMFFIPyCallManager::ThreadLocal()->SetArgument(ctx, py_arg_tvm_ffi_value, out);
}

/*!
 * \brief Convert a Python object to a FFI Any
 * \param py_arg The python argument to be set
 * \param out The output argument
 * \param c_api_ret_code The return code of the function
 * \return 0 on success, nonzero on failure
 */
TVM_FFI_INLINE int TVMFFIPyPyObjectToFFIAny(PyObject* py_arg, TVMFFIAny* out, int* c_api_ret_code) {
  return TVMFFIPyCallManager::ThreadLocal()->PyObjectToFFIAny(py_arg, out, c_api_ret_code);
}

/*!
 * \brief Get the size of the arg dispatch map
 * \return The size of the arg dispatch map
 */
TVM_FFI_INLINE size_t TVMFFIPyGetArgDispatchMapSize() {
  return TVMFFIPyCallManager::ThreadLocal()->GetArgDispatchMapSize();
}

//---------------------------------------------------------------------------------------------
// Free function wrapper for the Python callback path.
// Mirrors the pattern of TVMFFIPyFuncCall / TVMFFIPyConstructorCall: a top-level
// TVM_FFI_INLINE free function that forwards to the thread-local manager.
//---------------------------------------------------------------------------------------------

/*!
 * \brief C-callable Python callback entry point (TVMFFISafeCallType shape).
 *
 * Forwards to TVMFFIPyCallManager::ThreadLocal()->PyCallback. Designed to be
 * installed as the safe_call pointer for FFI functions that wrap a Python
 * callable.
 *
 * \note The `context` argument is interpreted as a TVMFFIPyCallbackClosure*
 *       by the manager (see TVMFFIPyConvertPyCallback).
 */
TVM_FFI_INLINE int TVMFFIPyCallback(void* context, const TVMFFIAny* packed_args, int32_t num_args,
                                    TVMFFIAny* result) noexcept {
  return TVMFFIPyCallManager::ThreadLocal()->PyCallback(context, packed_args, num_args, result);
}

/*!
 * \brief Create an FFI function handle from a Python callable + optional DLPack exchange API.
 *
 * Allocates a TVMFFIPyCallbackClosure on the heap, IncRefs the callable, and
 * registers it with the FFI function-creation API using TVMFFIPyCallback as the
 * safe-call entry point and TVMFFIPyCallbackClosure::Deleter as the deleter.
 *
 * Returns the raw FFI return code (TLS FFI error set on failure). The Cython
 * caller uses CHECK_CALL to translate it into a Python exception.
 *
 * \param callable The Python callable to wrap. Must be non-NULL.
 * \param dlpack_api Optional DLPack exchange API. May be NULL.
 * \param out_handle Destination for the new FFI function handle.
 * \return The return code from TVMFFIFunctionCreate (0 on success).
 */
TVM_FFI_INLINE int TVMFFIPyConvertPyCallback(PyObject* callable,
                                             const DLPackExchangeAPI* dlpack_api,
                                             TVMFFIObjectHandle* out_handle) noexcept {
  // Use nothrow new: plain `new` can throw std::bad_alloc, which in this
  // noexcept function would trigger std::terminate. On allocation failure,
  // set PyErr and return -1 so the Cython caller's CHECK_CALL surfaces it.
  auto* raw = new (std::nothrow) TVMFFIPyCallbackClosure{callable, dlpack_api};
  if (raw == nullptr) {
    PyErr_NoMemory();
    return -1;
  }
  // The callable's +1 is owned by the closure; TVMFFIPyCallbackClosure::Deleter
  // is responsible for Py_DecRef on destruction. By wiring the same Deleter as
  // the unique_ptr deleter, the failure path below (unique_ptr unwind) runs
  // the same cleanup as the success path (invoked by the FFI runtime).
  Py_IncRef(callable);
  std::unique_ptr<TVMFFIPyCallbackClosure, void (*)(void*)> closure(
      raw, &TVMFFIPyCallbackClosure::Deleter);
  int rc = TVMFFIFunctionCreate(closure.get(), &TVMFFIPyCallback, &TVMFFIPyCallbackClosure::Deleter,
                                out_handle);
  // On success, transfer ownership to the FFI function; on failure, let
  // unique_ptr unwind via Deleter (decrefs the callable, frees the closure).
  if (rc == 0) closure.release();
  return rc;
}

/*!
 * \brief Push a temporary FFI object to the call context that will be recycled after the call
 * \param ctx The call context
 * \param arg The FFI object to push
 */
TVM_FFI_INLINE void TVMFFIPyPushTempFFIObject(TVMFFIPyCallContext* ctx,
                                              TVMFFIObjectHandle arg) noexcept {
  // invariance: each ArgSetter can have at most one temporary Python object
  // so it ensures that we won't overflow the temporary Python object stack
  ctx->temp_ffi_objects[ctx->num_temp_ffi_objects++] = arg;
}

/*!
 * \brief Push a temporary Python object to the call context that will be recycled after the call
 * \param ctx The call context
 * \param arg The Python object to push
 */
TVM_FFI_INLINE void TVMFFIPyPushTempPyObject(TVMFFIPyCallContext* ctx, PyObject* arg) noexcept {
  // invariance: each ArgSetter can have at most one temporary Python object
  // so it ensures that we won't overflow the temporary Python object stack
  Py_IncRef(arg);
  ctx->temp_py_objects[ctx->num_temp_py_objects++] = arg;
}

/*!
 * \brief Push Extra temporary Python object to the call context that may go beyond one temp per
 *        argument budget, mainly used by value protocol.
 * \param ctx The call context
 * \param arg The Python object to push
 */
TVM_FFI_INLINE void TVMFFIPyPushExtraTempPyObject(TVMFFIPyCallContext* ctx, PyObject* arg) {
  Py_IncRef(arg);
  ctx->call_stack->extra_temp_py_objects_stack.emplace_back(arg);
}

//----------------------------------------------------------
// Helpers for MLIR redirection
//----------------------------------------------------------
/*!
 * \brief Function specialization that leverages MLIR packed safe call definitions.
 *
 * The MLIR execution engine generates functions that correspond to the packed signature.
 * As of now, it is hard to access the raw extern C function pointer of SafeCall
 * directly when we declare the signature in LLVM dialect.
 *
 * Note that in theory, the MLIR execution engine should be able to support
 * some form of "extern C" feature that directly exposes the function pointers
 * of C-compatible functions with an attribute tag. So we keep this feature
 * in the Python helper layer for now in case the MLIR execution engine supports it in the future.
 *
 * This helper enables us to create ffi::Function from the MLIR packed
 * safe call function pointer instead of following the redirection pattern
 * in `TVMFFIPyMLIRPackedSafeCall::Invoke`.
 *
 * \sa TVMFFIPyMLIRPackedSafeCall::Invoke
 */
class TVMFFIPyMLIRPackedSafeCall {
 public:
  TVMFFIPyMLIRPackedSafeCall(void (*mlir_packed_safe_call)(void**), PyObject* keep_alive_object)
      : mlir_packed_safe_call_(mlir_packed_safe_call), keep_alive_object_(keep_alive_object) {
    if (keep_alive_object_) {
      Py_IncRef(keep_alive_object_);
    }
  }

  ~TVMFFIPyMLIRPackedSafeCall() {
    TVMFFIPyWithAttachedThreadState thread_state;
    if (keep_alive_object_) {
      Py_DecRef(keep_alive_object_);
    }
  }

  static int Invoke(void* func, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* rv) {
    TVMFFIPyMLIRPackedSafeCall* self = reinterpret_cast<TVMFFIPyMLIRPackedSafeCall*>(func);
    int ret_code = 0;
    void* handle = nullptr;
    void* mlir_args[] = {&handle, const_cast<TVMFFIAny**>(&args), &num_args, &rv, &ret_code};
    (*self->mlir_packed_safe_call_)(mlir_args);
    return ret_code;
  }

  static void Deleter(void* self) { delete static_cast<TVMFFIPyMLIRPackedSafeCall*>(self); }

 private:
  void (*mlir_packed_safe_call_)(void**);
  PyObject* keep_alive_object_;
};

/*!
 * \brief Create a TVMFFIPyMLIRPackedSafeCall handle
 * \param mlir_packed_safe_call The MLIR packed safe call function
 * \param keep_alive_object The keep alive object
 * \return The TVMFFIPyMLIRPackedSafeCall object
 */
void* TVMFFIPyMLIRPackedSafeCallCreate(void (*mlir_packed_safe_call)(void**),
                                       PyObject* keep_alive_object) {
  return new TVMFFIPyMLIRPackedSafeCall(mlir_packed_safe_call, keep_alive_object);
}

/*!
 * \brief Call the MLIR packed safe call function
 * \param self The TVMFFIPyMLIRPackedSafeCall object
 * \param args The arguments
 * \param num_args The number of arguments
 * \param rv The result
 * \return The return code
 */
int TVMFFIPyMLIRPackedSafeCallInvoke(void* self, const TVMFFIAny* args, int32_t num_args,
                                     TVMFFIAny* rv) {
  return TVMFFIPyMLIRPackedSafeCall::Invoke(self, args, num_args, rv);
}

/*!
 * \brief Delete the TVMFFIPyMLIRPackedSafeCall object
 * \param self The TVMFFIPyMLIRPackedSafeCall object
 */
void TVMFFIPyMLIRPackedSafeCallDeleter(void* self) {
  return TVMFFIPyMLIRPackedSafeCall::Deleter(self);
}

/*!
 * \brief Deleter for Python objects
 * \param py_obj The Python object to delete
 */
extern "C" void TVMFFIPyObjectDeleter(void* py_obj) noexcept {
  TVMFFIPyWithAttachedThreadState thread_state;
  Py_DecRef(static_cast<PyObject*>(py_obj));
}

/*
 * \brief Dummy target to ensure testing is linked and we can run testcases
 */
extern "C" TVM_FFI_DLL int TVMFFITestingDummyTarget();

#endif  // TVM_FFI_PYTHON_HELPERS_H_
