//===- AMDGPUWaitcntUtils.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Shared waitcnt analysis primitives for AMDGPU passes.
///
/// This header is the public surface of a small "waitcnt estimation toolkit":
/// helpers that let a pass reason about VMEM-counter pending state without
/// owning a full WaitcntBrackets-style scoreboard.  It contains:
///
///   - the InstCounterType enumeration and the Waitcnt value class,
///   - encode/decode helpers for the various s_waitcnt encodings,
///   - VMEM-load classification predicates (which counter does a load
///     increment, is a load "pure" for VMEM-pending purposes, ...),
///   - a small VmemClearedState helper that tracks which VMEM counter
///     classes have been zeroed during a linear instruction scan.
///
/// SIInsertWaitcnts uses these primitives for its VMEM-load classification
/// (see updateVMCntOnly / getVmemType), keeping the bracket-based scoreboard
/// it maintains on top.  Other waitcnt-aware analyses (e.g. DAG mutations)
/// can rely on this header instead of reaching into pass-private code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUWAITCNTUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUWAITCNTUTILS_H

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"

namespace llvm {

class GCNSubtarget;
class MachineInstr;

namespace AMDGPU {

enum InstCounterType {
  LOAD_CNT = 0, // VMcnt prior to gfx12.
  DS_CNT,       // LKGMcnt prior to gfx12.
  EXP_CNT,      //
  STORE_CNT,    // VScnt in gfx10/gfx11.
  NUM_NORMAL_INST_CNTS,
  SAMPLE_CNT = NUM_NORMAL_INST_CNTS, // gfx12+ only.
  BVH_CNT,                           // gfx12+ only.
  KM_CNT,                            // gfx12+ only.
  X_CNT,                             // gfx1250.
  ASYNC_CNT,                         // gfx1250.
  TENSOR_CNT,                        // gfx1250.
  NUM_EXTENDED_INST_CNTS,
  VA_VDST_RD = NUM_EXTENDED_INST_CNTS, // gfx12+ expert mode only.
  VA_VDST_WR,                          // gfx12+ expert mode only.
  VM_VSRC,                             // gfx12+ expert mode only.
  NUM_EXPERT_INST_CNTS,
  NUM_INST_CNTS = NUM_EXPERT_INST_CNTS
};

StringLiteral getInstCounterName(InstCounterType T);

// Return an iterator over all counters between LOAD_CNT (the first counter)
// and \c MaxCounter (exclusive, default value yields an enumeration over
// all counters).
iota_range<InstCounterType>
inst_counter_types(InstCounterType MaxCounter = NUM_INST_CNTS);

/// Represents the hardware counter limits for different wait count types.
struct HardwareLimits {
  unsigned LoadcntMax; // Corresponds to Vmcnt prior to gfx12.
  unsigned ExpcntMax;
  unsigned DscntMax;     // Corresponds to LGKMcnt prior to gfx12.
  unsigned StorecntMax;  // Corresponds to VScnt in gfx10/gfx11.
  unsigned SamplecntMax; // gfx12+ only.
  unsigned BvhcntMax;    // gfx12+ only.
  unsigned KmcntMax;     // gfx12+ only.
  unsigned XcntMax;      // gfx1250.
  unsigned AsyncMax;     // gfx1250.
  unsigned VaVdstMax;    // gfx12+ expert mode only.
  unsigned VmVsrcMax;    // gfx12+ expert mode only.

  HardwareLimits() = default;

  /// Initializes hardware limits from ISA version.
  HardwareLimits(const IsaVersion &IV);

  unsigned get(InstCounterType T) const;
};

} // namespace AMDGPU

template <> struct enum_iteration_traits<AMDGPU::InstCounterType> {
  static constexpr bool is_iterable = true;
};

namespace AMDGPU {

/// Represents the counter values to wait for in an s_waitcnt instruction.
///
/// Large values (including the maximum possible integer) can be used to
/// represent "don't care" waits.
class Waitcnt {
  std::array<unsigned, NUM_INST_CNTS> Cnt;

public:
  unsigned get(InstCounterType T) const { return Cnt[T]; }
  void set(InstCounterType T, unsigned Val) { Cnt[T] = Val; }

  Waitcnt() { fill(Cnt, ~0u); }
  // Pre-gfx12 constructor.
  Waitcnt(unsigned VmCnt, unsigned ExpCnt, unsigned LgkmCnt, unsigned VsCnt)
      : Waitcnt() {
    Cnt[LOAD_CNT] = VmCnt;
    Cnt[EXP_CNT] = ExpCnt;
    Cnt[DS_CNT] = LgkmCnt;
    Cnt[STORE_CNT] = VsCnt;
  }

  // gfx12+ constructor.
  Waitcnt(unsigned LoadCnt, unsigned ExpCnt, unsigned DsCnt, unsigned StoreCnt,
          unsigned SampleCnt, unsigned BvhCnt, unsigned KmCnt, unsigned XCnt,
          unsigned AsyncCnt, unsigned TensorCnt, unsigned VaVdstRd,
          unsigned VaVdstWr, unsigned VmVsrc)
      : Waitcnt() {
    Cnt[LOAD_CNT] = LoadCnt;
    Cnt[DS_CNT] = DsCnt;
    Cnt[EXP_CNT] = ExpCnt;
    Cnt[STORE_CNT] = StoreCnt;
    Cnt[SAMPLE_CNT] = SampleCnt;
    Cnt[BVH_CNT] = BvhCnt;
    Cnt[KM_CNT] = KmCnt;
    Cnt[X_CNT] = XCnt;
    Cnt[ASYNC_CNT] = AsyncCnt;
    Cnt[TENSOR_CNT] = TensorCnt;
    Cnt[VA_VDST_RD] = VaVdstRd;
    Cnt[VA_VDST_WR] = VaVdstWr;
    Cnt[VM_VSRC] = VmVsrc;
  }

  bool hasWait() const {
    return any_of(Cnt, [](unsigned Val) { return Val != ~0u; });
  }

  bool hasWaitExceptStoreCnt() const {
    for (InstCounterType T : inst_counter_types()) {
      if (T == STORE_CNT)
        continue;
      if (Cnt[T] != ~0u)
        return true;
    }
    return false;
  }

  void add(AMDGPU::InstCounterType T, unsigned Count) {
    set(T, std::min(get(T), Count));
  }

  void clear(AMDGPU::InstCounterType T) { set(T, ~0u); }

  bool hasWaitStoreCnt() const { return Cnt[STORE_CNT] != ~0u; }

  bool hasWaitDepctr() const {
    return Cnt[VA_VDST_RD] != ~0u || Cnt[VA_VDST_WR] != ~0u ||
           Cnt[VM_VSRC] != ~0u;
  }

  Waitcnt combined(const Waitcnt &Other) const {
    // Does the right thing provided self and Other are either both pre-gfx12
    // or both gfx12+.
    Waitcnt Wait;
    for (InstCounterType T : inst_counter_types())
      Wait.Cnt[T] = std::min(Cnt[T], Other.Cnt[T]);
    return Wait;
  }

  void print(raw_ostream &OS) const {
    ListSeparator LS;
    for (InstCounterType T : inst_counter_types())
      OS << LS << getInstCounterName(T) << ": " << Cnt[T];
    if (LS.unused())
      OS << "none";
    OS << '\n';
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

  friend raw_ostream &operator<<(raw_ostream &OS, const AMDGPU::Waitcnt &Wait) {
    Wait.print(OS);
    return OS;
  }
};

Waitcnt decodeWaitcnt(const IsaVersion &Version, unsigned Encoded);

unsigned encodeWaitcnt(const IsaVersion &Version, const Waitcnt &Decoded);

// The following are only meaningful on targets that support
// S_WAIT_LOADCNT_DSCNT and S_WAIT_STORECNT_DSCNT.

/// \returns Decoded Waitcnt structure from given \p LoadcntDscnt for given
/// isa \p Version.
Waitcnt decodeLoadcntDscnt(const IsaVersion &Version, unsigned LoadcntDscnt);

/// \returns Decoded Waitcnt structure from given \p StorecntDscnt for given
/// isa \p Version.
Waitcnt decodeStorecntDscnt(const IsaVersion &Version, unsigned StorecntDscnt);

/// \returns \p Loadcnt and \p Dscnt components of \p Decoded  encoded as an
/// immediate that can be used with S_WAIT_LOADCNT_DSCNT for given isa
/// \p Version.
unsigned encodeLoadcntDscnt(const IsaVersion &Version, const Waitcnt &Decoded);

/// \returns \p Storecnt and \p Dscnt components of \p Decoded  encoded as an
/// immediate that can be used with S_WAIT_STORECNT_DSCNT for given isa
/// \p Version.
unsigned encodeStorecntDscnt(const IsaVersion &Version, const Waitcnt &Decoded);

/// Determine if \p MI is a gfx12+ single-counter S_WAIT_*CNT instruction,
/// and if so, which counter it is waiting on.
std::optional<AMDGPU::InstCounterType> counterTypeForInstr(unsigned Opcode);

//===----------------------------------------------------------------------===//
// VMEM-load classification primitives.
//
// Canonical home for the small predicates that answer "what kind of VMEM
// load is this?".  SIInsertWaitcnts uses these helpers directly (see
// updateVMCntOnly / getVmemType in SIInsertWaitcnts.cpp), and other passes
// that want lightweight waitcnt awareness can do the same without having
// to reach into SIInsertWaitcnts internals.
//===----------------------------------------------------------------------===//

/// \returns true iff \p MI increments only VMEM-class counters
/// (loadcnt/samplecnt/bvhcnt/storecnt; vmcnt/vscnt on pre-GFX12).
/// Includes BUF, image, and segment-specific FLAT; excludes generic FLAT.
bool updateVMCntOnly(const MachineInstr &MI);

/// \returns the VMEM completion family of \p MI: BVH_CNT for BVH images,
/// SAMPLE_CNT for sampler/MSAA images, LOAD_CNT otherwise.
/// \pre updateVMCntOnly(MI) must be true.
InstCounterType getVmemFamily(const MachineInstr &MI);

/// \returns the VMEM hardware counter \p MI increments under \p ST.
/// Pre-GFX12: always LOAD_CNT.  GFX12+: BVH images -> BVH_CNT,
/// sampler/MSAA images -> SAMPLE_CNT, everything else -> LOAD_CNT.
InstCounterType getVmemLoadCounter(const MachineInstr &MI,
                                   const GCNSubtarget &ST);

/// \returns true iff a VMEM counter alone is sufficient to wait for \p MI's
/// result.  This is updateVMCntOnly() plus generic FLAT loads on subtargets
/// where \see GCNSubtarget::hasFlatLgkmVMemCountInOrder holds.
bool isVmemCounterLoad(const MachineInstr &MI, const GCNSubtarget &ST);

/// \returns true iff \p MI is a non-store, side-effect-free VMEM load that
/// completes on a VMEM counter alone, with no pseudo-source memory operands
/// (i.e. not a spill reload).  For BUNDLE: every member of the bundle must
/// independently satisfy the same criteria.
bool isPureVMemLoad(const MachineInstr &MI, const GCNSubtarget &ST);

//===----------------------------------------------------------------------===//
// VmemClearedState
//
// Tracks which VMEM counter classes have been waited to zero during a linear
// scan of instructions.  Pre-GFX12 has a single unified VMEM counter
// (LOAD_CNT, encoded as the vmcnt field of S_WAITCNT); GFX12+ splits into
// LOAD_CNT, SAMPLE_CNT, and BVH_CNT, each with its own S_WAIT_*CNT opcode.
//
// Typical usage is to walk a basic-block range and feed each instruction to
// update(); the per-counter flags then answer "has there been a wait that
// fully cleared this counter so far in the scan?".  The class only models
// "wait to zero" (the interesting case for VMEM-pending classification);
// it does not track partial bracket levels the way WaitcntBrackets does.
//===----------------------------------------------------------------------===//
struct VmemClearedState {
  bool LoadCnt = false;
  bool SampleCnt = false;
  bool BvhCnt = false;

  /// Update internal state if \p MI is an s_waitcnt-family instruction that
  /// drives a VMEM counter to zero.  Non-wait instructions are ignored.
  void update(const MachineInstr &MI, const GCNSubtarget &ST);

  /// True if a wait observed by update() drove counter \p CT to zero.
  /// Returns false for non-VMEM counter types.
  bool clears(InstCounterType CT) const;

  /// True if any VMEM-class counter has been cleared so far.
  bool anyCleared() const { return LoadCnt || SampleCnt || BvhCnt; }

  /// True if every VMEM counter relevant on \p ST has been cleared.
  bool allCleared(const GCNSubtarget &ST) const;
};

} // namespace AMDGPU

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUWAITCNTUTILS_H
