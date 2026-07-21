//===-- AMDGPUVMemUseOrdering.cpp - AMDGPU VMEM Use Ordering --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Post-RA DAG mutation that keeps consumers of VMEM-pending registers
///       from being hoisted ahead of in-region VMEM loads.
///
///       A register unit is "VMEM-pending" at region entry if its closest
///       reaching definition (found by a depth-bounded backward walk) is a
///       pure VMEM load.  Hoisting such a consumer to the top of the region
///       forces SIInsertWaitcnts to emit an early wait that serialises the
///       in-region loads.
///
///       The mutation adds Artificial order edges from each in-region VMEM
///       load to every in-region consumer of a pending unit on a matching
///       counter class (loadcnt/samplecnt/bvhcnt on GFX12+, unified vmcnt
///       before), so a single later partial wait covers both.
///
///       Requires RemoveKillFlags=true (GCNPostScheduleDAGMILive sets it).
///       Registered only post-RA (createPostMachineScheduler): it relies on
///       physreg live-ins and defs that pre-RA schedulers do not have.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUVMemUseOrdering.h"
#include "AMDGPUWaitcntUtils.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-vmem-use-ordering"

// TODO: 8 was chosen conservatively; profiling on real shader workloads may
// show a lower value is sufficient and reduces compile-time for deep CFGs.
static cl::opt<unsigned> VMemUseOrderingMaxDepth(
    "amdgpu-vmem-use-ordering-max-depth",
    cl::desc("Maximum predecessor-block depth searched when classifying "
             "VMEM-pending registers"),
    cl::init(8), cl::Hidden);

// A region entered with many already-outstanding VMEM loads on one counter
// class is counter-bound: SIInsertWaitcnts emits an in-order descending
// partial-wait staircase that is already close to optimal.  Adding
// load->consumer edges on such a class hoists the region's own loads above
// the consumer wall, co-mingling them with the deep live-in counter and
// collapsing staircase rungs into a full drain.  Above this per-class
// pending-unit count we therefore leave the class alone.  0 disables the gate.
static cl::opt<unsigned> VMemUseOrderingMaxPendingDepth(
    "amdgpu-vmem-use-ordering-max-pending-depth",
    cl::desc("Skip adding VMEM-load ordering edges for a counter class once "
             "more than this many register units are already VMEM-pending at "
             "region entry on that class (0 disables the gate)"),
    cl::init(8), cl::Hidden);

namespace {

// True iff Reg is a VGPR/AGPR - the only regs a pure VMEM load can write, and
// hence the only reg units that can ever be VMEM-pending.  Seeding non-vector
// candidates would only make classify() walk scalar live-ins to full depth for
// units that can never be marked pending.
static bool isVectorPhysReg(MCRegister Reg, const TargetRegisterInfo &TRI) {
  const TargetRegisterClass *RC = TRI.getPhysRegBaseClass(Reg);
  return RC &&
         (SIRegisterInfo::isVGPRClass(RC) || SIRegisterInfo::isAGPRClass(RC));
}

class VMemUseOrdering : public ScheduleDAGMutation {
  const GCNSubtarget &ST;

  // Scratch bit-sets reused across regions (grown lazily, reset per region) to
  // avoid per-region heap traffic.  Indexed by the key named in each comment.
  BitVector VisitedMBBs;   // [MBB number]  block already seen by classify()
  BitVector DfsBV;         // [SUnit NodeNum] node seen by leaf-pruning DFS
  BitVector CandsBV;       // [reg unit]    candidate use awaiting classify
  BitVector DefinedBV;     // [reg unit]    defined somewhere in the region
  BitVector PendingBV[AMDGPU::NUM_INST_CNTS]; // [reg unit] pending, per counter class
  BitVector PendingAnyBV; // [reg unit] union of PendingBV: pending on any class

  // MCRegUnit is enum class : unsigned; BitVector needs a plain unsigned index.
  static unsigned ruIdx(MCRegUnit U) { return static_cast<unsigned>(U); }

  void growSU(unsigned N) {
    if (DfsBV.size() < N)
      DfsBV.resize(N);
  }
  void growRU(unsigned N) {
    if (CandsBV.size() < N)
      CandsBV.resize(N);
    if (DefinedBV.size() < N)
      DefinedBV.resize(N);
    for (BitVector &BV : PendingBV)
      if (BV.size() < N)
        BV.resize(N);
    if (PendingAnyBV.size() < N)
      PendingAnyBV.resize(N);
  }

  /// Mark reg unit \p U as VMEM-pending on counter class \p Cls: the first
  /// class seen sticks; a later different class makes the unit ambiguous (set
  /// in every VMEM class so it matches any counter); the same class is a no-op.
  void markPending(unsigned U, AMDGPU::InstCounterType Cls) {
    if (!PendingAnyBV.test(U)) { // first reaching VMEM def: its class sticks
      PendingAnyBV.set(U);
      PendingBV[Cls].set(U);
      return;
    }
    bool InL = PendingBV[AMDGPU::LOAD_CNT].test(U);
    bool InS = PendingBV[AMDGPU::SAMPLE_CNT].test(U);
    bool InB = PendingBV[AMDGPU::BVH_CNT].test(U);
    bool OnlyThis = (Cls == AMDGPU::LOAD_CNT && InL && !InS && !InB) ||
                    (Cls == AMDGPU::SAMPLE_CNT && InS && !InL && !InB) ||
                    (Cls == AMDGPU::BVH_CNT && InB && !InL && !InS);
    if (OnlyThis)
      return;
    PendingBV[AMDGPU::LOAD_CNT].set(U);
    PendingBV[AMDGPU::SAMPLE_CNT].set(U);
    PendingBV[AMDGPU::BVH_CNT].set(U);
  }

  /// Counter class completed by a pure VMEM load. getVmemLoadCounter treats a
  /// BUNDLE header as LOAD_CNT, so for a clause use its first VMEM-load member.
  AMDGPU::InstCounterType loadCounter(const MachineInstr &MI) const {
    if (!MI.isBundle())
      return AMDGPU::getVmemLoadCounter(MI, ST);
    for (auto It = std::next(MI.getIterator()), E = MI.getParent()->instr_end();
         It != E && It->isBundledWithPred(); ++It) {
      if (It->isMetaInstruction())
        continue;
      if (AMDGPU::isVmemCounterLoad(*It, ST))
        return AMDGPU::getVmemLoadCounter(*It, ST);
    }
    return AMDGPU::LOAD_CNT;
  }

  /// Backward walk from \p Stop in \p MBB, then predecessors (depth-bounded,
  /// each block visited once via VisitedMBBs).  Resolves each candidate in
  /// CandsBV at its closest reaching reference: a pure VMEM-load def marks it
  /// pending (markPending); any other def, or a use, just drops it.  CandsBV /
  /// CandsCount are backtracked on return so sibling paths see the original
  /// set; PendingBV and VisitedMBBs accumulate across paths.
  void classify(const MachineBasicBlock *MBB,
                MachineBasicBlock::const_iterator Stop, unsigned &CandsCount,
                const TargetRegisterInfo &TRI, unsigned Depth) {
    SmallVector<MCRegUnit, 16> Resolved; // restored on return (backtracking)
    for (auto It = MachineBasicBlock::const_reverse_iterator(Stop),
              E = MBB->rend();
         It != E && CandsCount > 0; ++It) {
      const MachineInstr &MI = *It;
      if (MI.isMetaInstruction())
        continue;
      // Classification is only needed once this instruction actually resolves
      // a candidate def, so defer it (most scanned instructions resolve none).
      int IsVMem = -1; // -1 unknown, 0 no, 1 yes
      AMDGPU::InstCounterType Cls = AMDGPU::LOAD_CNT;
      // Defs first: a pure VMEM-load def leaves the unit pending in its class.
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.isDef() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit U : TRI.regunits(MO.getReg().asMCReg())) {
          if (!CandsBV.test(ruIdx(U)))
            continue;
          if (IsVMem < 0) {
            IsVMem = AMDGPU::isPureVMemLoad(MI, ST) ? 1 : 0;
            if (IsVMem)
              Cls = loadCounter(MI);
          }
          CandsBV.reset(ruIdx(U));
          --CandsCount;
          Resolved.push_back(U);
          if (IsVMem)
            markPending(ruIdx(U), Cls);
        }
      }
      // A use reached before any def means the value was consumed (waited)
      // pre-region.  Defs run first, so a read-modify-write is classified by
      // its def.
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || MO.isDef() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit U : TRI.regunits(MO.getReg().asMCReg())) {
          if (!CandsBV.test(ruIdx(U)))
            continue;
          CandsBV.reset(ruIdx(U));
          --CandsCount;
          Resolved.push_back(U);
        }
      }
    }
    if (CandsCount > 0 && Depth < VMemUseOrderingMaxDepth) {
      for (const MachineBasicBlock *Pred : MBB->predecessors()) {
        unsigned PredNum = static_cast<unsigned>(Pred->getNumber());
        if (!VisitedMBBs.test(PredNum)) {
          VisitedMBBs.set(PredNum);
          classify(Pred, Pred->end(), CandsCount, TRI, Depth + 1);
        }
      }
    }
    // Backtrack so sibling predecessors and the caller see the original set.
    for (MCRegUnit U : Resolved) {
      CandsBV.set(ruIdx(U));
      ++CandsCount;
    }
  }

public:
  VMemUseOrdering(MachineFunction *MF) : ST(MF->getSubtarget<GCNSubtarget>()) {}

  void apply(ScheduleDAGInstrs *DAG) override {
    // EntrySU/ExitSU are not in DAG->SUnits, so every element has a real MI.
    if (DAG->SUnits.empty())
      return;
    const TargetRegisterInfo *TRI = DAG->TRI;

    // In-region pure VMEM loads and the counter class each completes.  With
    // none there is nothing to sink consumers past.
    SmallVector<std::pair<SUnit *, AMDGPU::InstCounterType>, 8> VMemLoads;
    for (SUnit &SU : DAG->SUnits) {
      const MachineInstr *MI = SU.getInstr();
      if (MI && AMDGPU::isPureVMemLoad(*MI, ST))
        VMemLoads.push_back({&SU, loadCounter(*MI)});
    }
    if (VMemLoads.empty())
      return;

    // Leaf pruning: drop load A if its successor cone reaches another in-region
    // load B of the same class, since B -> consumer already implies
    // A -> consumer.  DfsBV (reused) tracks the cone; LoadClass keeps each
    // load's ORIGINAL class so pruning one never hides it as a stand-in for
    // another.  A cross-class load cannot stand in for A.
    if (VMemLoads.size() > 1) {
      growSU(DAG->SUnits.size());
      SmallDenseMap<unsigned, AMDGPU::InstCounterType, 8> LoadClass;
      for (auto &[L, C] : VMemLoads)
        LoadClass[L->NodeNum] = C;
      SmallVector<SUnit *, 16> Stack;
      llvm::erase_if(
          VMemLoads, [&](const std::pair<SUnit *, AMDGPU::InstCounterType> &LC) {
            DfsBV.reset();
            Stack.clear();
            for (const SDep &D : LC.first->Succs)
              if (D.getSUnit()->NodeNum < DfsBV.size())
                Stack.push_back(D.getSUnit());
            while (!Stack.empty()) {
              SUnit *S = Stack.pop_back_val();
              if (DfsBV.test(S->NodeNum))
                continue;
              DfsBV.set(S->NodeNum);
              auto It = LoadClass.find(S->NodeNum);
              if (It != LoadClass.end() && It->second == LC.second)
                return true;
              for (const SDep &D : S->Succs)
                if (D.getSUnit()->NodeNum < DfsBV.size())
                  Stack.push_back(D.getSUnit());
            }
            return false;
          });
    }

    // Seed candidates from physreg uses read in-region (nothing else can have a
    // consumer here), minus units also defined in-region (those read the
    // in-region producer, which already carries a Data dep - so a live-in read
    // preceding an in-region redef is conservatively missed).  MCRegUnit gives
    // uniform sub-register aliasing.  CandsBV/DefinedBV are reused bit-sets.
    const MachineBasicBlock *MBB = DAG->begin()->getParent();
    growRU(TRI->getNumRegUnits());
    CandsBV.reset();
    DefinedBV.reset();
    for (SUnit &SU : DAG->SUnits) {
      const MachineInstr *MI = SU.getInstr();
      if (!MI || MI->isMetaInstruction())
        continue;
      for (const MachineOperand &MO : MI->operands()) {
        if (!MO.isReg() || !MO.getReg().isPhysical())
          continue;
        MCRegister Reg = MO.getReg().asMCReg();
        if (MO.isDef())
          for (MCRegUnit U : TRI->regunits(Reg))
            DefinedBV.set(ruIdx(U));
        else if (isVectorPhysReg(Reg, *TRI)) // only vector units can be pending
          for (MCRegUnit U : TRI->regunits(Reg))
            CandsBV.set(ruIdx(U));
      }
    }
    CandsBV.reset(DefinedBV); // CandsBV &= ~DefinedBV
    unsigned CandsCount = CandsBV.count();
    if (CandsCount == 0)
      return;

    // Classify each candidate by its closest reaching definition into the
    // per-class PendingBV bit-sets (an ambiguous unit ends up set in every
    // class).  VisitedMBBs is sized to the whole function's block count.
    for (BitVector &BV : PendingBV)
      BV.reset();
    PendingAnyBV.reset();
    unsigned NumBlocks = MBB->getParent()->getNumBlockIDs();
    if (VisitedMBBs.size() < NumBlocks)
      VisitedMBBs.resize(NumBlocks);
    VisitedMBBs.reset();
    VisitedMBBs.set(static_cast<unsigned>(MBB->getNumber()));
    classify(MBB, DAG->begin(), CandsCount, *TRI, 0);
    if (PendingAnyBV.none())
      return;

    // Region-entry pending depth per VMEM counter class (an ambiguous unit is
    // set in every class, so it counts toward each).
    unsigned PendingDepth[AMDGPU::NUM_INST_CNTS] = {};
    for (unsigned C = 0; C != AMDGPU::NUM_INST_CNTS; ++C)
      PendingDepth[C] = PendingBV[C].count();

    LLVM_DEBUG({
      dbgs() << "VMemUseOrdering: pending units by class in "
             << MBB->getFullName() << "\n";
      for (unsigned C = 0; C != AMDGPU::NUM_INST_CNTS; ++C)
        if (PendingDepth[C])
          dbgs() << "  class " << C << ": " << PendingDepth[C]
                 << (VMemUseOrderingMaxPendingDepth &&
                             PendingDepth[C] > VMemUseOrderingMaxPendingDepth
                         ? " (saturated)"
                         : "")
                 << "\n";
    });

    // Gate (see VMemUseOrderingMaxPendingDepth): a load whose counter class is
    // already deep at region entry can never contribute an edge, so drop it
    // once here rather than re-testing the invariant per consumer below.
    if (VMemUseOrderingMaxPendingDepth)
      llvm::erase_if(
          VMemLoads,
          [&](const std::pair<SUnit *, AMDGPU::InstCounterType> &LC) {
            return PendingDepth[LC.second] > VMemUseOrderingMaxPendingDepth;
          });
    if (VMemLoads.empty())
      return;

    // Bucket the surviving loads by counter class so each consumer visits only
    // the loads on the classes it waits on (pre-GFX12 collapses to one class).
    SmallVector<SUnit *, 8> LoadsByClass[AMDGPU::NUM_INST_CNTS];
    for (auto &[L, C] : VMemLoads)
      LoadsByClass[C].push_back(L);

    // For each SUnit reading a pending unit, add Artificial order edges from
    // the matching in-region loads.  addEdge() drops self- and cycle-forming
    // edges; redundant edges are harmless (Artificial latency is 0).
    const unsigned AllMask = (1u << AMDGPU::LOAD_CNT) |
                             (1u << AMDGPU::SAMPLE_CNT) | (1u << AMDGPU::BVH_CNT);
    for (SUnit &SU : DAG->SUnits) {
      MachineInstr *MI = SU.getInstr();
      if (!MI || MI->isMetaInstruction())
        continue;
      // Counter classes this SU waits on via pending units.  An ambiguous unit
      // is set in every class, so it selects all buckets; pre-GFX12 this
      // reduces to "reads any pending unit".
      unsigned ClassMask = 0;
      for (const MachineOperand &MO : MI->uses()) {
        if (!MO.isReg() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit U : TRI->regunits(MO.getReg().asMCReg())) {
          unsigned u = ruIdx(U);
          if (!PendingAnyBV.test(u)) // fast reject: most units aren't pending
            continue;
          for (unsigned C = 0; C != AMDGPU::NUM_INST_CNTS; ++C)
            if (PendingBV[C].test(u))
              ClassMask |= 1u << C;
        }
        if (ClassMask == AllMask) // cannot gain more classes; stop scanning
          break;
      }
      if (!ClassMask)
        continue;
      [[maybe_unused]] unsigned Added = 0;
      for (unsigned C = 0; C != AMDGPU::NUM_INST_CNTS; ++C) {
        if (!(ClassMask & (1u << C)))
          continue;
        for (SUnit *L : LoadsByClass[C])
          if (L != &SU && DAG->addEdge(&SU, SDep(L, SDep::Artificial)))
            ++Added;
      }
      LLVM_DEBUG(if (Added) dbgs()
                 << "VMemUseOrdering: added " << Added << " edge(s) to SU("
                 << SU.NodeNum << ") " << *MI);
    }
  }
};

} // end namespace

std::unique_ptr<ScheduleDAGMutation>
llvm::createAMDGPUVMemUseOrderingDAGMutation(MachineFunction *MF) {
  return std::make_unique<VMemUseOrdering>(MF);
}
