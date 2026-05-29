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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
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

namespace {

class VMemUseOrdering : public ScheduleDAGMutation {
  const GCNSubtarget &ST;

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

  /// Backward walker: scan \p MBB upward from just before \p Stop, then recurse
  /// into predecessors (depth-bounded; \p Visited scans each block once). For a
  /// unit in \p Cands, its closest reaching reference resolves it: a pure
  /// VMEM-load def records it in \p Pending under the class it completes; any
  /// other def, or a use (value already consumed and thus waited), drops it.
  /// Explicit s_waitcnts are not modeled, so an already-waited value may stay
  /// pending - harmless, only adds latency-0 edges. Conflicting classes across
  /// paths map to NUM_INST_CNTS ("any").
  void classify(const MachineBasicBlock *MBB,
                MachineBasicBlock::const_iterator Stop,
                SmallDenseSet<MCRegUnit, 16> Cands,
                const TargetRegisterInfo &TRI,
                SmallDenseMap<MCRegUnit, AMDGPU::InstCounterType, 16> &Pending,
                SmallPtrSetImpl<const MachineBasicBlock *> &Visited,
                unsigned Depth) {
    for (auto It = MachineBasicBlock::const_reverse_iterator(Stop),
              E = MBB->rend();
         It != E && !Cands.empty(); ++It) {
      const MachineInstr &MI = *It;
      if (MI.isMetaInstruction())
        continue;
      bool IsVMem = AMDGPU::isPureVMemLoad(MI, ST);
      AMDGPU::InstCounterType Cls = IsVMem ? loadCounter(MI) : AMDGPU::LOAD_CNT;
      SmallVector<MCRegUnit, 4> Resolved;
      // Defs first: a pure VMEM-load def leaves the unit pending in its class.
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.isDef() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit U : TRI.regunits(MO.getReg().asMCReg())) {
          if (!Cands.count(U))
            continue;
          Resolved.push_back(U);
          if (IsVMem) {
            auto [It2, Inserted] = Pending.try_emplace(U, Cls);
            if (!Inserted && It2->second != Cls)
              It2->second = AMDGPU::NUM_INST_CNTS; // ambiguous: match any class
          }
        }
      }
      for (MCRegUnit U : Resolved)
        Cands.erase(U);
      // A use reached before any def means the value was consumed (waited)
      // pre-region.  Defs run first, so a read-modify-write is classified by
      // its def.
      Resolved.clear();
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || MO.isDef() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit U : TRI.regunits(MO.getReg().asMCReg()))
          if (Cands.count(U))
            Resolved.push_back(U);
      }
      for (MCRegUnit U : Resolved)
        Cands.erase(U);
    }
    if (Cands.empty() || Depth >= VMemUseOrderingMaxDepth)
      return;
    for (const MachineBasicBlock *Pred : MBB->predecessors())
      if (Visited.insert(Pred).second)
        classify(Pred, Pred->end(), Cands, TRI, Pending, Visited, Depth + 1);
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
    // A -> consumer.  Must stay within a class; a cross-class load cannot stand
    // in for A.
    if (VMemLoads.size() > 1) {
      SmallDenseMap<SUnit *, AMDGPU::InstCounterType, 8> LoadClass;
      for (auto &[L, C] : VMemLoads)
        LoadClass[L] = C;
      SmallPtrSet<SUnit *, 32> Visited;
      SmallVector<SUnit *, 16> Stack;
      llvm::erase_if(VMemLoads,
                     [&](std::pair<SUnit *, AMDGPU::InstCounterType> &LC) {
                       Stack.clear();
                       Visited.clear();
                       for (const SDep &D : LC.first->Succs)
                         Stack.push_back(D.getSUnit());
                       while (!Stack.empty()) {
                         SUnit *S = Stack.pop_back_val();
                         if (!Visited.insert(S).second)
                           continue;
                         auto It = LoadClass.find(S);
                         if (It != LoadClass.end() && It->second == LC.second)
                           return true;
                         for (const SDep &D : S->Succs)
                           Stack.push_back(D.getSUnit());
                       }
                       return false;
                     });
    }

    // Seed candidates from physreg uses read in-region (nothing else can have a
    // consumer here), minus units also defined in-region (those read the
    // in-region producer, which already carries a Data dep - so a live-in read
    // preceding an in-region redef is conservatively missed).  MCRegUnit gives
    // uniform sub-register aliasing.
    SmallDenseSet<MCRegUnit, 16> Cands;
    SmallDenseSet<MCRegUnit, 16> Defined;
    for (SUnit &SU : DAG->SUnits) {
      const MachineInstr *MI = SU.getInstr();
      if (!MI || MI->isMetaInstruction())
        continue;
      for (const MachineOperand &MO : MI->operands()) {
        if (!MO.isReg() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit U : TRI->regunits(MO.getReg().asMCReg())) {
          if (MO.isDef())
            Defined.insert(U);
          else
            Cands.insert(U);
        }
      }
    }
    for (MCRegUnit U : Defined)
      Cands.erase(U);
    if (Cands.empty())
      return;

    // Classify each candidate by its closest reaching definition.
    const MachineBasicBlock *MBB = DAG->begin()->getParent();
    SmallDenseMap<MCRegUnit, AMDGPU::InstCounterType, 16> Pending;
    SmallPtrSet<const MachineBasicBlock *, 16> Visited;
    Visited.insert(MBB);
    classify(MBB, DAG->begin(), std::move(Cands), *TRI, Pending, Visited, 0);
    if (Pending.empty())
      return;

    LLVM_DEBUG({
      dbgs() << "VMemUseOrdering: " << Pending.size() << " pending unit(s) in "
             << MBB->getFullName() << "\n";
      for (const auto &[U, C] : Pending)
        dbgs() << "  pending: " << printRegUnit(U, TRI) << " (class "
               << static_cast<unsigned>(C) << ")\n";
    });

    // For each SUnit reading a pending unit, add Artificial order edges from
    // the matching in-region loads.  addEdge() drops self- and cycle-forming
    // edges; redundant edges are harmless (Artificial latency is 0).
    for (SUnit &SU : DAG->SUnits) {
      MachineInstr *MI = SU.getInstr();
      if (!MI || MI->isMetaInstruction())
        continue;
      // Counter classes this SU waits on via pending units.  Pre-GFX12 all are
      // LOAD_CNT, so this reduces to "reads any pending unit".
      SmallDenseSet<unsigned, 4> Classes;
      bool AnyClass = false;
      for (const MachineOperand &MO : MI->uses()) {
        if (!MO.isReg() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit U : TRI->regunits(MO.getReg().asMCReg())) {
          auto It = Pending.find(U);
          if (It == Pending.end())
            continue;
          if (It->second == AMDGPU::NUM_INST_CNTS)
            AnyClass = true;
          else
            Classes.insert(static_cast<unsigned>(It->second));
        }
      }
      if (!AnyClass && Classes.empty())
        continue;
      [[maybe_unused]] unsigned Added = 0;
      for (auto &[L, LCls] : VMemLoads) {
        if (L == &SU)
          continue;
        // Only match a load on a counter class the consumer waits on.
        if (!AnyClass && !Classes.contains(static_cast<unsigned>(LCls)))
          continue;
        if (DAG->addEdge(&SU, SDep(L, SDep::Artificial)))
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
