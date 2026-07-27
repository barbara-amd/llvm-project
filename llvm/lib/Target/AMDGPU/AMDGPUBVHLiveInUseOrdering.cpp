//===-- AMDGPUBVHLiveInUseOrdering.cpp - AMDGPU BVH Live-In Use Ordering --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Post-RA DAG mutation for ray-traversal code.
///
///       A BVH result (image_bvh_intersect_ray) produced in a predecessor
///       block and consumed at the top of this region has no producer SUnit
///       here, so the post-RA scheduler leaves the consumer ahead of the
///       region's own VMEM loads.  SIInsertWaitcnts then waits for the BVH
///       result before issuing those loads, serialising otherwise-independent
///       in-flight loads (pre-GFX12 they even share vmcnt with the BVH result;
///       GFX12+ splits into bvhcnt/loadcnt but still issues the loads after the
///       bvhcnt wait).
///
///       To let the loads overlap, add Artificial order edges from every
///       in-region VMEM load to each consumer of a BVH-pending live-in, sinking
///       the loads above the consumer.
///
///       Scope is intentionally narrow (see apply()): only the first region of
///       a block, only live-ins whose reaching def is a BVH load in every
///       predecessor, resolved by one backward scan per direct predecessor.
///
///       Post-RA only: relies on physreg live-ins/defs.  Requires
///       RemoveKillFlags (set by GCNPostScheduleDAGMILive).
//
//===----------------------------------------------------------------------===//

#include "AMDGPUBVHLiveInUseOrdering.h"
#include "AMDGPUWaitcntUtils.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-bvh-livein-use-ordering"

namespace {

using RegUnitSet = SmallDenseSet<unsigned, 16>;

// MCRegUnit is an enum class; the sets index by plain unsigned.
static unsigned ruIdx(MCRegUnit U) { return static_cast<unsigned>(U); }

// Only VGPR/AGPR units can hold a BVH result, so only they can be BVH-pending.
static bool isVectorPhysReg(MCRegister Reg, const TargetRegisterInfo &TRI) {
  const TargetRegisterClass *RC = TRI.getPhysRegBaseClass(Reg);
  return RC &&
         (SIRegisterInfo::isVGPRClass(RC) || SIRegisterInfo::isAGPRClass(RC));
}

static bool isBVHImageLoad(const MachineInstr &MI) {
  if (!SIInstrInfo::isImage(MI))
    return false;
  const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(MI.getOpcode());
  if (!Info)
    return false;
  const AMDGPU::MIMGBaseOpcodeInfo *Base =
      AMDGPU::getMIMGBaseOpcodeInfo(Info->BaseOpcode);
  return Base && Base->BVH;
}

class BVHLiveInUseOrdering : public ScheduleDAGMutation {
  const GCNSubtarget &ST;

  /// For a BUNDLE header this scans the bundled members.
  static bool definesUnit(const MachineInstr &MI, MCRegUnit U,
                          const TargetRegisterInfo &TRI) {
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.isDef() || !MO.getReg().isPhysical())
        continue;
      for (MCRegUnit RU : TRI.regunits(MO.getReg().asMCReg()))
        if (RU == U)
          return true;
    }
    return false;
  }

  /// Is the def of \p U a BVH image load?  For a BUNDLE, classify by the member
  /// that actually defines \p U, not by the bundle as a whole.
  static bool defOfUnitIsBVH(const MachineInstr &MI, MCRegUnit U,
                             const TargetRegisterInfo &TRI) {
    if (!MI.isBundle())
      return isBVHImageLoad(MI);
    for (auto It = std::next(MI.getIterator()), E = MI.getParent()->instr_end();
         It != E && It->isBundledWithPred(); ++It)
      if (definesUnit(*It, U, TRI))
        return isBVHImageLoad(*It);
    return false;
  }

  /// A wait that fully drains the BVH result: s_wait_bvhcnt 0 on GFX12+, an
  /// s_waitcnt vmcnt(0) on the pre-GFX12 unified counter.
  bool isBVHClearingWait(const MachineInstr &MI) const {
    if (ST.hasExtendedWaitCounts())
      return MI.getOpcode() == AMDGPU::S_WAIT_BVHCNT &&
             MI.getOperand(0).isImm() && MI.getOperand(0).getImm() == 0;
    if (MI.getOpcode() != AMDGPU::S_WAITCNT || !MI.getOperand(0).isImm())
      return false;
    AMDGPU::Waitcnt W =
        AMDGPU::decodeWaitcnt(AMDGPU::getIsaVersion(ST.getCPU()),
                              MI.getOperand(0).getImm());
    return W.get(AMDGPU::LOAD_CNT) == 0;
  }

  /// Drop from \p Alive every unit that \p Pred does not prove BVH-pending.  A
  /// unit stays only if its closest reaching reference here is a BVH-load def
  /// with no intervening clearing wait; a non-BVH def, a use before any def, a
  /// clearing wait, or never being referenced all disqualify it.
  void scanPredecessor(const MachineBasicBlock *Pred, RegUnitSet &Alive,
                       const TargetRegisterInfo &TRI) {
    RegUnitSet Local = Alive; // still unresolved in this pred
    for (const MachineInstr &MI : make_range(Pred->rbegin(), Pred->rend())) {
      if (Local.empty())
        return;
      if (MI.isMetaInstruction())
        continue;
      if (isBVHClearingWait(MI)) {
        for (unsigned U : Local)
          Alive.erase(U);
        return;
      }
      // Defs before uses: the reaching def resolves the unit (covers RMW).
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.isDef() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit RU : TRI.regunits(MO.getReg().asMCReg())) {
          if (!Local.erase(ruIdx(RU)))
            continue;
          if (!defOfUnitIsBVH(MI, RU, TRI))
            Alive.erase(ruIdx(RU));
        }
      }
      // A use reached before any def: consumed inside this pred, not a live-in.
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || MO.isDef() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit RU : TRI.regunits(MO.getReg().asMCReg()))
          if (Local.erase(ruIdx(RU)))
            Alive.erase(ruIdx(RU));
      }
    }
    for (unsigned U : Local)
      Alive.erase(U);
  }

public:
  BVHLiveInUseOrdering(MachineFunction *MF)
      : ST(MF->getSubtarget<GCNSubtarget>()) {}

  void apply(ScheduleDAGInstrs *DAG) override {
    // No BVH ray-tracing loads -> no BVH-pending units possible.
    if (!ST.hasBVHRayTracingInsts())
      return;
    if (DAG->SUnits.empty())
      return;
    const TargetRegisterInfo *TRI = DAG->TRI;
    const MachineBasicBlock *MBB = DAG->begin()->getParent();

    // Only a block's first region sees true block live-ins; a later region's
    // live-ins are produced earlier in the same block.  Bail if any real
    // instruction precedes this region.
    MachineBasicBlock::const_iterator RegionBegin = DAG->begin();
    for (MachineBasicBlock::const_iterator I = MBB->begin(); I != RegionBegin;
         ++I)
      if (!I->isMetaInstruction())
        return;

    // With no predecessors there is nothing to confirm a BVH def against (an
    // entry block's vector live-ins are function arguments, not BVH results).
    if (MBB->pred_empty())
      return;

    // A cross-block BVH result can only arrive on a live-in vector unit.
    RegUnitSet LiveInUnits;
    for (const auto &LI : MBB->liveins())
      if (isVectorPhysReg(LI.PhysReg, *TRI))
        for (MCRegUnit U : TRI->regunits(LI.PhysReg))
          LiveInUnits.insert(ruIdx(U));
    if (LiveInUnits.empty())
      return;

    // Collect the in-region VMEM loads and the live-in units read before any
    // in-region def (the candidate consumers).
    SmallVector<SUnit *, 8> VMemLoads;
    RegUnitSet Defined;
    RegUnitSet Cands;
    for (SUnit &SU : DAG->SUnits) {
      const MachineInstr *MI = SU.getInstr();
      if (!MI || MI->isMetaInstruction())
        continue;
      if (AMDGPU::isPureVMemLoad(*MI, ST))
        VMemLoads.push_back(&SU);
      for (const MachineOperand &MO : MI->uses()) {
        if (!MO.isReg() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit U : TRI->regunits(MO.getReg().asMCReg())) {
          unsigned Idx = ruIdx(U);
          if (!Defined.contains(Idx) && LiveInUnits.contains(Idx))
            Cands.insert(Idx);
        }
      }
      for (const MachineOperand &MO : MI->operands()) {
        if (!MO.isReg() || !MO.isDef() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit U : TRI->regunits(MO.getReg().asMCReg()))
          Defined.insert(ruIdx(U));
      }
    }
    if (VMemLoads.empty() || Cands.empty())
      return;

    // A unit stays pending only if every direct predecessor proves its
    // reaching def is a BVH load (one backward scan each, no recursion).  This
    // covers the rotated ray-traversal loop and the simple diamond, and
    // conservatively skips anything else.
    RegUnitSet Pending = Cands;
    for (const MachineBasicBlock *Pred : MBB->predecessors()) {
      scanPredecessor(Pred, Pending, *TRI);
      if (Pending.empty())
        return;
    }

    LLVM_DEBUG(dbgs() << "BVHLiveInUseOrdering: " << Pending.size()
                      << " BVH-pending unit(s) in " << MBB->getFullName()
                      << "\n");

    // Walk the region in order adding Artificial edges load -> consumer.  A
    // unit is only a live-in BVH result until redefined here, so test uses
    // before erasing this instruction's defs: a later use of a redefined unit
    // reads the new value and already carries a Data dep.  addEdge() drops
    // self/cycle edges (excluding a load that depends on the consumer);
    // redundant edges are harmless as Artificial latency is 0.
    RegUnitSet PendingAtPoint = Pending;
    for (SUnit &SU : DAG->SUnits) {
      MachineInstr *MI = SU.getInstr();
      if (!MI || MI->isMetaInstruction())
        continue;
      bool ReadsPending = false;
      for (const MachineOperand &MO : MI->uses()) {
        if (!MO.isReg() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit U : TRI->regunits(MO.getReg().asMCReg()))
          if (PendingAtPoint.contains(ruIdx(U))) {
            ReadsPending = true;
            break;
          }
        if (ReadsPending)
          break;
      }
      if (ReadsPending) {
        [[maybe_unused]] unsigned Added = 0;
        for (SUnit *L : VMemLoads)
          if (L != &SU && DAG->addEdge(&SU, SDep(L, SDep::Artificial)))
            ++Added;
        LLVM_DEBUG(if (Added) dbgs() << "BVHLiveInUseOrdering: added " << Added
                                     << " edge(s) to SU(" << SU.NodeNum << ") "
                                     << *MI);
      }
      // An in-region redefinition ends the live-in value's reach.
      for (const MachineOperand &MO : MI->operands()) {
        if (!MO.isReg() || !MO.isDef() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit U : TRI->regunits(MO.getReg().asMCReg()))
          PendingAtPoint.erase(ruIdx(U));
      }
      if (PendingAtPoint.empty())
        break;
    }
  }
};

} // end namespace

std::unique_ptr<ScheduleDAGMutation>
llvm::createAMDGPUBVHLiveInUseOrderingDAGMutation(MachineFunction *MF) {
  return std::make_unique<BVHLiveInUseOrdering>(MF);
}
