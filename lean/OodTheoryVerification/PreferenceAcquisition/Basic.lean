import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic

namespace OodTheoryVerification.PreferenceAcquisition

noncomputable def coreWeight (sigma : ℝ) : ℝ :=
  Real.log ((1 - sigma) / sigma)

noncomputable def shortcutWeight (rho : ℝ) : ℝ :=
  Real.log (rho / (1 - rho))

theorem odds_preference_iff (rho sigma : ℝ)
    (hsigma_pos : 0 < sigma) (hrho_lt : rho < 1) :
    (1 - sigma) / sigma < rho / (1 - rho) ↔ 1 - sigma < rho := by
  have hden : 0 < 1 - rho := sub_pos.mpr hrho_lt
  constructor
  · intro h
    have hcross := (div_lt_div_iff₀ hsigma_pos hden).mp h
    nlinarith
  · intro h
    apply (div_lt_div_iff₀ hsigma_pos hden).mpr
    nlinarith

theorem shortcut_weight_gt_core_iff (rho sigma : ℝ)
    (hrho_pos : 0 < rho) (hrho_lt : rho < 1)
    (hsigma_pos : 0 < sigma) (hsigma_lt : sigma < 1) :
    coreWeight sigma < shortcutWeight rho ↔ 1 - sigma < rho := by
  have hcore : 0 < (1 - sigma) / sigma :=
    div_pos (sub_pos.mpr hsigma_lt) hsigma_pos
  have hshortcut : 0 < rho / (1 - rho) :=
    div_pos hrho_pos (sub_pos.mpr hrho_lt)
  unfold coreWeight shortcutWeight
  rw [Real.log_lt_log_iff hcore hshortcut]
  exact odds_preference_iff rho sigma hsigma_pos hrho_lt

end OodTheoryVerification.PreferenceAcquisition
