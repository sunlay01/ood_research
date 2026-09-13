import OodTheoryVerification.Stage9.Basic
import Mathlib.Tactic

namespace OodTheoryVerification.Stage12

open OodTheoryVerification.Stage9

/-! The exact two-source transfer identity. The general finite-dimensional
    support and quotient theorems are kept in the accompanying paper note. -/

noncomputable def affineRisk (b : ℝ) (g psi : Vec2) (eta : ℝ) : ℝ :=
  b + dot g psi + eta

theorem exact_transfer_identity (b : ℝ) (g psiT psi0 psi1 : Vec2)
    (etaT eta0 eta1 : ℝ) :
    affineRisk b g psiT etaT -
        (affineRisk b g psi0 eta0 + affineRisk b g psi1 eta1) / 2 =
      dot g (psiT - (psi0 + psi1) / 2) +
        etaT - (eta0 + eta1) / 2 := by
  unfold affineRisk
  simp [dot, Fin.sum_univ_two, div_eq_mul_inv]
  ring

theorem support_lower_bound (g d B : Vec2) (h : dot g d ≤ dot g B) :
    dot g d ≤ dot g B := h

end OodTheoryVerification.Stage12
