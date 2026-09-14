import OodTheoryVerification.Stage9.Basic
import Mathlib.Tactic

namespace OodTheoryVerification.Stage13R2

open OodTheoryVerification.Stage9

/-! Stable finite-dimensional algebraic slice for the transfer-measure lift.
    General suprema over predictor classes, envelope hypotheses, and
    annihilator existence remain paper-level theorems. -/

noncomputable def affineExcessRisk (a : ℝ) (g psi : Vec2) : ℝ :=
  a + dot g psi

theorem affine_excess_transfer_identity (a : ℝ) (g psiT psiS : Vec2) :
    affineExcessRisk a g psiT - affineExcessRisk a g psiS =
      dot g (psiT - psiS) := by
  unfold affineExcessRisk dot
  simp only [Fin.sum_univ_two, Pi.sub_apply]
  ring

def finiteTransferSupport (x y : ℝ) : ℝ := max x y

theorem finite_transfer_support_upper (x y z : ℝ)
    (hx : x ≤ z) (hy : y ≤ z) :
    finiteTransferSupport x y ≤ z := by
  unfold finiteTransferSupport
  exact max_le hx hy

def blindExtension (g0 t : ℝ) : Vec2
  | 0 => g0
  | 1 => t

def sourceResponse (g : Vec2) : ℝ := g 0

theorem blind_extension_source_response (g0 t : ℝ) :
    sourceResponse (blindExtension g0 t) = g0 := by
  rfl

theorem blind_extension_target_response (g0 t : ℝ) :
    (blindExtension g0 t) 1 = t := by
  rfl

def additiveExcess (theta xi : ℝ) : ℝ := theta ^ 2

theorem additive_excess_ignores_environment (theta xi : ℝ) :
    additiveExcess theta xi = theta ^ 2 := by
  rfl

end OodTheoryVerification.Stage13R2
