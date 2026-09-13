import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace OodTheoryVerification.Stage11

/-! A deliberately small, exact supervised representation.

    `QuadState` stores the three moments needed by one-dimensional linear
    regression with squared loss.  The point of this file is not to claim a
    universal representation: it machine-checks the restricted model used by
    the Stage 11 hard gate. -/

structure QuadState where
  m : ℝ
  c : ℝ
  s : ℝ

def quadRisk (w : ℝ) (q : QuadState) : ℝ :=
  w ^ 2 * q.m - 2 * w * q.c + q.s

noncomputable def empiricalRisk {n : ℕ} (w : ℝ) (x y : Fin n → ℝ) : ℝ :=
  (n : ℝ)⁻¹ * ∑ i, (w * x i - y i) ^ 2

def empiricalState {n : ℕ} (x y : Fin n → ℝ) : QuadState :=
  { m := ∑ i, (x i) ^ 2
    c := ∑ i, x i * y i
    s := ∑ i, (y i) ^ 2 }

theorem squared_loss_expand (w x y : ℝ) :
    (w * x - y) ^ 2 = w ^ 2 * x ^ 2 - 2 * w * (x * y) + y ^ 2 := by
  ring

theorem empirical_state_risk_identity {n : ℕ} (w : ℝ)
    (x y : Fin n → ℝ) :
    empiricalRisk w x y = (n : ℝ)⁻¹ * quadRisk w (empiricalState x y) := by
  unfold empiricalRisk quadRisk empiricalState
  have hsum :
      (∑ i, (w * x i - y i) ^ 2) =
        ∑ i, (w ^ 2 * (x i) ^ 2 - 2 * w * (x i * y i) + (y i) ^ 2) := by
    apply Finset.sum_congr rfl
    intro i hi
    ring
  rw [hsum]
  have hsum2 :
      (∑ i, (w ^ 2 * (x i) ^ 2 - 2 * w * (x i * y i) + (y i) ^ 2)) =
        w ^ 2 * (∑ i, (x i) ^ 2) - 2 * w * (∑ i, x i * y i) + ∑ i, (y i) ^ 2 := by
    rw [Finset.sum_add_distrib, Finset.sum_sub_distrib]
    rw [← Finset.mul_sum, ← Finset.mul_sum]
  rw [hsum2]

theorem linear_state_risk_identity (w : ℝ) (q : QuadState) :
    quadRisk w q = (w ^ 2) * q.m + (-2 * w) * q.c + q.s := by
  unfold quadRisk
  ring

end OodTheoryVerification.Stage11
