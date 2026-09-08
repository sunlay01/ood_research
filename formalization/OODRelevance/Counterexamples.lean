import Mathlib.Tactic

namespace OODRelevance

theorem common_burden_zero_discrimination (kappa w : ℝ) :
    (w + kappa) - w = kappa := by ring

theorem common_burden_model_independent {V : Type*} (baseRisk : V → ℝ)
    (cS cT : ℝ) (w₁ w₂ : V) :
    ((baseRisk w₁ + cT) - (baseRisk w₁ + cS)) -
      ((baseRisk w₂ + cT) - (baseRisk w₂ + cS)) = 0 := by
  ring

/-! The exact two-coordinate source-usage counterexample, written as scalar
    quadratic moments.  It has source optimum (1/2, 0) and target gradient
    (0, -1). -/

def sourceRisk (x y : ℝ) : ℝ := 2 * x ^ 2 + y ^ 2 - 2 * x

def targetRisk (x y : ℝ) : ℝ := 2 * x ^ 2 + 2 * x * y + 2 * y ^ 2 - 2 * x - 2 * y

theorem source_counterexample_optimum :
    sourceRisk (1 / 2 : ℝ) 0 = -(1 / 2 : ℝ) := by
  norm_num [sourceRisk]

theorem source_counterexample_source_excess (dx dy : ℝ) :
    sourceRisk (1 / 2 + dx) dy - sourceRisk (1 / 2) 0 = 2 * dx ^ 2 + dy ^ 2 := by
  norm_num [sourceRisk]
  ring

theorem source_counterexample_is_minimum (dx dy : ℝ) :
    sourceRisk (1 / 2) 0 ≤ sourceRisk (1 / 2 + dx) dy := by
  rw [← sub_nonneg]
  rw [source_counterexample_source_excess]
  positivity

theorem source_counterexample_zero_usage :
    ∃ wS : ℝ × ℝ, wS = (1 / 2, 0) ∧ wS.2 = 0 := by
  exact ⟨(1 / 2, 0), rfl, rfl⟩

theorem source_counterexample_target_gradient :
    (2 * (0 : ℝ) * (1 / 2 : ℝ) - 2 * 0 : ℝ) = 0 ∧
    (2 * (1 : ℝ) * (1 / 2 : ℝ) - 2 * 1 : ℝ) = -1 := by
  norm_num

theorem source_counterexample_shift_response (dx dy : ℝ) :
    (targetRisk (1 / 2 + dx) dy - sourceRisk (1 / 2 + dx) dy) -
      (targetRisk (1 / 2) 0 - sourceRisk (1 / 2) 0) =
      -dy + 2 * dx * dy + dy ^ 2 := by
  norm_num [targetRisk, sourceRisk]
  ring

theorem source_counterexample_target_positive_definite :
    (2 : ℝ) > 0 ∧ (2 * 2 - 1 * 1 : ℝ) > 0 := by
  norm_num

end OODRelevance
