import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Tactic

namespace OODRelevance

/-! Euclidean support is the inner-product form used after whitening a
    positive-definite source metric. -/

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem cauchy_support_upper (g d : V) (radius : ℝ)
    (hd : ‖d‖ ≤ radius) : |inner ℝ g d| ≤ radius * ‖g‖ := by
  calc
    |inner ℝ g d| ≤ ‖g‖ * ‖d‖ := abs_real_inner_le_norm g d
    _ ≤ ‖g‖ * radius := mul_le_mul_of_nonneg_left hd (norm_nonneg g)
    _ = radius * ‖g‖ := by ring

theorem cauchy_support_attained (g : V) (radius : ℝ) (hr : 0 ≤ radius) :
    ∃ d, ‖d‖ ≤ radius ∧ |inner ℝ g d| = radius * ‖g‖ := by
  by_cases hg : g = 0
  · subst g
    refine ⟨0, ?_, ?_⟩
    · simpa using hr
    · simp
  · let d : V := (radius / ‖g‖) • g
    refine ⟨d, ?_, ?_⟩
    · rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (div_nonneg hr (norm_nonneg g))]
      field_simp
      exact le_rfl
    · change |inner ℝ g ((radius / ‖g‖) • g)| = radius * ‖g‖
      rw [inner_smul_right, real_inner_self_eq_norm_sq]
      rw [abs_of_nonneg]
      · field_simp
      · positivity

theorem cauchy_support_isGreatest (g : V) (radius : ℝ) (hr : 0 ≤ radius) :
    IsGreatest {value : ℝ | ∃ d : V, ‖d‖ ≤ radius ∧ value = |inner ℝ g d|}
      (radius * ‖g‖) := by
  obtain ⟨d, hd, hvalue⟩ := cauchy_support_attained g radius hr
  constructor
  · exact ⟨d, hd, hvalue.symm⟩
  · intro value hvalueMem
    obtain ⟨x, hx, rfl⟩ := hvalueMem
    exact cauchy_support_upper g x radius hx

end OODRelevance
