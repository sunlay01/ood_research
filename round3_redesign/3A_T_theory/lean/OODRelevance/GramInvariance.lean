import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Tactic

namespace OODRelevance

variable {V W : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  [NormedAddCommGroup W] [InnerProductSpace ℝ W]

theorem gram_invariant_under_isometry (F : V →ₗ[ℝ] W)
    (hF : ∀ x y, inner ℝ (F x) (F y) = inner ℝ x y) (u v : V) :
    inner ℝ (F u) (F v) = inner ℝ u v := by
  exact hF u v

theorem normalized_gram_invariant_under_isometry
    (F : V →ₗ[ℝ] W)
    (hF : ∀ x y, inner ℝ (F x) (F y) = inner ℝ x y)
    (hFnorm : ∀ x, ‖F x‖ = ‖x‖) (u v : V) :
    inner ℝ (F u) (F v) / (‖F u‖ * ‖F v‖) =
      inner ℝ u v / (‖u‖ * ‖v‖) := by
  rw [hF u v, hFnorm u, hFnorm v]

end OODRelevance
