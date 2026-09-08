import Mathlib.Tactic

namespace OODRelevance

variable {V W : Type*} [AddCommGroup V] [Module ℝ V]
  [AddCommGroup W] [Module ℝ W]

def response (L : V →ₗ[ℝ] W) (s : V) : W := L s

theorem response_additive (L : V →ₗ[ℝ] W) (s t : V) :
    response L (s + t) = response L s + response L t := by
  exact L.map_add s t

theorem response_three_additive (L : V →ₗ[ℝ] W) (a b c : V) :
    response L (a + b + c) = response L a + response L b + response L c := by
  simp [response, map_add, add_assoc]

end OODRelevance
