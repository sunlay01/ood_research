import Mathlib.Tactic

namespace OODRelevance

/-! Response-space statements are kept abstract.  No claim about causal or
    latent semantics is encoded here. -/

variable {V : Type*} [AddCommGroup V]

def additiveResponse (A : V → V → V) (s t : V) : Prop :=
  A (s + t) = A s + A t

theorem direct_sum_two_unique (u v u' v' : V)
    (hzero : ∀ a b : V, a + b = 0 → a = 0 ∧ b = 0)
    (heq : u + v = u' + v') :
    u = u' ∧ v = v' := by
  have hz : (u - u') + (v - v') = 0 := by
    calc
      (u - u') + (v - v') = (u + v) - (u' + v') := by abel
      _ = 0 := sub_eq_zero.mpr heq
  have h := hzero (u - u') (v - v') hz
  exact ⟨sub_eq_zero.mp h.1, sub_eq_zero.mp h.2⟩

theorem direct_sum_response_decomposition_unique
    (A B C D : V) (hzero : ∀ a b : V, a + b = 0 → a = 0 ∧ b = 0)
    (h : A + B = C + D) : A = C ∧ B = D := by
  exact direct_sum_two_unique A B C D hzero h

theorem additive_response_of_linear
    [Module ℝ V] {W : Type*} [AddCommGroup W] [Module ℝ W]
    (L : W →ₗ[ℝ] V) (s t : W) :
    L (s + t) = L s + L t := by
  exact L.map_add s t

end OODRelevance
