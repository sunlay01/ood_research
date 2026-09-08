import Mathlib.Algebra.Module.LinearMap.Defs
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic

namespace OODRelevance

theorem first_order_null_scaling {V : Type*} [AddCommGroup V] [Module ℝ V]
    (A : V →ₗ[ℝ] V →ₗ[ℝ] ℝ) (epsilon : ℝ) (he : 0 ≤ epsilon)
    (u : V) :
    epsilon * A u u = A (Real.sqrt epsilon • u) (Real.sqrt epsilon • u) := by
  simp only [map_smul, smul_eq_mul]
  change epsilon * A u u = Real.sqrt epsilon * (Real.sqrt epsilon * A u u)
  ring_nf
  rw [Real.sq_sqrt he]
  ring

end OODRelevance
