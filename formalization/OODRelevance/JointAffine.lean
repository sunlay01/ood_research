import Mathlib.Tactic

namespace OODRelevance

variable {V W Y : Type*} [AddCommGroup V] [Module ℝ V]
  [AddCommGroup W] [Module ℝ W] [AddCommGroup Y] [Module ℝ Y]

/- The affine response's world-dependent part is linear even when its static
  offset is nonzero.  This is the algebraic core used by the 3C-B/3E-B audit. -/
theorem affine_world_part_additive
    (A : V →ₗ[ℝ] W) (O : V →ₗ[ℝ] Y) (P : Y →ₗ[ℝ] W) (u v : V) :
    A (u + v) + P (O (u + v)) =
      (A u + P (O u)) + (A v + P (O v)) := by
  simp only [map_add]
  abel

/- Any source-only policy has exactly the original response on an invisible
  direction.  This is the norm-free kernel lower-bound core. -/
theorem affine_policy_kernel_residual
    (O : V →ₗ[ℝ] Y) (A : V →ₗ[ℝ] W) (B : Y →ₗ[ℝ] W)
    {v : V} (hv : v ∈ LinearMap.ker O) :
    A v - B (O v) = A v := by
  have hO : O v = 0 := hv
  simp [hO]

theorem zero_steering_affine_specialization
    (A : V →ₗ[ℝ] W) (O : V →ₗ[ℝ] Y) (P : Y →ₗ[ℝ] W) (u : V)
    (hP : P = 0) :
    A u + P (O u) = A u := by
  simp [hP]

end OODRelevance
