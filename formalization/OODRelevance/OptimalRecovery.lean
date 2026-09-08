import OODRelevance.SourceExposure
import Mathlib.Tactic

namespace OODRelevance

/-! The algebraic core of 3E.  Operator norms and Moore--Penrose inverses are
    intentionally kept in the accompanying paper proof. -/

variable {V W Y : Type*} [AddCommGroup V] [Module ℝ V]
  [AddCommGroup W] [Module ℝ W] [AddCommGroup Y] [Module ℝ Y]

theorem invisible_pair_same_observation (O : V →ₗ[ℝ] Y) {v : V}
    (hv : v ∈ LinearMap.ker O) : O v = O (-v) := by
  have hv0 : O v = 0 := hv
  calc
    O v = 0 := hv0
    _ = -0 := by simp
    _ = -(O v) := by rw [hv0]
    _ = O (-v) := (O.map_neg v).symm

theorem response_zero_on_kernel_iff (O : V →ₗ[ℝ] Y) (A : V →ₗ[ℝ] W) :
    (∀ v : V, O v = 0 → A v = 0) ↔ LinearMap.ker O ≤ LinearMap.ker A := by
  constructor
  · intro h v hv
    exact h v hv
  · intro h v hv
    exact h hv

theorem zero_recovery_criterion (O : V →ₗ[ℝ] Y) (A : V →ₗ[ℝ] W) :
    (∀ v : V, O v = 0 → A v = 0) ↔ factorsThroughObservation O A := by
  rw [response_zero_on_kernel_iff]
  exact kernel_inclusion_iff_factorsThroughObservation O A

def duplicateObservation (O : V →ₗ[ℝ] Y) : V →ₗ[ℝ] (Y × Y) := O.prod O

theorem duplicate_observation_kernel (O : V →ₗ[ℝ] Y) :
    LinearMap.ker (duplicateObservation O) = LinearMap.ker O := by
  ext v
  simp [duplicateObservation]

theorem more_information_preserves_zero_recovery
    (O₁ : V →ₗ[ℝ] Y) (O₂ : V →ₗ[ℝ] Y) (A : V →ₗ[ℝ] W)
    (hinfo : LinearMap.ker O₂ ≤ LinearMap.ker O₁)
    (hzero : LinearMap.ker O₁ ≤ LinearMap.ker A) :
    LinearMap.ker O₂ ≤ LinearMap.ker A := by
  exact hinfo.trans hzero

end OODRelevance
