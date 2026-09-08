import Mathlib.LinearAlgebra.Quotient.Basic
import Mathlib.LinearAlgebra.Isomorphisms
import Mathlib.Tactic

namespace OODRelevance

/-! Finite-dimensional source-exposure statements are phrased for abstract
    linear maps.  No probabilistic or causal interpretation is encoded. -/

variable {V W Y : Type*} [AddCommGroup V] [Module ℝ V]
  [AddCommGroup W] [Module ℝ W] [AddCommGroup Y] [Module ℝ Y]

def exposedImage (L : V →ₗ[ℝ] W) (E : Submodule ℝ V) : Submodule ℝ W :=
  E.map L

theorem exposed_membership_iff (L : V →ₗ[ℝ] W) (E : Submodule ℝ V) (x : V) :
    L x ∈ exposedImage L E ↔ ∃ y, y ∈ E ∧ L y = L x := by
  rw [exposedImage, Submodule.mem_map]

theorem irrelevant_source_variation_no_change (L : V →ₗ[ℝ] W)
    {u : V} (hu : L u = 0) : L (u + u) = L u := by
  simp [hu, map_add]

theorem source_reference_span_invariant (x y : V) :
    Submodule.span ℝ ({y - x} : Set V) = Submodule.span ℝ ({x - y} : Set V) := by
  rw [show x - y = -(y - x) by abel]
  simpa using (Submodule.span_neg (R := ℝ) ({y - x} : Set V)).symm

def factorsThroughObservation (O : V →ₗ[ℝ] Y) (A : V →ₗ[ℝ] W) : Prop :=
  ∃ B : LinearMap.range O →ₗ[ℝ] W,
    B.comp O.rangeRestrict = A

theorem factorsThroughObservation_implies_kernel_inclusion
    (O : V →ₗ[ℝ] Y) (A : V →ₗ[ℝ] W)
    (h : factorsThroughObservation O A) :
    LinearMap.ker O ≤ LinearMap.ker A := by
  rcases h with ⟨B, hB⟩
  intro x hx
  have hx' : O.rangeRestrict x = 0 := by
    apply Subtype.ext
    exact hx
  change A x = 0
  rw [← hB]
  simp [hx']

theorem kernel_inclusion_factorsThroughObservation
    (O : V →ₗ[ℝ] Y) (A : V →ₗ[ℝ] W)
    (h : LinearMap.ker O ≤ LinearMap.ker A) :
    factorsThroughObservation O A := by
  let Q : V ⧸ LinearMap.ker O →ₗ[ℝ] W :=
    (LinearMap.ker O).liftQ A h
  let B : LinearMap.range O →ₗ[ℝ] W :=
    Q.comp O.quotKerEquivRange.symm.toLinearMap
  refine ⟨B, ?_⟩
  ext x
  change Q (O.quotKerEquivRange.symm ⟨O x, O.mem_range_self x⟩) = A x
  rw [LinearMap.quotKerEquivRange_symm_apply_image]
  simpa [Q] using congrArg (fun L => L x) (Submodule.liftQ_mkQ A h)

theorem kernel_inclusion_iff_factorsThroughObservation
    (O : V →ₗ[ℝ] Y) (A : V →ₗ[ℝ] W) :
    LinearMap.ker O ≤ LinearMap.ker A ↔ factorsThroughObservation O A := by
  constructor
  · exact kernel_inclusion_factorsThroughObservation O A
  · exact factorsThroughObservation_implies_kernel_inclusion O A

theorem response_additive_source_contrast (L : V →ₗ[ℝ] W) (u v : V) :
    L (u + v) = L u + L v := by
  exact L.map_add u v

end OODRelevance
