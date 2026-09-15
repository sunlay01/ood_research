import Mathlib.Analysis.Normed.Operator.BoundedLinearMaps

/-!
# Restricted counterfactual repair

This file formalizes the order-theoretic part of the Phase-0 interface.  The
response spaces are finite-dimensional real normed spaces in the intended use,
but the statements only need continuous linear maps.  Numerical optimization,
the spectral-slack certificate, and the concrete toy matrices remain audited
by Python; this file does not claim to formalize those numerical procedures.
-/

namespace OODRelevance

open Set

variable {U Y Z : Type*}
variable [NormedAddCommGroup U] [NormedSpace ℝ U]
variable [NormedAddCommGroup Y] [NormedSpace ℝ Y]
variable [NormedAddCommGroup Z] [NormedSpace ℝ Z]

abbrev ResponseMap (Y Z : Type*) [NormedAddCommGroup Y] [NormedSpace ℝ Y]
    [NormedAddCommGroup Z] [NormedSpace ℝ Z] := Y →L[ℝ] Z

noncomputable def repairRisk (A : U →L[ℝ] Z) (O : U →L[ℝ] Y) (Pi : ResponseMap Y Z) : ℝ :=
  (1 / 2 : ℝ) * ‖A + Pi.comp O‖ ^ 2

abbrev Admissible (Y Z : Type*) [NormedAddCommGroup Y] [NormedSpace ℝ Y]
    [NormedAddCommGroup Z] [NormedSpace ℝ Z] := Set (ResponseMap Y Z)

def IsClassOptimal (A : U →L[ℝ] Z) (O : U →L[ℝ] Y)
    (C : Admissible Y Z) (Pi : ResponseMap Y Z) : Prop :=
  Pi ∈ C ∧ ∀ Q, Q ∈ C → repairRisk A O Pi ≤ repairRisk A O Q

theorem information_floor_inherited
    (A : U →L[ℝ] Z) (O : U →L[ℝ] Y) (C : Admissible Y Z)
    (Pi : ResponseMap Y Z) (Rinfo : ℝ)
    (_hopt : IsClassOptimal A O C Pi)
    (hfloor : ∀ Q : ResponseMap Y Z, Rinfo ≤ repairRisk A O Q) :
    Rinfo ≤ repairRisk A O Pi := by
  exact hfloor Pi

theorem nested_class_monotonicity
    (A : U →L[ℝ] Z) (O : U →L[ℝ] Y)
    {C₁ C₂ : Admissible Y Z} (hsub : C₁ ⊆ C₂)
    {Pi₁ Pi₂ : ResponseMap Y Z}
    (hopt₁ : IsClassOptimal A O C₁ Pi₁)
    (hopt₂ : IsClassOptimal A O C₂ Pi₂) :
    repairRisk A O Pi₂ ≤ repairRisk A O Pi₁ := by
  exact hopt₂.2 Pi₁ (hsub hopt₁.1)

theorem repair_decomposition
    (Rinfo Rclass Ractual : ℝ) :
    Ractual = Rinfo + (Rclass - Rinfo) + (Ractual - Rclass) := by
  ring

theorem zero_observation_repairRisk
    (A : U →L[ℝ] Z) (Pi : ResponseMap Y Z)
    (O : U →L[ℝ] Y) (hO : O = 0) :
    repairRisk A O Pi = (1 / 2 : ℝ) * ‖A‖ ^ 2 := by
  subst O
  simp [repairRisk]

theorem zero_observation_class_optimal_is_information_limited
    (A : U →L[ℝ] Z) (O : U →L[ℝ] Y) (hO : O = 0)
    (C : Admissible Y Z) (Pi : ResponseMap Y Z)
    (_hopt : IsClassOptimal A O C Pi) :
    repairRisk A O Pi = (1 / 2 : ℝ) * ‖A‖ ^ 2 := by
  exact zero_observation_repairRisk A Pi O hO

end OODRelevance
