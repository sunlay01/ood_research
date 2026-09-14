import OodTheoryVerification.Stage9.Basic
import Mathlib.Tactic

namespace OodTheoryVerification.Stage13R

open OodTheoryVerification.Stage9

/-! Stable algebraic witnesses for Stage 13R.  General Gateaux derivatives,
    absolutely-continuous path integration, and Taylor remainders remain
    paper theorems; this file checks only finite-dimensional identities that
    do not depend on a calculus formalisation. -/

def affineRisk (b : ℝ) (g xi : Vec2) : ℝ := b + dot g xi

def affineDirectional (g delta : Vec2) : ℝ := dot g delta

def esfTwo (g d₀ d₁ : Vec2) : ℝ :=
  max (affineDirectional g d₀) (affineDirectional g d₁)

theorem affine_directional_increment (b : ℝ) (g xi delta : Vec2) :
    affineRisk b g (xi + delta) - affineRisk b g xi =
      affineDirectional g delta := by
  unfold affineRisk affineDirectional dot
  simp only [Fin.sum_univ_two, Pi.add_apply]
  ring

theorem affine_esf_two_is_finite_support (b : ℝ) (g xi d₀ d₁ : Vec2) :
    max (affineRisk b g (xi + d₀) - affineRisk b g xi)
        (affineRisk b g (xi + d₁) - affineRisk b g xi) =
      esfTwo g d₀ d₁ := by
  rw [affine_directional_increment, affine_directional_increment]
  rfl

theorem affine_esf_two_translation_invariant (g xi d₀ d₁ : Vec2) :
    esfTwo g d₀ d₁ =
      esfTwo g (xi + d₀ - xi) (xi + d₁ - xi) := by
  unfold esfTwo affineDirectional dot
  simp only [Fin.sum_univ_two, Pi.add_apply, Pi.sub_apply]
  congr 1 <;> ring_nf

/-! A finite-dimensional separation witness.  The two risks have identical
    parameter-side first and second derivatives (and all higher derivatives
    vanish), while their environment derivatives and target values differ. -/

def parameterRisk (theta xi : ℝ) : ℝ := theta ^ 2 + xi

def parameterRiskTilde (theta xi : ℝ) : ℝ := theta ^ 2 + 2 * xi

def parameterDerivative (theta _xi : ℝ) : ℝ := 2 * theta

def parameterDerivativeTilde (theta _xi : ℝ) : ℝ := 2 * theta

def parameterSecondDerivative (_theta _xi : ℝ) : ℝ := 2

def parameterSecondDerivativeTilde (_theta _xi : ℝ) : ℝ := 2

def parameterHigherDerivative (_order : ℕ) (_theta _xi : ℝ) : ℝ := 0

def parameterHigherDerivativeTilde (_order : ℕ) (_theta _xi : ℝ) : ℝ := 0

def environmentDerivative (_theta _xi : ℝ) : ℝ := 1

def environmentDerivativeTilde (_theta _xi : ℝ) : ℝ := 2

theorem parameter_first_derivatives_match (theta xi : ℝ) :
    parameterDerivative theta xi = parameterDerivativeTilde theta xi := by
  rfl

theorem parameter_second_derivatives_match (theta xi : ℝ) :
    parameterSecondDerivative theta xi = parameterSecondDerivativeTilde theta xi := by
  rfl

theorem parameter_higher_derivatives_match (order : ℕ) (theta xi : ℝ) :
    parameterHigherDerivative order theta xi =
      parameterHigherDerivativeTilde order theta xi := by
  rfl

theorem environment_derivatives_separate (theta xi : ℝ) :
    environmentDerivative theta xi ≠ environmentDerivativeTilde theta xi := by
  norm_num [environmentDerivative, environmentDerivativeTilde]

theorem target_risks_separate {theta xi : ℝ} (hxi : xi ≠ 0) :
    parameterRisk theta xi ≠ parameterRiskTilde theta xi := by
  unfold parameterRisk parameterRiskTilde
  intro h
  have hz : xi = 0 := by linarith [h]
  exact hxi hz

end OodTheoryVerification.Stage13R
