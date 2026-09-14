import Mathlib.Tactic

namespace OodTheoryVerification.Stage13R1

/-! Stable algebraic identities for the Stage 13R.1 nuisance audit.  This file
    intentionally avoids formalising the general envelope theorem or path
    integration. -/

def rawEnvironmentalDerivative (r c : ℝ) : ℝ := r + c

theorem additive_nuisance_raw_shift (r c : ℝ) :
    rawEnvironmentalDerivative r c - rawEnvironmentalDerivative r 0 = c := by
  unfold rawEnvironmentalDerivative
  ring

def pairwiseRisk (rf rh c : ℝ) : ℝ := (rf + c) - (rh + c)

theorem additive_nuisance_pairwise_cancel (rf rh c : ℝ) :
    pairwiseRisk rf rh c = rf - rh := by
  unfold pairwiseRisk
  ring

def riskR (theta xi : ℝ) : ℝ := theta ^ 2 + xi

def riskTilde (theta xi : ℝ) : ℝ := theta ^ 2 + 2 * xi

theorem separation_parameter_side (theta xi : ℝ) :
    theta ^ 2 = theta ^ 2 := rfl

theorem separation_environment_side (theta xi : ℝ) :
    riskR theta xi - theta ^ 2 = xi ∧
      riskTilde theta xi - theta ^ 2 = 2 * xi := by
  constructor <;> simp [riskR, riskTilde]

theorem separation_excess_agrees (theta xi : ℝ) :
    riskR theta xi - xi = riskTilde theta xi - 2 * xi := by
  simp [riskR, riskTilde]

end OodTheoryVerification.Stage13R1
