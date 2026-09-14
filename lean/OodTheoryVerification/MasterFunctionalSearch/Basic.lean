import Mathlib.Tactic

namespace OodTheoryVerification.MasterFunctionalSearch

/-! Finite algebraic audit for the quotient risk-landscape candidate.  The
    general quotient-space and loss-class theory remains in the paper note. -/

noncomputable def quotientRadius (d₀ d₁ d₂ d₃ : ℝ) : ℝ :=
  (max (max d₀ d₁) (max d₂ d₃) - min (min d₀ d₁) (min d₂ d₃)) / 2

def pairwiseDiameter (d₀ d₁ d₂ d₃ : ℝ) : ℝ :=
  max (max d₀ d₁) (max d₂ d₃) - min (min d₀ d₁) (min d₂ d₃)

theorem quotient_radius_half_pairwise_diameter
    (d₀ d₁ d₂ d₃ : ℝ) :
    quotientRadius d₀ d₁ d₂ d₃ = pairwiseDiameter d₀ d₁ d₂ d₃ / 2 := by
  unfold quotientRadius pairwiseDiameter
  ring

def additiveShift (d c : ℝ) : ℝ := d + c

theorem pairwise_shift_cancel (d₀ d₁ c : ℝ) :
    additiveShift d₀ c - additiveShift d₁ c = d₀ - d₁ := by
  unfold additiveShift
  ring

end OodTheoryVerification.MasterFunctionalSearch
