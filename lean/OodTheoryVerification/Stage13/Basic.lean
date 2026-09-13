import OodTheoryVerification.Stage9.Basic
import OodTheoryVerification.Stage11R.Basic
import Mathlib.Tactic

namespace OodTheoryVerification.Stage13

open OodTheoryVerification.Stage9
open OodTheoryVerification.Stage11R

/-! Clean algebraic witnesses for the Stage 13 method translations. General
    RKHS, optimization, and mechanism-identification results remain in the
    paper-level method cards. -/

noncomputable def exposureEnergyTwo (g d0 d1 : Vec2) : ℝ :=
  (response g d0) ^ 2 / 2 + (response g d1) ^ 2 / 2

theorem vrex_exposure_identity (g d : Vec2) :
    exposureEnergyTwo g d (-d) = (response g d) ^ 2 := by
  unfold exposureEnergyTwo
  simp [response, dot, Fin.sum_univ_two]
  ring

def twoMix (a r0 r1 : ℝ) : ℝ := a * r0 + (1 - a) * r1

theorem groupDRO_two_mix_le_max {a r0 r1 : ℝ}
    (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    twoMix a r0 r1 ≤ max r0 r1 := by
  unfold twoMix
  by_cases h : r0 ≤ r1
  · rw [max_eq_right h]
    nlinarith [mul_nonpos_of_nonneg_of_nonpos ha0 (sub_nonpos.mpr h)]
  · have h' : r1 ≤ r0 := le_of_not_ge h
    rw [max_eq_left h']
    nlinarith [mul_nonpos_of_nonneg_of_nonpos (by linarith : 0 ≤ 1 - a)
      (sub_nonpos.mpr h')]

theorem groupDRO_two_mix_endpoints (r0 r1 : ℝ) :
    twoMix 1 r0 r1 = r0 ∧ twoMix 0 r0 r1 = r1 := by
  constructor <;> simp [twoMix]

def coralRisk (m c s w : ℝ) : ℝ := w ^ 2 * m - 2 * w * c + s

theorem coral_label_shift_counterexample :
    coralRisk 1 1 1 1 = 0 ∧ coralRisk 1 (-1) 1 1 = 4 := by
  norm_num [coralRisk]

def linearRiskGradient (m c w : ℝ) : ℝ := 2 * (m * w - c)

theorem ideal_irm_stationary_identity (m c w : ℝ) :
    linearRiskGradient m c w = 2 * (m * w - c) := rfl

theorem ideal_irm_stationary_implies (m c w : ℝ)
    (h : linearRiskGradient m c w = 0) : m * w = c := by
  unfold linearRiskGradient at h
  linarith

def secondDirection : Vec2
  | 0 => 0
  | 1 => 1

theorem blind_direction_witness :
    response secondDirection sourceDirection = 0 ∧
      response secondDirection secondDirection = 1 := by
  constructor <;> simp [response, dot, sourceDirection, secondDirection,
    Fin.sum_univ_two]

end OodTheoryVerification.Stage13
