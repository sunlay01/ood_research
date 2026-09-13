import OodTheoryVerification.Stage9.Basic
import OodTheoryVerification.Stage11.Basic
import Mathlib.Tactic

namespace OodTheoryVerification.Stage11R

open OodTheoryVerification.Stage9

/-! A concrete two-coordinate witness for the coordinate-free annihilator
    statement. The abstract `V* / S°` quotient is kept in the paper note; this
    lemma checks its elementary source-response consequence. -/

def sourceDirection : Vec2
  | 0 => 1
  | 1 => 0

def response (g d : Vec2) : ℝ := dot g d

theorem annihilator_response (g h : Vec2) (hh : h 0 = 0) :
    response (g + h) sourceDirection = response g sourceDirection := by
  unfold response sourceDirection dot
  simp only [Fin.sum_univ_two]
  simp [sourceDirection, hh]

theorem exposure_energy_two_sources (g : Vec2) :
    ((response g sourceDirection) ^ 2 +
      (response g (fun i => -sourceDirection i)) ^ 2) / 2 =
      (g 0) ^ 2 := by
  unfold response dot
  simp only [Fin.sum_univ_two]
  simp [sourceDirection]

end OodTheoryVerification.Stage11R
