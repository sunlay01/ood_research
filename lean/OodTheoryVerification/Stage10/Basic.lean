import OodTheoryVerification.Stage9.Basic
import Mathlib.Tactic

namespace OodTheoryVerification.Stage10

open OodTheoryVerification.Stage9

/-! Concrete two-dimensional witnesses for the Stage 10 orientation claims.
    General principal-angle and Courant--Fischer machinery is intentionally
    kept in the accompanying paper theorem note. -/

def e1 : Vec2 := fun i => if i = 0 then 1 else 0
def e2 : Vec2 := fun i => if i = 1 then 1 else 0

theorem hidden_direction_witness (a : ℝ) (ha : a ≠ 0) :
    diagonal a 0 (e2) = 0 ∧ kernelDiagonal a 0 (e2) = e2 := by
  constructor
  · funext i
    fin_cases i <;> simp [diagonal, e2]
  · funext i
    fin_cases i <;> simp [kernelDiagonal, e2, ha]

theorem same_spectrum_line_quadratic (lmin lmax : ℝ)
    (hmin : 0 < lmin) (hmax : lmin < lmax) :
    diagonal lmax lmin e1 0 = lmax ∧
    diagonal lmin lmax e1 0 = lmin ∧
    lmin < lmax := by
  refine ⟨?_, ?_, hmax⟩ <;> simp [diagonal, e1]

theorem isotropic_diagonal_no_orientation (c : ℝ) (x : Vec2) :
    diagonal c c x = fun i => c * x i := by
  funext i
  fin_cases i <;> simp [diagonal]

end OodTheoryVerification.Stage10
