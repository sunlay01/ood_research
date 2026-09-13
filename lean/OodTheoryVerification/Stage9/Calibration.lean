import OodTheoryVerification.Stage9.Basic
import Mathlib.Tactic

namespace OodTheoryVerification.Stage9

theorem scaled_diagonal_range (a b eps : ℝ) (heps : eps ≠ 0) :
    (∀ x : Vec2, rangeDiagonal (eps^2 * a) (eps^2 * b) x =
      rangeDiagonal a b x) := by
  intro x
  funext i
  fin_cases i <;> simp [rangeDiagonal, heps]

theorem scaled_diagonal_kernel (a b eps : ℝ) (heps : eps ≠ 0) :
    (∀ x : Vec2, kernelDiagonal (eps^2 * a) (eps^2 * b) x =
      kernelDiagonal a b x) := by
  intro x
  funext i
  fin_cases i <;> simp [kernelDiagonal, heps]

end OodTheoryVerification.Stage9
