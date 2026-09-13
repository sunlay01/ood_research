import Mathlib.Data.Real.Basic
import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic

namespace OodTheoryVerification.Stage9

/-! Explicit two-dimensional definitions used for the first machine-checked
    Stage 9 slice. The scientific target family is kept as data, independent
    of the operator. -/

abbrev Vec2 := Fin 2 → ℝ

def dot (x y : Vec2) : ℝ := ∑ i, x i * y i

def normSq (x : Vec2) : ℝ := dot x x

def diagonal (a b : ℝ) (x : Vec2) : Vec2
  | 0 => a * x 0
  | 1 => b * x 1

noncomputable def rangeDiagonal (a b : ℝ) (x : Vec2) : Vec2
  | 0 => if a ≠ 0 then x 0 else 0
  | 1 => if b ≠ 0 then x 1 else 0

noncomputable def kernelDiagonal (a b : ℝ) (x : Vec2) : Vec2
  | 0 => if a = 0 then x 0 else 0
  | 1 => if b = 0 then x 1 else 0

theorem dot_nonneg (x : Vec2) : 0 ≤ normSq x := by
  simp [normSq, dot, Fin.sum_univ_two]
  exact add_nonneg (mul_self_nonneg _) (mul_self_nonneg _)

theorem range_kernel_decomp (a b : ℝ) (x : Vec2)
    (ha : 0 ≤ a) (hb : 0 ≤ b) :
    (∀ i, diagonal a b (rangeDiagonal a b x) i = diagonal a b x i) ∧
    (∀ i, diagonal a b (kernelDiagonal a b x) i = 0) := by
  constructor
  · intro i
    fin_cases i
    · by_cases h : a = 0 <;> simp [diagonal, rangeDiagonal, h]
    · by_cases h : b = 0 <;> simp [diagonal, rangeDiagonal, h]
  · intro i
    fin_cases i
    · by_cases h : a = 0 <;> simp [diagonal, kernelDiagonal, h]
    · by_cases h : b = 0 <;> simp [diagonal, kernelDiagonal, h]

theorem zero_kernel_coverage_iff (a b : ℝ) (U : Set Vec2)
    (ha : 0 ≤ a) (hb : 0 ≤ b) :
    (∀ x ∈ U, kernelDiagonal a b x = 0) ↔
      (∀ x ∈ U, (if a = 0 then x 0 else 0) = 0 ∧
        (if b = 0 then x 1 else 0) = 0) := by
  constructor <;> intro h x hx
  · constructor
    · simpa [kernelDiagonal] using congrFun (h x hx) 0
    · simpa [kernelDiagonal] using congrFun (h x hx) 1
  · funext i
    fin_cases i
    · by_cases ha0 : a = 0
      · simpa [kernelDiagonal, ha0] using (h x hx).1
      · simp [kernelDiagonal, ha0]
    · by_cases hb0 : b = 0
      · simpa [kernelDiagonal, hb0] using (h x hx).2
      · simp [kernelDiagonal, hb0]

end OodTheoryVerification.Stage9
