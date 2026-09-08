import Mathlib.Tactic

namespace OODRelevance

/-! Compact algebraic facts used by the 3E-C sharp geometry audit.  Norms,
top singular values, and PSD order remain in the paper/Python layer. -/

theorem symmetric_two_point_square (z a : ℝ) :
    ((z + a) ^ 2 + (z - a) ^ 2) / 2 = z ^ 2 + a ^ 2 := by
  ring

theorem static_tax_scalar_lower_bound (z a : ℝ) :
    max ((z + a) ^ 2) ((z - a) ^ 2) / 2 ≥ (z ^ 2 + a ^ 2) / 2 := by
  have h := le_max_left ((z + a) ^ 2) ((z - a) ^ 2)
  have h' := le_max_right ((z + a) ^ 2) ((z - a) ^ 2)
  have hs : ((z + a) ^ 2 + (z - a) ^ 2) / 2 = z ^ 2 + a ^ 2 :=
    symmetric_two_point_square z a
  nlinarith

theorem projection_complement_split
    {V : Type*} [AddCommGroup V] [Module ℝ V]
    (P Q : V →ₗ[ℝ] V) (hPQ : P + Q = 1) (v : V) :
    P v + Q v = v := by
  have h := congrArg (fun L : V →ₗ[ℝ] V => L v) hPQ
  simpa using h

theorem right_zero_from_projection_split
    {V W : Type*} [AddCommGroup V] [Module ℝ V]
    [AddCommGroup W] [Module ℝ W]
    (E : V →ₗ[ℝ] W) (P Q : V →ₗ[ℝ] V)
    (hE : E.comp P = 0) (hPQ : P + Q = 1) :
    E = E.comp Q := by
  ext v
  have hsplit := projection_complement_split P Q hPQ v
  have hzero : E (P v) = 0 := by
    have h := congrArg (fun L : V →ₗ[ℝ] W => L v) hE
    simpa using h
  change E v = E (Q v)
  calc
    E v = E (P v + Q v) := by rw [hsplit]
    _ = E (P v) + E (Q v) := by rw [map_add]
    _ = E (Q v) := by rw [hzero, zero_add]

theorem gram_sum_of_cross_zero
    {m n : Type*} [Fintype m] [Fintype n]
    (A E : Matrix m n ℝ)
    (hAE : A * E.transpose = 0)
    (hEA : E * A.transpose = 0) :
    (A + E) * (A + E).transpose =
      A * A.transpose + E * E.transpose := by
  simp only [Matrix.transpose_add, Matrix.mul_add, Matrix.add_mul]
  rw [hAE, hEA]
  simp

end OODRelevance
