import Mathlib.Tactic

namespace OODRegularizer

theorem local_quadratic_minimizer (h j a delta lambda : ℝ)
    (hpos : h + lambda * j > 0)
    (hstationary : (h + lambda * j) * delta + lambda * a = 0) :
    delta = -lambda * a / (h + lambda * j) := by
  apply (eq_div_iff (ne_of_gt hpos)).2
  linarith

theorem whitened_minimizer_identity (h j a lambda : ℝ)
    (hpos : h > 0) (hmetric : h + lambda * j > 0) :
    Real.sqrt h * (-lambda * a / (h + lambda * j)) =
      -lambda * ((a / Real.sqrt h) / (1 + lambda * (j / h))) := by
  have hsqrt : Real.sqrt h ≠ 0 := by positivity
  have hne : h ≠ 0 := ne_of_gt hpos
  have hsquare : Real.sqrt h * Real.sqrt h = h := by
    rw [Real.mul_self_sqrt (le_of_lt hpos)]
  field_simp [hsqrt, hne]
  nlinarith

theorem steering_identity (q b k lambda : ℝ) :
    q * (-lambda * ((b) / (1 + lambda * k))) =
      -lambda * q * ((1 + lambda * k)⁻¹ * b) := by
  field_simp
  ring

theorem curvature_ratio_identity (q k lambda : ℝ)
    (hne : 1 + lambda * k ≠ 0) (hq : q ≠ 0) :
    (q * ((1 + lambda * k)⁻¹ * q)) / (q * q) = (1 + lambda * k)⁻¹ := by
  field_simp [hne, hq]
  ring

end OODRegularizer
