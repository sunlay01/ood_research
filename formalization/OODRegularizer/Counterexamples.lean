import Mathlib.Tactic

namespace OODRegularizer

theorem same_curvature_opposite_steering :
    let q : ℝ := 1
    let k : ℝ := 1
    let aPlus : ℝ := 1
    let aMinus : ℝ := -1
    let lambda : ℝ := 1 / 4
    (-lambda * q * ((1 + lambda * k)⁻¹ * aPlus)) < 0 ∧
      0 < (-lambda * q * ((1 + lambda * k)⁻¹ * aMinus)) := by
  norm_num

end OODRegularizer
