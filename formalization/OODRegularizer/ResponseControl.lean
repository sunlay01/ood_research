import Mathlib.Tactic
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.Deriv.Mul
import OODRegularizer.LocalAction

namespace OODRegularizer

theorem exact_blind_direction (q k lambda : ℝ)
    (hk : k = 0) (hq : q ≠ 0) :
    q * ((1 + lambda * k)⁻¹ * q) / (q * q) = 1 := by
  subst k
  simpa using curvature_ratio_identity q 0 lambda (by norm_num) hq

theorem infinitesimal_control_derivative (q k : ℝ) (hq : q ≠ 0) :
    deriv (fun lambda : ℝ => (q * ((1 + lambda * k)⁻¹ * q)) / (q * q))
      0 = -k := by
  have hratio :
      (fun lambda : ℝ => (q * ((1 + lambda * k)⁻¹ * q)) / (q * q)) =
        (fun lambda : ℝ => (1 + lambda * k)⁻¹) := by
    funext lambda
    field_simp [hq]
  rw [hratio]
  have h :=
    ((hasDerivAt_const (𝕜 := ℝ) 0 1).add
      ((hasDerivAt_id 0).mul_const k)).inv (by norm_num)
  simpa using h.deriv

theorem psd_scalar_monotone (q k lambda₁ lambda₂ : ℝ)
    (hk : 0 ≤ k) (h1 : 0 ≤ lambda₁) (hle : lambda₁ ≤ lambda₂)
    (h2 : 0 < 1 + lambda₁ * k) :
    q * ((1 + lambda₂ * k)⁻¹ * q) / (q * q) ≤
      q * ((1 + lambda₁ * k)⁻¹ * q) / (q * q) := by
  by_cases hq : q = 0
  · simp [hq]
  have h2pos : 0 < 1 + lambda₂ * k := by nlinarith
  have hinv : (1 + lambda₂ * k)⁻¹ ≤ (1 + lambda₁ * k)⁻¹ := by
    exact (inv_le_inv₀ h2pos h2).2 (by nlinarith)
  rw [curvature_ratio_identity q k lambda₂ (ne_of_gt h2pos) hq,
    curvature_ratio_identity q k lambda₁ (ne_of_gt h2) hq]
  exact hinv

end OODRegularizer
