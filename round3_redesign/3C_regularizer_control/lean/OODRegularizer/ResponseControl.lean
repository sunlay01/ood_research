import Mathlib.Tactic

namespace OODRegularizer

theorem exact_blind_direction (q k lambda : ℝ)
    (hk : k = 0) :
    q * ((1 + lambda * k)⁻¹ * q) / (q * q) = 1 := by
  subst k
  by_cases hq : q = 0
  · simp [hq]
  · field_simp [hq]

theorem infinitesimal_control_derivative (q k : ℝ) (hq : q ≠ 0) :
    (fun lambda : ℝ => (q * ((1 + lambda * k)⁻¹ * q)) / (q * q))'
      0 = -k := by
  field_simp [hq]
  ring

theorem psd_scalar_monotone (q k lambda₁ lambda₂ : ℝ)
    (hk : 0 ≤ k) (h1 : 0 ≤ lambda₁) (hle : lambda₁ ≤ lambda₂)
    (h2 : 0 < 1 + lambda₁ * k) :
    q * ((1 + lambda₂ * k)⁻¹ * q) / (q * q) ≤
      q * ((1 + lambda₁ * k)⁻¹ * q) / (q * q) := by
  by_cases hq : q = 0
  · simp [hq]
  have h2pos : 0 < 1 + lambda₂ * k := by nlinarith
  have hinv : (1 + lambda₂ * k)⁻¹ ≤ (1 + lambda₁ * k)⁻¹ := by
    exact inv_le_inv₀ (by positivity) (by nlinarith) (by nlinarith)
  have hq2 : 0 ≤ q * q := sq_nonneg q
  apply (div_le_div_iff₀ (by positivity) (by positivity)).2
  nlinarith

end OODRegularizer
