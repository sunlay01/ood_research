import Mathlib.Algebra.Module.LinearMap.Defs
import Mathlib.Tactic

namespace OODRelevance

/-! The deterministic quadratic core.  A bilinear form is represented by a
    linear map into linear maps, so additivity is available to the kernel. -/

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

abbrev BilinearForm (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  V →ₗ[ℝ] V →ₗ[ℝ] ℝ

def risk (B : BilinearForm V) (m : V →ₗ[ℝ] ℝ) (c : ℝ) (w : V) : ℝ :=
  B w w - 2 * m w + c

theorem source_excess (B : BilinearForm V) (m : V →ₗ[ℝ] ℝ) (c : ℝ)
    (w d : V) (hsym : ∀ x y, B x y = B y x) (hnormal : ∀ x, B x w = m x) :
    risk B m c (w + d) - risk B m c w = B d d := by
  simp only [risk, map_add, LinearMap.add_apply]
  rw [hsym w d, hnormal d]
  ring

def shiftDifference (B : BilinearForm V) (m : V →ₗ[ℝ] ℝ) (c : ℝ)
    (B₀ : BilinearForm V) (m₀ : V →ₗ[ℝ] ℝ) (c₀ : ℝ) (w : V) : ℝ :=
  risk B m c w - risk B₀ m₀ c₀ w

theorem shift_polynomial (B B₀ : BilinearForm V) (m m₀ : V →ₗ[ℝ] ℝ)
    (c c₀ : ℝ) (w d : V) :
    shiftDifference B m c B₀ m₀ c₀ (w + d) -
        shiftDifference B m c B₀ m₀ c₀ w =
      ((B - B₀) d w + (B - B₀) w d - 2 * (m - m₀) d) + (B - B₀) d d := by
  simp only [shiftDifference, risk, map_add, LinearMap.add_apply, LinearMap.sub_apply]
  ring

end OODRelevance
