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

noncomputable def exposure {m : ℕ} (d : Fin m → Vec2) (g : Vec2) : Vec2 :=
  (m : ℝ)⁻¹ • ∑ e, (response g (d e)) • d e

theorem exposure_energy {m : ℕ} (hm : (m : ℝ) ≠ 0)
    (d : Fin m → Vec2) (g : Vec2) :
    response g (exposure d g) =
      (m : ℝ)⁻¹ * ∑ e, (response g (d e)) ^ 2 := by
  simp [exposure, response, dot, Fin.sum_univ_two, Finset.sum_apply]
  simp_rw [Finset.mul_sum]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro e he
  ring

theorem exposure_kernel_iff {m : ℕ} (hm : (m : ℝ) ≠ 0)
    (d : Fin m → Vec2) (g : Vec2) :
    exposure d g = 0 ↔ ∀ e, response g (d e) = 0 := by
  constructor
  · intro h e
    have henergy : response g (exposure d g) = 0 := by rw [h]; simp [response, dot]
    have hsum : ∑ e, (response g (d e)) ^ 2 = 0 := by
      rw [exposure_energy hm d g] at henergy
      exact (mul_eq_zero.mp henergy).resolve_left (inv_ne_zero hm)
    have hnonneg : ∀ e ∈ (Finset.univ : Finset (Fin m)), 0 ≤ (response g (d e)) ^ 2 := by
      intro e he
      exact sq_nonneg _
    have hz := (Finset.sum_eq_zero_iff_of_nonneg hnonneg).mp hsum e (Finset.mem_univ e)
    exact (sq_eq_zero_iff).mp hz
  · intro h
    unfold exposure
    apply smul_eq_zero.mpr
    right
    apply Finset.sum_eq_zero
    intro e he
    rw [h e]
    simp

def exposureKernel {m : ℕ} (d : Fin m → Vec2) : Set Vec2 :=
  {g | exposure d g = 0}

def sourceAnnihilator {m : ℕ} (d : Fin m → Vec2) : Set Vec2 :=
  {g | ∀ e, response g (d e) = 0}

theorem exposure_kernel_eq_annihilator {m : ℕ} (hm : (m : ℝ) ≠ 0)
    (d : Fin m → Vec2) :
    exposureKernel d = sourceAnnihilator d := by
  ext g
  exact exposure_kernel_iff hm d g

noncomputable def exposureOp {V : Type} [AddCommGroup V] [Module ℝ V]
    {m : ℕ} (d : Fin m → V) (g : V →ₗ[ℝ] ℝ) : V :=
  (m : ℝ)⁻¹ • ∑ e, (g (d e)) • d e

def dualTransform {V : Type} [AddCommGroup V] [Module ℝ V]
    (g : V →ₗ[ℝ] ℝ) (Tinv : V →ₗ[ℝ] V) : V →ₗ[ℝ] ℝ :=
  g.comp Tinv

theorem exposure_pairing_covariant {V : Type} [AddCommGroup V] [Module ℝ V]
    {m : ℕ} (d : Fin m → V) (g : V →ₗ[ℝ] ℝ)
    (T Tinv : V →ₗ[ℝ] V)
    (hleft : Tinv.comp T = LinearMap.id) :
    dualTransform g Tinv
        (exposureOp (fun e => T (d e)) (dualTransform g Tinv)) =
      g (exposureOp d g) := by
  unfold dualTransform exposureOp
  rw [map_smul, map_sum]
  simp_rw [map_smul]
  have hTT : ∀ v : V, Tinv (T v) = v := by
    intro v
    have h := LinearMap.congr_fun hleft v
    simpa using h
  simp_rw [LinearMap.comp_apply, hTT]
  rw [map_sum]
  simp_rw [map_smul]

end OodTheoryVerification.Stage11R
