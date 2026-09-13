import OodTheoryVerification.Stage9.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace OodTheoryVerification.Stage9

/-! A fully explicit support identity for the product class with A = diag(a,0),
    a > 0. This is the coordinate form of the range/kernel theorem. -/

def productClass (a rho kappa : ℝ) : Set Vec2 := {δ |
  0 ≤ rho ∧ 0 ≤ kappa ∧
  δ 1 ≤ kappa ∧ -kappa ≤ δ 1 ∧
  δ 0 ≤ rho * a ∧ -rho * a ≤ δ 0 }

noncomputable def support (U : Set Vec2) (g : Vec2) : ℝ :=
  sSup {v : ℝ | ∃ δ ∈ U, v = dot g δ}

theorem product_support_upper (a rho kappa : ℝ) (ha : 0 < a)
    (hr : 0 ≤ rho) (hk : 0 ≤ kappa) (g : Vec2) :
    ∀ δ ∈ productClass a rho kappa,
      dot g δ ≤ rho * a * |g 0| + kappa * |g 1| := by
  intro δ hδ
  rcases hδ with ⟨_, _, hδ1, hδ1neg, hδ0, hδ0neg⟩
  have h0 : g 0 * δ 0 ≤ rho * a * |g 0| := by
    by_cases hg : 0 ≤ g 0
    · rw [abs_of_nonneg hg]
      nlinarith
    · have hg' : g 0 ≤ 0 := le_of_not_ge hg
      rw [abs_of_nonpos hg']
      nlinarith
  have h1 : g 1 * δ 1 ≤ kappa * |g 1| := by
    by_cases hg : 0 ≤ g 1
    · rw [abs_of_nonneg hg]
      nlinarith
    · have hg' : g 1 ≤ 0 := le_of_not_ge hg
      rw [abs_of_nonpos hg']
      nlinarith
  simpa [dot, Fin.sum_univ_two] using add_le_add h0 h1

theorem product_support_exact (a rho kappa : ℝ) (ha : 0 < a)
    (hr : 0 ≤ rho) (hk : 0 ≤ kappa) (g : Vec2) :
    support (productClass a rho kappa) g =
      rho * a * |g 0| + kappa * |g 1| := by
  let b : ℝ := rho * a * |g 0| + kappa * |g 1|
  let S : Set ℝ := {v : ℝ | ∃ δ ∈ productClass a rho kappa, v = dot g δ}
  let δ : Vec2 := fun i => if i = 0 then rho * a * (if 0 ≤ g 0 then 1 else -1) else
    kappa * (if 0 ≤ g 1 then 1 else -1)
  have h_upper : ∀ v ∈ S, v ≤ b := by
    intro v hv
    rcases hv with ⟨δ, hδ, rfl⟩
    exact product_support_upper a rho kappa ha hr hk g δ hδ
  have h_nonempty : S.Nonempty := by
    have hδ : δ ∈ productClass a rho kappa := by
      change 0 ≤ rho ∧ 0 ≤ kappa ∧ δ 1 ≤ kappa ∧ -kappa ≤ δ 1 ∧
        δ 0 ≤ rho * a ∧ -rho * a ≤ δ 0
      constructor
      · exact hr
      constructor
      · exact hk
      constructor
      · by_cases hh : 0 ≤ g 1 <;> simp [δ, hh] <;> nlinarith [hk]
      constructor
      · by_cases hh : 0 ≤ g 1 <;> simp [δ, hh] <;> nlinarith [hk]
      constructor
      · by_cases hg : 0 ≤ g 0 <;> simp [δ, hg] <;> positivity
      · by_cases hg : 0 ≤ g 0 <;> simp [δ, hg] <;> positivity
    refine ⟨dot g δ, ?_⟩
    exact ⟨δ, hδ, rfl⟩
  have h_attain : ∃ v ∈ S, v = b := by
    have hδ : δ ∈ productClass a rho kappa := by
      change 0 ≤ rho ∧ 0 ≤ kappa ∧ δ 1 ≤ kappa ∧ -kappa ≤ δ 1 ∧
        δ 0 ≤ rho * a ∧ -rho * a ≤ δ 0
      constructor
      · exact hr
      constructor
      · exact hk
      constructor
      · by_cases hh : 0 ≤ g 1 <;> simp [δ, hh] <;> nlinarith [hk]
      constructor
      · by_cases hh : 0 ≤ g 1 <;> simp [δ, hh] <;> nlinarith [hk]
      constructor
      · by_cases hg : 0 ≤ g 0 <;> simp [δ, hg] <;> positivity
      · by_cases hg : 0 ≤ g 0 <;> simp [δ, hg] <;> positivity
    refine ⟨b, ⟨δ, hδ, ?_⟩, rfl⟩
    dsimp [b, δ]
    simp [dot, Fin.sum_univ_two]
    by_cases h0 : 0 ≤ g 0
    · have ha0 : |g 0| = g 0 := abs_of_nonneg h0
      by_cases h1 : 0 ≤ g 1
      · have ha1 : |g 1| = g 1 := abs_of_nonneg h1
        simp [h0, h1, ha0, ha1, dot, Fin.sum_univ_two]
        ring
      · have ha1 : |g 1| = -g 1 := abs_of_neg (lt_of_not_ge h1)
        simp [h0, h1, ha0, ha1, dot, Fin.sum_univ_two]
        ring
    · have ha0 : |g 0| = -g 0 := abs_of_neg (lt_of_not_ge h0)
      by_cases h1 : 0 ≤ g 1
      · have ha1 : |g 1| = g 1 := abs_of_nonneg h1
        simp [h0, h1, ha0, ha1, dot, Fin.sum_univ_two]
        ring
      · have ha1 : |g 1| = -g 1 := abs_of_neg (lt_of_not_ge h1)
        simp [h0, h1, ha0, ha1, dot, Fin.sum_univ_two]
        ring
  apply csSup_eq_of_forall_le_of_forall_lt_exists_gt h_nonempty h_upper
  intro w hw
  rcases h_attain with ⟨v, hv, hvb⟩
  exact ⟨v, hv, by linarith⟩

end OodTheoryVerification.Stage9
