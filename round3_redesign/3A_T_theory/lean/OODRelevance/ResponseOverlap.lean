import Mathlib.Tactic

namespace OODRelevance

variable {V : Type*} [AddCommGroup V]

theorem overlap_nonidentifiability {v : V} (hv : v ≠ 0) :
    v + 0 = 0 + v ∧ (v, 0) ≠ (0, v) := by
  constructor
  · simp
  · intro h
    have : v = 0 := congrArg Prod.fst h
    exact hv this

theorem identical_response_images_are_not_label_identifiable
    {A B : V → V} (himages : ∀ x, ∃ y, A x = B y)
    (a : V) : ∃ b : V, A a = B b := himages a

end OODRelevance
