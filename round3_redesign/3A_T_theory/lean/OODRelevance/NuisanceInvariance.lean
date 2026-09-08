import Mathlib.Tactic

namespace OODRelevance

/-! The inverse source metric is represented by its induced dual bilinear
    form. A block-diagonal inverse metric on a product is the sum of the two
    block forms. This proves the nuisance theorem in arbitrary dimensions
    without depending on matrix-index APIs. -/

variable {E N : Type*} [AddCommGroup E] [Module ℝ E]
  [AddCommGroup N] [Module ℝ N]

abbrev DualForm (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  V →ₗ[ℝ] V →ₗ[ℝ] ℝ

def blockDual (QE : DualForm E) (QN : DualForm N) (g : E × N) : ℝ :=
  QE g.1 g.1 + QN g.2 g.2

theorem block_nuisance_invariance (QE : DualForm E) (QN : DualForm N) (gE : E) :
    blockDual QE QN (gE, 0) = QE gE gE := by
  simp [blockDual]

theorem block_nuisance_zero_contribution (QN : DualForm N) :
    QN (0 : N) 0 = 0 := by
  simp

end OODRelevance
