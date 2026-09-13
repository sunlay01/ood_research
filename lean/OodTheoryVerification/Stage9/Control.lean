import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace OodTheoryVerification.Stage9

/-! Monotonicity of the Stage 9 transfer certificate.

    The geometric lemmas bound `rho` and `kappa`; learner-side analysis bounds
    the corresponding sensitivities. This file records the resulting control
    composition independently of any particular regularizer. -/

theorem transfer_term_mono
    (rho kappa s n repr rhoBar kappaBar sBar nBar reprBar : ℝ)
    (hrho : 0 ≤ rho) (hkappa : 0 ≤ kappa)
    (hs : 0 ≤ s) (hn : 0 ≤ n) (hrepr : 0 ≤ repr)
    (hrhoBar : rho ≤ rhoBar) (hkappaBar : kappa ≤ kappaBar)
    (hsBar : s ≤ sBar) (hnBar : n ≤ nBar) (hreprBar : repr ≤ reprBar) :
    rho * s + kappa * n + 2 * repr ≤
      rhoBar * sBar + kappaBar * nBar + 2 * reprBar := by
  have h1 : rho * s ≤ rhoBar * sBar := by
    have hrhoBar0 : 0 ≤ rhoBar := le_trans hrho hrhoBar
    calc
      rho * s ≤ rhoBar * s := mul_le_mul_of_nonneg_right hrhoBar hs
      _ ≤ rhoBar * sBar := mul_le_mul_of_nonneg_left hsBar hrhoBar0
  have h2 : kappa * n ≤ kappaBar * nBar := by
    have hkappaBar0 : 0 ≤ kappaBar := le_trans hkappa hkappaBar
    calc
      kappa * n ≤ kappaBar * n := mul_le_mul_of_nonneg_right hkappaBar hn
      _ ≤ kappaBar * nBar := mul_le_mul_of_nonneg_left hnBar hkappaBar0
  nlinarith

theorem nonvacuous_of_control
    (barRisk transferBound epsilonTransfer : ℝ)
    (hcontrol : transferBound ≤ epsilonTransfer)
    (heps : barRisk + epsilonTransfer < 1) :
    barRisk + transferBound < 1 := by
  linarith

end OodTheoryVerification.Stage9
