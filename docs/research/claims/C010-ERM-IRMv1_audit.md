# Audit: C010 ERM--IRMv1 Mechanism Analysis

## Claim

In the declared scalar population SCM, distinguish ERM mixture stationarity from standard IRMv1's radial response operator; identify when a source relation design controls effective nuisance use, when it leaves a blind branch, and how the resulting quantity enters target risk.

## Internal Verdict

`PARTIAL / ALGEBRA_AND_IMPLEMENTATION_PASS`.

- ERM's mixture-gradient condition, relation/mean/covariance telescope, and pre-specified projector split are exact equalities.
- Standard IRMv1 controls `w.T grad R_e(w)`, not the tangent gradient. This is directly recomputed in population code.
- In the restricted scalar centered family, a full-rank quadratic relation design gives the explicit `w_A^2` bound in C010.
- The target-transport inequality is a conditional population bound. Its target moment norm, base residual, and common-SCM assumption cannot be hidden in a source-only penalty.

## Failure Taxonomy

| Failure | Cause | C010 evidence |
| --- | --- | --- |
| ERM nuisance reliance | mixture stationarity allows environment cancellation | two-source ERM has zero mixture gradient but nonzero per-environment radial and tangent responses |
| IRMv1 blind branch | two relation points do not identify a quadratic response | C002-IRM source-optimal, zero-penalty branch |
| zero predictor | scalar radial constraint has a trivial zero | strict source-fit condition selects the nontrivial U-only root |
| target geometry | finite penalty bound contains `||Delta M||` | target coverage/budget remains an explicit theorem input |
| representation coordinate claim | `B,w` are nonidentifiable | effective `theta=B.Tw` and `w.TB_A` are invariant, `B_A` is not |

## Prior-Art Audit

`HIGH-RISK COLLISION / NOT A PAPER CLAIM`.

- Kamath et al. (2021) already give IRMv1 population failures and study the importance of representative environment sets.
- Lai and Wang (2024) reinterpret IRMv1 as a classifier-gradient total variation objective.
- Chen et al. (2025) is the closest derivative/moment transfer analysis and must be compared against the exact scalar quadratic-design bridge.
- Anchor Regression and DRIG remain related uncertainty-set-matched positive controls, not equivalent results for the current relation/moment family.

The possible research value is only the reusable operator/control/blind/error interface if it yields distinct predictions across several regularizers. C010 alone must not be positioned as a new IRMv1 theorem.
