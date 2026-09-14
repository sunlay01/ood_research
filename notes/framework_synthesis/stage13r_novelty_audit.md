# Stage 13R novelty audit

## Search protocol

The audit searched primary/official sources and the repository ledger for
2023--2026 work using the following exact families: environmental/distributional
risk derivatives; local distributional robustness and tangent DRO; path-integral
risk bounds; influence/Gateaux views of domain shift; parameter/environment
mixed derivatives; and Moment Alignment follow-ups. The primary Moment Alignment
paper was read from arXiv:2506.07378, not inferred from a title or abstract.

## Evidence classification

| Component | Closest evidence | Classification |
|---|---|---|
| `S_D=sup_{delta in D}D_xiR[delta]` | support functions and local robust optimization | component-level / often equivalent |
| path integral of local risk derivative | calculus/transport and pathwise sensitivity literature | neighboring mathematical idea; no exact DG composition found |
| source-state exposure operator | Stage 12/V-REx and REx risk variance; covariance geometry | strong component overlap |
| source quotient and blind fiber | optimal recovery and partial identification | strong structural overlap |
| parameter derivative moment alignment | Chen et al. 2025 Moment Alignment | distinct primitive; direct separation witness above |
| complete `state -> D_xiR -> path certificate -> multi-regularizer map` | no exact end-to-end match found in checked sources | negative evidence only, not proof of novelty |

Recent OOD theory inspected through official records includes transformation-
class OOD bounds (Montasser, Shao & Abbe, NeurIPS 2024), meta-distribution/domain
count theory (Dwork, Hu & Shao, NeurIPS 2025), distance-dimension theory
(Bhattacharjee, Rittler & Chaudhuri, ALT 2026), and non-stationary DG (Pham et
al., UAI 2024). These use transformation classes, domain shattering, shared
feature projections, or temporal evolution; none was found to state the exact
ESF package. This is neighboring/negative evidence, not a novelty certificate.

## Referee attack

- If `D` is reverse-engineered per algorithm, ESF is invalid; this audit freezes
  one tangent family before translation.
- If the path is chosen using target data, the result is oracle/assumption-level,
  not source-only; the report labels this explicitly.
- If `S_D` is called a global robustness certificate from source-local values,
  the quadratic negative control falsifies it.
- If parameter and environment derivatives are conflated, the all-order witness
  separates them.
- If nonlinear remainder is omitted, the C2 Taylor bound is required.
- Singular source exposure and learned representations remain open interfaces.

## Novelty conclusion

`No exact end-to-end overlap found, but strong component-level overlap.` ESF is
not justified as a new primitive mathematical object. Its possible contribution
is an OOD-specific theorem organization and cross-method separation results.
The path-control gap and restricted derivative bridges require `REVISE-ESF-MASTER`.
