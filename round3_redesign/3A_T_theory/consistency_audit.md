# Python / Mathematics / Formalization Consistency Audit

| Python object | Mathematical object | Formalization status |
|---|---|---|
| `source.second` | \(M_S=E_S[XX^\top]\) | `B` in `QuadraticRisk.lean`; symmetry and normal equation are explicit |
| `hessian(source)` | \(H_S=2M_S\) | Normalization checked symbolically; Lean support theorem is stated after metric whitening |
| `analytic_gradient` | \(g_s=2(M_T-M_S)w^*-2(m_T-m_S)\) | Linear part of `shift_polynomial`; residual expectation layer is not formalized |
| `source_excess` | \(\delta^\top M_S\delta\) | `source_excess` theorem compiled |
| `leading_relevance` | \(\sqrt{g^\top H_S^{-1}g}\) | Euclidean dual norm after source-metric whitening via `cauchy_support_isGreatest` |
| `local_vulnerability` | \(\sqrt{2\epsilon}\sqrt{g^\top H_S^{-1}g}\) | Radius-times-dual-norm support maximum after whitening |
| `response_polynomial` | \(g^\top\delta+\delta^\top A_s\delta\) | `shift_polynomial` theorem compiled |
| `curvature_coefficient` | \(K_s=\|M_S^{-1/2}A_sM_S^{-1/2}\|_{op}\) | Exact null-gradient scaling formalized; operator-norm identification remains a paper proof |

The exact normalization relation is
\[
L_s=\sqrt{g^\top M_S^{-1}g}
=\sqrt{2}\,\sqrt{g^\top H_S^{-1}g}
=\sqrt{2}\times\texttt{leading\_relevance}.
\]

The Python implementation computes the finite trust-region value using the
`M_S`-whitened quadratic matrix. It does not use `H_S` in that whitening, so
the curvature bound has no accidental factor of one half. Existing numerical
tests and the new symbolic check both confirm this convention.
