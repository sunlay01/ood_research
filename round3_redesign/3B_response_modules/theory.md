# 3B Response-Module Theory

## T1: direct-sum response decomposition

Let `U_1,...,U_M` be subspaces of a response vector space. If their sum is a
direct sum, every `u` in `U_1+...+U_M` has a unique representation
`u=u_1+...+u_M` with `u_m in U_m`. Existence follows from membership in the
sum. If two representations exist, subtracting them gives a vector in the
direct sum equal to zero; directness forces every component difference to be
zero.

This is response-level identifiability only. It becomes semantic
identifiability only if the structural-to-response map is itself justified.

## T2: overlap non-identifiability

If `U_1 intersect U_2` contains a nonzero `v`, then `v+0` and `0+v` are two
different module contributions with the same observed response. If
`U_1=U_2`, no response-only procedure can decide which label generated a
response. This is a strict negative theorem, not a failure of clustering.

## T3: additive mixed shifts

For a linear local response operator `A_S`,
\[
g_{s_1+s_2}=A_S(s_1+s_2)=A_S(s_1)+A_S(s_2)=g_{s_1}+g_{s_2}.
\]
The corresponding whitened responses add as well. Nonlinear environment
parameterizations must report an interaction residual rather than silently
using this equality.

## T4: intrinsic coordinate invariance

Under `H'=T^{-T}HT^{-1}` and `g_i'=T^{-T}g_i`,
\[
g_i'^T(H')^{-1}g_j'=g_i^TH^{-1}g_j.
\]
Therefore the full relevance Gram matrix, its rank, normalized similarities,
and the angles between response subspaces are invariant. Coordinate singular
vectors are not intrinsic objects.

## What is actually estimated

The runner estimates response rank, module assignments, internal SVD ranks,
bootstrap co-membership stability and held-out mixed-shift reconstruction.
Point clustering is a baseline; the subspace report is primary. These are
diagnostics, not theorems about a hidden causal decomposition.

## Taxonomy

`direct-sum uniqueness` is a `theorem` about response vectors.  `overlap
ambiguity` is a `counterexample` to response-only identification.  The
whitened Gram and additive mixed-shift identities are `exact equality` results
under the stated population-moment construction.  Numerical rank, selected
module count, bootstrap stability, ARI/NMI, and mixed reconstruction are
`diagnostic` quantities.  The current benchmark is deliberately classified as
`LOW-RANK-BUT-NONSEMANTIC` because its low-dimensional response geometry does
not meet the post-hoc semantic alignment gate.
