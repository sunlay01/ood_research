# Stage 11R: Metric/Gauge Interface Revision

## 1. Decision and scope

**Decision: `ADVANCE` to the Stage 12 population-theorem gate.** This is a
revision of the Stage 11 interface, not a claim that the population theorem or
any algorithm mapping is complete. The exact supervised state from Stage 11 is
retained. The change is to separate coordinate-free risk transfer from an
optional metric-dependent explanatory decomposition.

## 2. Primal/dual risk representation

Let `V` be the distribution-state space and `V*` its algebraic dual. A domain
shift is `delta in V`; a predictor-dependent risk sensitivity is `g_f in V*`.
The primitive risk perturbation is the pairing

```text
g_f(delta) = <g_f, delta>_{V*,V}.
```

For a fixed finite feature span `phi`, `Psi(P)=E_P[phi(X,Y)]` is in `V` and

```text
R_P(f) = b_f + g_f(Psi(P)) + eta_f(P).
```

The linear-regression witness remains exact with
`Psi(P)=(E[XX^T], E[XY], E[Y^2])` and `g_w=(ww^T,-2w,1)` under the declared
Frobenius/Euclidean pairing. The state is defined before the learner and does
not contain a risk vector.

## 3. Exposure operator without an identified metric

For source shifts `delta_e=Psi(P_e)-barPsi`, define the finite-rank operator

```text
A : V* -> V,
A g = (1/m) sum_e g(delta_e) delta_e.
```

It satisfies the intrinsic identity

```text
g(A g) = (1/m) sum_e g(delta_e)^2.
```

Thus `S_A(g)=sqrt(g(A g))` is a source-exposure seminorm, not an arbitrary
Euclidean norm. Under a state-coordinate map `delta'=T delta` and the dual map
`g'=T^{-T}g`, the operator is `A'=T A T^T` and both the pairing and `S_A` are
unchanged.

This operator records only source-excited directions. It is not asserted to be
invertible or to cover the physical target family.

## 4. Blind directions and the observable quotient

Let `S=span{delta_e}`. The source-annihilator is

```text
S° = {h in V* : h(delta)=0 for every delta in S} = ker A.
```

Source directional responses identify `g` only through the restriction map
`r_S:g -> g|_S`. Equivalently, the observable object is the quotient class

```text
[g] in V*/S°  ~=  S*.
```

If `h in S°`, every source response of `g+h` equals that of `g`. This is an
identifiability statement, not a claim that a large regularization penalty can
recover the blind component. A Euclidean `N_A=||P_{ker A}g||` is therefore not a
primitive learner-controllable quantity and is not source-risk identifiable in
general.

## 5. Fundamental target-risk certificate

For an externally declared admissible target-shift set `U_phys subset V`, the
coordinate-free certificate is the support function

```text
h_U(g) = sup_{delta in U_phys} g(delta).
```

The master interface must be organized as

```text
R_T(f) <= bar R_S(f) + h_Uphys(g_f) + 2 epsilon_repr.
```

The support/gauge expression is fundamental. A split expression involving
`rho` and `kappa` is only a relaxation for a target family that actually
factorizes through exposed and blind components.

## 6. Optional metric-aware explanatory layer

If the scientific model declares a positive-definite primal metric `G`, the
dual metric is `G^{-1}`. Under `delta'=T delta`, it must transform as

```text
G' = T^{-T} G T^{-1}.
```

Then whitening uses `delta_tilde=G^(1/2) delta` and
`g_tilde=G^(-1/2) g`; the pairing is unchanged. Metric-aware projections onto
`S` and `S°`, dual norms, pseudoinverses, principal angles, `rho`, and `kappa`
are legitimate only after this metric is declared and transformed together
with coordinates. In these whitened coordinates the Stage 9/10 Euclidean
lemmas can be reused as metric-relative corollaries.

The previous unrestricted Euclidean `N_A` interpretation is withdrawn. The
metric-aware replacement is a dual-metric projection norm, for example

```text
N_{A,G}(g) = || P^{G^{-1}}_{S°} g ||_{G^{-1}}.
```

Its role is explanatory; the support function remains the sharp target
certificate.

## 7. Revision theorems and proof status

The revision consists of four theorem obligations:

1. primal/dual affine risk representation;
2. exposure covariance and `S_A` invariance under paired coordinate changes;
3. `ker A=S°` and source-observable quotient/indistinguishability;
4. metric/gauge covariance of any split decomposition.

The exact quadratic representation, finite-source exposure-energy identity,
finite-dimensional `exposureKernel = sourceAnnihilator` equality, and an
abstract linear-map paired-covariance theorem are Lean-checked in
`lean/OodTheoryVerification/Stage11R/Basic.lean`. The quotient identification
`V*/S° ~= S*`, metric-aware projection, and support-function statements remain
paper proofs with deterministic numerical witnesses in
`experiments/risk_representation_revision_tests.py`. No Taylor theorem, Stage
12 master theorem, algorithm mapping, or new regularizer is claimed here.

## 8. Deterministic audit

The revision tests check:

- primal/dual pairing and `A`-energy identities;
- covariance `A'=TAT^T` and invariance of `g(A g)`;
- `ker A=S°` and indistinguishable source responses;
- exact target support under transformed coordinates;
- failure of raw Euclidean `N_A` under a shear;
- invariance of the metric-aware dual projection norm under the paired
  transformation;
- strict looseness of the split support relaxation for a coupled target set.

Run:

```text
python notes/framework_synthesis/experiments/risk_representation_revision_tests.py
```

The test suite is deterministic and has no data download, randomness,
optimization, or neural-network training.

## 9. Gate interpretation

Stage 11's risk-representation hypothesis survives. The former `REVISE` was
caused by an interface error: treating a dual space as the same Euclidean
space as the primal state and treating blind sensitivity as directly
observable. After this revision, support/gauge and source-exposure seminorms
are the coordinate-free backbone; metric-dependent split quantities are
explicitly subordinate corollaries. Stage 12 may now formulate the population
master theorem, but Stage 12.5 exact novelty audit remains mandatory before any
bound-derived method or algorithm claim.
