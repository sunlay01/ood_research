# Stage 13R: Environmental Sensitivity Master-Functional Revision

## 1. Decision and scientific question

Stage 13 found a real but partial common calculus: risk-level methods share the
finite-dimensional affine state, fixed-state MMD/CORAL have projection bridges,
ideal IRM is structural, while IRMv1 and Fishr are not universally exact. This
revision tests whether an environment-side derivative gives a stronger common
functional without changing the state separately per algorithm.

**Decision: `REVISE-ESF-MASTER`.** The environmental-sensitivity functional is
coherent and produces a genuine path-integral risk theorem, and several methods
have natural exact/dual/projection translations. A source-only global theorem
still needs an explicit path-control assumption, and derivative methods remain
restricted/surrogate cases. The result is not a new regularizer.

## 2. Frozen master functional

Let `V` be a finite-dimensional environment/distribution state space and let

```text
R : F x V -> R,
S_D(f; xi) = sup_{delta in D(xi)} D_xi R(f,xi)[delta].
```

`D(xi)` is a predeclared tangent family, independent of the native algorithm.
The derivative is with respect to the environment state, not model
parameters. The state and tangent gauge are fixed before method translation.

## 3. Gate A: coherence

### Coordinate covariance

For `xi'=T xi`, tangent vectors transform as `delta'=T delta` and covectors as
`D_{xi'}R=T^{-T}D_xiR`. With `D'(xi')=T D(xi)`,

```text
D_{xi'}R[delta'] = D_xiR[delta],
S_{D'}(f;xi') = S_D(f;xi).
```

Thus ESF is coordinate invariant only when state, dual, and tangent geometry
are transformed together.

### Affine reduction

If `R(f,xi)=b_f+g_f(xi)`, then `D_xiR=g_f` and

```text
S_D(f;xi) = sup_{delta in D(xi)} g_f(delta) = h_{D(xi)}(g_f).
```

This is exactly the Stage 12 support certificate, not a parallel theory.

### Nonlinear remainder

If `R(f,.)` is C2 on the segment `xi+t delta`,

```text
R(f,xi+delta)=R(f,xi)+D_xiR(f,xi)[delta]+Rem_f(xi,delta),
|Rem_f| <= (L_f/2)||delta||^2
```

whenever the Hessian operator norm is at most `L_f` on that segment. The bound
is local unless a uniform region and uniform `L_f` are declared.

Gate A passes as a coherent extension/reparameterization of Stage 12, with the
nonlinear residual explicitly exposed.

## 4. Gate B: path-integral OOD theorem

Let `gamma:[0,1]->V` be absolutely continuous, with `gamma(0)=xi_S` and
`gamma(1)=xi_T`, and assume `t -> R(f,gamma(t))` is absolutely continuous.
The chain rule gives

```text
R(f,xi_T)-R(f,xi_S)
  = integral_0^1 D_xi R(f,gamma(t))[gamma_dot(t)] dt.
```

If `D(xi)` is the unit ball of a gauge `p_xi`, then

```text
D_xiR[gamma_dot] <= S_D(f;gamma(t)) p_gamma(t)(gamma_dot(t))
```

and therefore

```text
R_T(f)-R_S(f)
 <= integral_0^1 S_D(f;gamma(t)) p_gamma(t)(gamma_dot(t)) dt.
```

This is a genuine local-to-global theorem: it separates path length, local
risk sensitivity, and the target path. For the affine model it reduces to the
support bound along a path. A source-only certificate requires an additional
assumption that the path remains in a region where `S_D` is bounded or that a
source-estimable envelope controls it.

### Negative control

Take `R(f,xi)=a xi^2`, source state `xi_S=0`, target `xi_T=M`, and a tangent
gauge `p(delta)=|delta|`. The local ESF at the observed source is zero, while
`R(f,M)-R(f,0)=aM^2` is arbitrarily large. Along the path, the integrated ESF
is `2aM^2`, so the theorem is correct but source-local sensitivity alone is not
a global robustness certificate.

Gate B passes as a path theorem but fails as a source-only theorem without path
coverage/control assumptions. This is the main reason the overall decision is
`REVISE-ESF-MASTER`.

## 5. Gate C: frozen-ESF method translations

| Method | Frozen-ESF translation | Status | Required bridge/limitation |
|---|---|---|---|
| ERM | controls only `R_S`; ESF is supplied by an external target/path family | `NO-NATURAL-BRIDGE` to sensitivity | no source-only path control |
| V-REx | predeclared source tangent ellipsoid gives `S_D=rho sqrt(g(A g))=rho sqrt(Var_e R_e)` | `DUAL` | exact only in affine state and declared ellipsoid |
| GroupDRO | finite displacement support over `conv{xi_e}` | `DUAL` | finite support is not infinitesimal ESF; outside hull uncontrolled |
| MM-REx | bounded affine coefficient family is an exact support of extrapolated paths | `EXACT` under declared family | negative weights/radius must be explicit |
| fixed-state MMD | primal state displacement with `|D_xiR[delta]|<=||D_xiR||_*||delta||` | `UPPER-BOUND` | fixed witness class and target radius |
| CORAL | second-moment projection of `delta`; projected ESF | `PROJECTION-RELAXATION` | conditional/label residual remains |
| ideal IRM | common parameter optimum plus mixed derivative `D_xiD_thetaR` can restrict admissible mechanism paths | `SURROGATE-WITH-CONDITIONS` | structural identifiability and curvature |
| IRMv1 | fixed-scale `||D_theta R||^2` is not `S_D`; mixed-derivative control gives only a local surrogate | `SURROGATE-WITH-CONDITIONS` | stationarity, scale, optimization and path assumptions |
| Fishr | gradient covariance probes `D_xi Cov(D_theta ell)` in a richer state | `NO-NATURAL-BRIDGE` in minimal state | higher moments and covariance-to-ESF theorem absent |

The same ESF definition is used throughout. The tangent family is not changed
per algorithm; what changes is whether the native objective exactly observes,
dually bounds, projects, or fails to identify ESF.

## 6. Gate D: separation from Moment Alignment

Moment Alignment (Chen, Si, Zhang & Zhao, arXiv:2506.07378, UAI 2025) defines
transfer measures and bounds them using parameter-side derivatives
`D_theta^k L_mu(theta*)` under convex-hull/IRM or bounded-gradient assumptions.
It unifies IRM, gradient matching, Hessian matching and feature-moment duality.

ESF instead fixes the environment derivative `D_xi R`; parameter derivatives
enter only through mixed objects `D_xiD_theta^k R` when a derivative method is
audited. The target quantity is an integrated environment-path risk change,
not Moment Alignment's excess-risk transfer measure.

### All-order separation witness

Let

```text
R(theta,xi)=q(theta)+xi,
Rtilde(theta,xi)=q(theta)+2 xi,
q(theta)=theta^2.
```

For every source environment `xi_e` and every order `k>=1`,
`D_theta^k R(theta,xi_e)=D_theta^k Rtilde(theta,xi_e)` (indeed for every `xi`),
so any all-order parameter-alignment statistic is identical. But
`D_xi R=1` and `D_xi Rtilde=2`, and at target `xi_T=1` the risks differ by one.
Thus parameter-side moment alignment cannot identify environment-side risk
motion. This is a rigorous separation of information, not a claim that one
method dominates the other.

Gate D passes: ESF is not merely Moment Alignment with a renamed derivative.
The separation does not establish that every ESF component is novel; the
support, path, and derivative ingredients have neighboring prior art.

## 7. Gate E: cross-method consequences

### Shared path-control barrier

Any method whose population statistic depends only on source risk responses
factors through the source-observable quotient `V*/S°`. Therefore ERM, V-REx,
GroupDRO and bounded MM-REx all share the same impossibility for unrestricted
target paths leaving the source span, even though their native objectives are
different. ESF makes this a path statement: finite source observations do not
bound the integral over an uncontrolled segment.

### Support ordering

If two declared tangent/target families satisfy `D_1(xi) subset D_2(xi)` along a
path, then `S_{D_1}(f;xi)<=S_{D_2}(f;xi)` pointwise and the integrated ESF bound
is ordered. GroupDRO, MM-REx and finite-mixture DRO are therefore comparable by
uncertainty-family inclusion, not by algorithm names.

### Projection warning

CORAL or marginal MMD can make a primal projection small while the conditional
part of `D_xiR` remains large. The same label-channel counterexample from Stage
13 is a cross-method negative control for ESF.

These are genuine consequences of combining local sensitivity, path integration,
and the source quotient; they do not imply a universal exact bridge.

## 8. Proof status and required revision

Paper proofs: Gate A covariance/remainder, Gate B chain-rule path theorem, the
Moment Alignment separation, and the cross-method ordering/barrier theorems.
Deterministic checks: `experiments/stage13r_esf_master_tests.py`.
Lean checks: `lean/OodTheoryVerification/Stage13R/Basic.lean` contains the
finite affine ESF/support identity and a concrete path-difference algebraic
witness. It does not formalize general absolute continuity, Bochner integration,
or Moment Alignment.

Before Stage 14, revise ESF in one of two ways: (i) declare a source-estimable
path envelope/coverage condition and prove a non-vacuous integrated certificate,
or (ii) scope ESF explicitly as a local/path-oracle theory rather than a
source-only OOD theorem. Do not design a new regularizer in this stage.

## 9. Final verdict

`REVISE-ESF-MASTER`.

The master functional is mathematically coherent, exactly reduces to Stage 12
in affine models, and is separated from Moment Alignment by an all-order
parameter-indistinguishability witness. It also gives natural mappings for
multiple method families. The missing piece is a non-vacuous source-only control
of the environmental path, while IRMv1/Fishr remain restricted or absent. The
next stage must repair this interface before any bound-derived method or Stage
14 extension.
