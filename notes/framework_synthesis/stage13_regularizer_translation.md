# Stage 13: OOD Regularizer Translation and Mechanism Typing

## 1. Decision and scientific question

Stage 12.5 found no exact end-to-end duplicate, but strong component-level
overlap with robust optimization, optimal recovery, partial identification,
REx, and discrepancy theory. Stage 13 therefore tests a stronger claim than
shared notation:

> Do heterogeneous OOD regularizers act on different, provable components of
> one declared population transfer calculus?

All translations below reuse the same upstream supervised state whenever the
method permits it. A method-specific projection or a richer state is recorded
explicitly rather than hidden. The final decision is **`PARTIAL-UNIFICATION`**:
the common calculus yields exact cross-method relations for risk-level methods
and rigorous restricted bridges for state-geometry and mechanism methods, but
it does not provide one exact state-level theorem for Fishr or unrestricted
deep IRMv1.

## 2. Common translation protocol

Use a fixed finite-dimensional state `Psi(P) in V` and affine risk

```text
R_P(f) = b_f + g_f(Psi(P)) + eta_f(P).
```

For source states `Psi_e`, define `delta_e=Psi_e-barPsi`,
`S=span{delta_e}`, and

```text
A : V* -> V,       A g = (1/m) sum_e g(delta_e) delta_e.
```

Stage 12 supplies the exact transfer identity and target support
`h_U(g)=sup_{delta in U}g(delta)`, with residual `2 epsilon_repr`. A method
translation must therefore identify its controlled population object and then
state the target-family assumption needed to turn that object into a target
certificate.

## 3. Mechanism taxonomy

| Type | Controlled object | Representative methods |
|---|---|---|
| I sensitivity | `g(A g)` or a risk-vector norm | V-REx |
| II robust support | `sup_{delta in U} g(delta)` for declared `U` | GroupDRO, MM-REx |
| III state geometry | a norm/projection of `Psi(P_e)-Psi(P_e')` | fixed-state MMD, CORAL |
| IV mechanism fiber | admissible conditional/mechanism class | ideal IRM, restricted IRMv1 |
| V derivative-rich state | gradient covariance beyond minimal `Psi` | Fishr |

The taxonomy is a theorem map, not a claim that the objectives are
interchangeable.

## 4. ERM baseline

Native population objective:

```text
min_f bar R_S(f).
```

Population object: source mean risk only. ERM supplies no target support or
blind-direction control. Under the Stage 12 assumption `delta_T in U`, the
generic support certificate still applies, but that bridge comes from the
declared target family, not from ERM. Status: **`PROVED-UNDER-RESTRICTIONS`**
(source-fit statement only).

## 5. V-REx exact translation

Native objective:

```text
bar R_S(f) + lambda Var_e R_e(f).
```

In the exact affine state model, `R_e-barR_S=g_f(delta_e)`, hence

```text
Var_e R_e(f) = g_f(A g_f).
```

This is an **`EXACT`** translation, including the normalization convention.
If `U` lies in the exposed ellipsoid
`{delta in S: delta^T A_S^dagger delta <= rho^2}`, Stage 12 gives

```text
R_T(f) <= bar R_S(f) + rho*sqrt(Var_e R_e(f)) + 2 epsilon_repr.
```

Limitation theorem: if `U` contains a direction outside `S`, one can keep all
source risk responses (and therefore V-REx) fixed while changing the dual
extension on that direction without bound. V-REx controls exposed sensitivity,
not blind sensitivity.

## 6. GroupDRO exact translation

For target mixtures `Psi_alpha=sum_e alpha_e Psi_e`, `alpha` in the simplex,
the affine model gives

```text
R_alpha(f)=sum_e alpha_e R_e(f),
sup_alpha R_alpha(f)=max_e R_e(f).
```

With `U_conv=conv{delta_e}`,

```text
max_e R_e(f) = bar R_S(f) + h_Uconv(g_f)
```

when residuals are zero. Status: **`EXACT`** for the observed finite-group
convex hull. The target bridge is exact only for targets in that hull; a target
state outside it is not determined by the worst source-group risk.

## 7. MM-REx extrapolation translation

To make the uncertainty family explicit, use the affine coefficient set

```text
B_r = { beta in R^m : 1^T beta=0, ||beta||_2 <= r },
Psi_MM(beta)=barPsi + sum_e beta_e delta_e.
```

Negative mixture weights are allowed by this extrapolation family. For source
risk vector `r_S=(R_e)`,

```text
sup_beta sum_e (1/m+beta_e) R_e
  = bar R_S + r ||r_S - bar R_S*1||_2.
```

This is exact for the declared unconstrained affine `l2` family. If a native
implementation clips weights, imposes nonnegativity, or uses a different
radius, the equality changes; the correct status for the practical family is
**`PROVED-UNDER-RESTRICTIONS`**. `U_MM` is an affine-hull support, not the
GroupDRO convex hull. It can be unbounded if the coefficient set has no radius.

## 8. Stage 13A gate

V-REx and finite-group GroupDRO have exact translations. MM-REx has an exact
translation only after its coefficient set is declared. All three reuse the
same affine risk state and each reaches a Stage 12 target statement under a
different target-family assumption. Stage 13A: **`PASS`**.

## 9. Fixed-state MMD

For a fixed RKHS feature map, let `Psi(P)=mu_P` be its mean embedding and let
`||delta||_H=MMD(P,Q)`. The reproducing dual inequality is

```text
|g_f(delta)| <= ||g_f||_{H*} ||delta||_H.
```

Thus fixed-state MMD controls a state-distance/witness norm. A target family
with `||delta_T||_H <= rho` gives
`R_T <= barR_S + rho||g_f||_{H*} + 2epsilon_repr`. This is
**`PROVED-UNDER-RESTRICTIONS`**: the state and witness class are fixed, and the
target RKHS radius is declared. MMD does not by itself identify conditional
label shifts.

Learned representations are not silently included. If `Psi_theta=L_theta Psi`
changes during training, then the state, exposure operator, target family and
dual sensitivity all change. The current theorem scope is fixed-state MMD only;
learned-map variants require a separate operator/dependency theorem.

## 10. CORAL projection theorem

In the exact supervised quadratic state
`Psi=(M,c,s)=(E[XX^T],E[XY],E[Y^2])`,

```text
Delta R = <ww^T,Delta M>_F - 2 w^T Delta c + Delta s.
```

CORAL controls the `M` projection. It does not control the full risk state:
`Delta M=0` does not imply `Delta R=0` when `Delta c` or `Delta s` changes.
The status is **`PROVED-UNDER-RESTRICTIONS`** (exact projection translation,
restricted target bridge). The explicit label-flip witness is included in the
deterministic tests.

## 11. Stage 13B gate

Fixed-state MMD has a valid dual norm bridge; CORAL has an exact supervised-
state projection and a conditional-shift counterexample. Learned-representation
MMD/CORAL is explicitly outside the fixed-state theorem. Stage 13B:
**`PASS`** for the fixed-state scope, **`REQUIRES-REPRESENTATION-EXTENSION`**
for learned maps.

## 12. Ideal IRM mechanism restriction

For `R_e(w)=w^T M_e w-2w^T c_e+s_e`,

```text
grad_w R_e(w)=2(M_e w-c_e).
```

Therefore a common stationary classifier satisfies `M_e w=c_e` for every
environment. Under a declared structural model in which this common optimum
identifies the admissible mechanism family, the compatible fiber
`E_adm([g])` shrinks. This is **`PROVED-UNDER-RESTRICTIONS`**. It is not an
identity with `g(A g)`, because it constrains supervised-state relations rather
than source risk dispersion.

## 13. IRMv1 surrogate audit

At a fixed representation and classifier coordinate `a`, the population
penalty is

```text
lambda sum_e || d/da R_e(a) |_(a=1) ||^2.
```

It is an exact derivative-statistic translation, but zero penalty means only
stationarity at the selected scale. It does not imply common global optimality
without convexity, curvature, scale, and optimization assumptions. A smooth
counterexample is `R_1(a)=(a-1)^2` and
`R_2(a)=((a-1)^2-1)^2`: both derivatives vanish at `a=1`, while the second
risk is minimized at `a=0,2`, so the fixed-scale surrogate does not establish
the ideal common-optimum condition. Status: **`RELAXATION`**; a restricted
TV/functional bridge can be added only with its own assumptions.

## 14. Fishr richer-state audit

For linear prediction, per-example parameter gradients are
`q=(w^T X-Y)X`. Their covariance contains

```text
E[(w^T X-Y)^2 XX^T]
```

which is not determined by the minimal quadratic state. To make it
algorithm-independent for all `w`, one needs at least fourth-order `X` moments,
third-order `X,Y` moments, and second-order `X,Y^2` moments (or an equivalent
gradient-feature state). Fishr therefore **`REQUIRES-RICHER-STATE`**. A tiny
witness with equal `(E[XX^T],E[XY],E[Y^2])` but different gradient covariance is
included in the deterministic tests. No larger state is silently substituted
into the Stage 12 theorem.

## 15. Cross-method theorem table

| Method | Native objective | State used | Population object | Mechanism type | Stage-12 bridge | Blind-direction status | Assumptions | Counterexample | Final status |
|---|---|---|---|---|---|---|---|---|---|
| ERM | `min bar R_e` | common affine state | source mean risk | baseline | generic `h_U(g)` only | no blind control | target family external | target outside source support | PROVED-UNDER-RESTRICTIONS |
| V-REx | `barR+lambda Var(R_e)` | same state | `g(A g)` | I | exposed ellipsoid | blind uncontrolled | exact affine risk | source span misses target | EXACT |
| GroupDRO | `min max_e R_e` | same state | `h_conv{delta_e}(g)` | II | exact convex-hull support | outside hull uncontrolled | finite observed groups | outside-hull target | EXACT |
| MM-REx | affine risk extrapolation | same state | `h_U_MM(g)` | II | exact declared affine family | beyond affine hull uncontrolled | bounded coefficient set | unbounded weights | PROVED-UNDER-RESTRICTIONS |
| fixed MMD | source RKHS discrepancy | same fixed embedding | state norm / witness IPM | III | dual-norm bound | conditional blind directions | fixed RKHS + target radius | label-channel swap | PROVED-UNDER-RESTRICTIONS |
| CORAL | covariance matching | projection of same quadratic state | `Delta M` | III | projected risk only | `c,s` blind | moment restriction | `Delta M=0, Delta c!=0` | PROVED-UNDER-RESTRICTIONS |
| ideal IRM | common population optimum | same supervised state | `M_e w=c_e` | IV | mechanism-family assumption | depends on fiber | structural identifiability | finite-environment ambiguity | PROVED-UNDER-RESTRICTIONS |
| IRMv1 | fixed-scale gradient penalty | derivative of same risk | `sum ||grad_a R_e||^2` | IV | surrogate only | not implied away | curvature/scale/optimization | stationary non-optimum | RELAXATION |
| Fishr | match gradient covariances | richer derivative state | `Cov(q_e)` | V | requires richer state | not identified by minimal state | higher moments | equal minimal state, different Cov | REQUIRES-RICHER-STATE |

## 16. Common-state integrity test

The risk-level methods ERM, V-REx, GroupDRO, and MM-REx are `SAME-STATE`.
CORAL is `FIXED-PROJECTION-OF-SAME-STATE`; fixed-state MMD is
`SAME-STATE` after declaring the RKHS embedding. Ideal IRM and restricted IRMv1
are `SAME-STATE` plus a derivative/mechanism functional. Fishr is
`REQUIRES-RICHER-STATE`. Learned representations are
`LEARNED-MAP-FROM-SAME-STATE` only after a future map theorem. Thus the current
result is not a collection of unrelated per-method states.

## 17. Cross-method consequences

1. **Risk-level support ordering.** If two declared coefficient/target sets
   satisfy `U_1 subset U_2`, then `h_U1(g)<=h_U2(g)` for every `g`. GroupDRO,
   bounded MM-REx, and any finite-mixture DRO are therefore comparable through
   their uncertainty sets, not through names of regularizers.
2. **Shared blind barrier.** Any certificate depending only on source risk
   responses factors through `V*/S°`; it cannot control target shifts outside
   `S` without a target-family or dual-fiber restriction. This simultaneously
   limits ERM, V-REx, GroupDRO, and MM-REx.
3. **Projection versus full-state control.** CORAL can make `Delta M` zero
   while V-REx/GroupDRO still see conditional risk changes through `Delta c`
   and `Delta s`; marginal MMD has the same label-information limitation.
4. **Sensitivity/support duality.** V-REx and MM-REx control dual norms of the
   centered source-risk vector, while GroupDRO takes an `l_infinity`/simplex
   support. They become the same Stage 12 object only when their declared
   uncertainty sets coincide.

These are cross-method consequences rather than a relabeling of each native
objective.

## 18. Mandatory counterexamples

The deterministic suite includes: V-REx zero with large blind target risk;
GroupDRO correct on its convex hull but wrong outside it; CORAL zero with a
label-channel change; fixed marginal MMD with a conditional change; IRMv1
stationary-but-nonoptimal risks; Fishr covariance outside the minimal state;
and the shared source-risk blind barrier.

## 19. Verification status

`experiments/stage13_regularizer_translation_tests.py` is a deterministic
NumPy audit with no data download, randomness, optimization, or neural-network
training. Lean checks the exact V-REx exposure identity, finite-group
GroupDRO/max-risk identity, a CORAL label-shift witness, the ideal-IRM
stationarity identity, and a concrete blind-direction witness in
`lean/OodTheoryVerification/Stage13/Basic.lean`. The general RKHS, support
minimality, quotient, learned-map, and Fishr higher-moment claims remain paper
proofs with executable witnesses.

## 20. Unification decision

The result is not `UNIFICATION-PASS`: Fishr needs a genuinely richer state and
IRMv1 is only a surrogate without a general target bridge. It is also not
`FAIL-AS-WRAPPER`: four risk-level methods share one exact affine state, two
have exact support translations, and the cross-method blind/projection/support
ordering consequences are nontrivial. Decision: **`PARTIAL-UNIFICATION`**.

## 21. Final scientific verdict

1. Exact: V-REx and finite-group GroupDRO; declared-family MM-REx is exact under its coefficient set.
2. Restricted: ERM, MM-REx in practical variants, fixed-state MMD, CORAL, ideal IRM.
3. Surrogate/richer-state: IRMv1 is `RELAXATION`; Fishr `REQUIRES-RICHER-STATE`.
4. Four risk-level methods share the same affine state; CORAL uses its projection.
5. V-REx controls exposed risk sensitivity; GroupDRO controls convex-hull support.
6. MM-REx controls bounded affine-hull support; MMD controls a fixed witness norm.
7. CORAL does not control conditional/label coordinates; ideal IRM constrains common optima.
8. Fishr requires fourth/third/second higher moments beyond the minimal state.
9. Shared theorem: source-risk-only certificates cannot control unrestricted blind shifts.
10. Stage 12 is genuinely reused, but no universal deep bridge is established.
11. New regularizer design and large benchmarks are not authorized by this result.
12. Final decision: **`PARTIAL-UNIFICATION`**.
