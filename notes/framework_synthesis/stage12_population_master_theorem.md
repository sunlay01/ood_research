# Stage 12: Population Theorem Package

## Decision and scope

This stage is finite-dimensional. Let `V` be a finite-dimensional real vector
space, `V*` its dual, and write

```text
R_P(f) = b_f + g_f(Psi(P)) + eta_f(P).
```

The package contains an exact transfer identity, sharp support minimality, a
source-observable quotient barrier, blind-direction impossibility, and a
positive exposure-seminorm corollary under explicit coverage. Decision:
**`ADVANCE-TO-STAGE-12.5`** for the exact novelty audit. No algorithm mapping
or new regularizer is authorized here.

## 1. Exact transfer identity

Let `bar Psi_S=(1/m)sum_e Psi(P_e)`, `delta_T=Psi(P_T)-bar Psi_S`, and
`bar eta_S=(1/m)sum_e eta_f(P_e)`. Direct subtraction gives

```text
R_T(f) - bar R_S(f) = g_f(delta_T) + eta_f(P_T) - bar eta_S(f).
```

If `|eta_f(P)| <= epsilon_repr(f)` for every source and target distribution,
then `R_T(f) <= bar R_S(f) + g_f(delta_T) + 2 epsilon_repr(f)`. For an
externally declared family `U_phys` with `delta_T in U_phys`,

```text
R_T(f) <= bar R_S(f) + h_Uphys(g_f) + 2 epsilon_repr(f),
h_U(g) = sup_{delta in U} g(delta).
```

The residual is supplied by the representation model; it is not hidden in
the support term.

## 2. Sharpness of support

Fix `g` and a nonempty target family `U`. If a uniform additive certificate
`B(g,U)` satisfies `g(delta) <= B(g,U)` for every `delta in U`, then by the
definition of a supremum

```text
B(g,U) >= sup_{delta in U} g(delta) = h_U(g).
```

Thus `h_U(g)` is the smallest certificate using exactly the information
`delta_T in U`. A later split `rho*S+kappa*N`, IPM estimate, or DRO support
bound must be identified as an exact specialization or a relaxation under
additional information.

## 3. Exposure operator and observable quotient

For source shifts `delta_e`, let `S=span{delta_e}` and define

```text
A : V* -> V,       A g = (1/m) sum_e g(delta_e) delta_e.
```

The intrinsic energy identity is
`g(A g)=(1/m)sum_e g(delta_e)^2`. Finite-dimensional linear algebra gives

```text
ker A = S° = {h in V* : h(delta)=0 for every delta in S}.
```

Source directional responses determine only `g|_S`, equivalently the class
`[g]` in `V*/S°`. If `h in S°`, replacing `g` by `g+h` leaves every source
response unchanged.

## 4. Blind ambiguity theorem

Let `H_adm` be the allowed dual extensions consistent with the declared model.
For a representative `g`, define its compatible fiber

```text
E_adm([g]) = {q in H_adm : q-g in S°},
C_U([g]) = sup_{q in E_adm([g])} h_U(q).
```

Any population certificate using only source directional responses is at least
`C_U([g])`: all members of this fiber have identical source observations, so
one certificate must cover their target supports. Finiteness requires a
declared restriction on the admissible blind extensions.

## 5. Blind-direction impossibility

Assume a target direction `delta_perp` is outside `S` and no norm or structural
bound is imposed on `h in S°`. There is `h0 in S°` with
`h0(delta_perp) != 0`. For `h_t=t*h0`, and a target family containing
`delta_perp`, choose the sign of `h0` to obtain

```text
h_U(g+h_t) >= (g+h_t)(delta_perp) -> +infinity.
```

Therefore source-only responses cannot yield a finite target-risk certificate
for unrestricted blind target directions. This is non-identifiability, not
merely a loose `kappa` term.

## 6. Exposure-seminorm domination

The source seminorm is `||g||_A=sqrt(g(A g))`. If `U_rho` is an exposed-span
ellipsoid

```text
U_rho = {delta in S : delta^T A_S^dagger delta <= rho^2},
```

with `A_S` positive definite on `S`, Cauchy--Schwarz gives

```text
h_Urho(g) <= rho ||g||_A.
```

For nonzero `g|_S`, an aligned ellipsoid boundary point attains equality.
Thus `rho*S_A` is a corollary of explicit target geometry, not the primitive
certificate.

Conversely, if `h_U(g) <= C||g||_A` for every `g in V*` with finite `C`, then
`U` cannot contain a shift outside `S`: a functional in `S°` has zero
`A`-seminorm but nonzero value on that shift. Hence `U subset S` is necessary
for unrestricted dual sensitivities. Coupled target families can make a split
`rho*S+kappa*N` strictly looser than exact support.

## 7. Proof status and audit

The exact transfer identity and concrete finite-dimensional witnesses are
Lean-checked in `lean/OodTheoryVerification/Stage12/Basic.lean`. General
supremum minimality, quotient lower bounds, annihilator existence, and the
ellipsoid theorem are paper proofs with deterministic checks in
`experiments/stage12_population_theorem_tests.py`. No stochastic training,
benchmark, neural network, algorithm mapping, or novelty claim is included.

## 8. Gate

The package has both required sides: a positive exposed-span domination theorem
and a blind-direction impossibility theorem. Stage 12.5 exact novelty audit is
the next authorized action. This document does not claim that any existing
regularizer instantiates the package.
