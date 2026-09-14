# Stage 13R.1: Fixed Physical Tangent Re-audit

## Frozen family

Before looking at any native objective, fix the finite-dimensional physical
tangent family

```text
D_phys(xi) = {delta in V : ||delta||_2 <= 1}.
```

This is a declared Euclidean unit ball on the upstream supervised environment
state. It is not chosen to reproduce V-REx, GroupDRO, MMD, or IRM. The common
quantity is therefore always

```text
S_phys(f;xi) = sup_{||delta||_2 <= 1} D_xi R(f,xi)[delta]
             = ||D_xi R(f,xi)||_2
```

in the finite-dimensional smooth case.

## Method comparison

| Method | Fixed-`D_phys` relation | Status |
|---|---|---|
| V-REx | `Var_e R_e = g(A g)` controls only source-exposed directions. It upper-bounds `S_phys` only under an additional domination/coverage condition `I <= c A` on the whole state; otherwise blind directions make the ratio infinite. | `UPPER-BOUND` under coverage, otherwise `NO-CONTROL` |
| GroupDRO | `max_e R_e` is support over the finite observed convex hull. It controls `S_phys` only if the unit ball is contained in a declared displacement hull with a Lipschitz conversion; the frozen unit ball is not generally that hull. | `RELAXATION` / `NO-CONTROL` outside hull |
| fixed-state MMD | For the declared Euclidean/RKHS witness norm, `|D_xiR[delta]| <= ||D_xiR||_*||delta||` gives a generic dual upper bound on the same `S_phys`; conditional shifts require the state to include them. | `UPPER-BOUND` |
| ideal IRM | Common stationarity constrains supervised mechanism equations and can shrink an admissible path family, but it does not equal or directly bound `||D_xiR||_2` without extra structural and mixed-derivative assumptions. | `SURROGATE` |

The left-hand side remains the same for every row. No row is allowed to change
`D_phys` to make its native penalty exact. At least V-REx and GroupDRO require
additional coverage geometry; ideal IRM requires a mechanism theorem; MMD is a
generic norm inequality. Consequently there is no new fixed-tangent
cross-method theorem supporting an independent ESF master.

## Referee attacks

- The unit ball is external and fixed before algorithm analysis.
- Source groups do not automatically cover all unit-ball directions.
- `I <= cA` is a target-state coverage assumption, not a property of V-REx.
- MMD's dual bound is valid only for a fixed witness/state and does not imply
  conditional or label-shift control.
- Ideal IRM is not identified with an environment derivative; its bridge is
  structural and conditional.

**Gate result:** no fixed-`D_phys` independent unification survives.
