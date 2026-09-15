# Phase-0 Preregistration

## Question

Can a finite-dimensional, response-space intervention class support a coherent
decomposition into source information, class reachability, and actual-response
realization without coordinate or spectral-slack contradictions?

## Quantity

For fixed `A : U -> Z`, `O_S : U -> Y`, and `Pi : Y -> Z`:

```text
R_adap(Pi) = 1/2 ||A + Pi O_S||_op^2
R_info = 1/2 ||A P_ker(O_S)||_op^2
R_C* = inf_{Pi in C} R_adap(Pi)
Delta_control(C) = R_C* - R_info
```

`z_0` and the full affine regret are excluded from this first gate.

## Pre-registered checks

1. Head-only/full-class toy: `R_info=0`, `R_head*=1/2`, `R_full*=0`, and an
   actual `Pi=[-1/2,-1/2]` has `R_actual=1/4`.
2. Zero-observation toy: every class remains at `R_info=1`.
3. Spectral-slack toy: nonzero `E` may still attain `R_info`.
4. One hundred random nested classes satisfy `R_C2* <= R_C1*`.
5. Source-coordinate and orthogonal response-coordinate transport preserve the
   class optimum.
6. A dense grid agrees with the tiny-class numerical optimizer within the
   registered tolerance.

## Stop rules

Failure of any check stops the route before panel reuse or neural intervention.
Passing these checks does not authorize claims about actual algorithms or target
OOD risk; those require the later membership, locality, anti-leakage, and risk
relevance gates.
