# Theory research program

## Goal

Develop mathematically tractable statistical representations and bounds for out-of-distribution learning and domain generalization.

## Primary objects

The default level is distributions, functions, hypothesis classes, risks, source-environment structure, and explicitly stated shift assumptions:

\[
P_e,\quad P_S,\quad P_T,\quad \mathcal F,\quad f,\quad \ell,\quad R_e(f),\quad \widehat R_e(f).
\]

Candidate tools include change of measure, IPMs and dual discrepancies, conditional decompositions, source-environment variance, coupling or transport, minimax robust risk, PAC-Bayes, function-level stability, concentration, covering, and complexity bounds. No particular divergence or decomposition is accepted in advance.

## Current status

This branch has **no accepted core representation system**. In particular, A/O/Pi is not imported as a default theory, and no microscopic gradient or response object is assumed to control target risk.

The branch does not assume a bridge of the form

```text
gradient geometry -> optimization path -> learned representation
             -> distribution shift -> OOD risk
```

Such a bridge would be a separate theorem-level research result requiring its own assumptions and evidence.

## Research workflow

### T1: Fix the quantity of interest

Write the target quantity before choosing a representation. Examples include

\[
R_T(f)-R_S(f),
\qquad
R_T(f)-R_T(f_T^*),
\qquad
\sup_{P\in\mathcal P_{\mathrm{adm}}} R_P(f).
\]

"OOD theory" is not a sufficient target.

### T2: Fix the information structure

State what is observable and when. In source-only DG, source environments are available and the target is unavailable at training time. Target-dependent oracle terms may appear in a theorem only when labeled as oracle terms. Separate source-estimable, assumption-controlled, and irreducible quantities.

### T3: Derive a same-level decomposition

Seek a risk-level or distribution-level identity before applying inequalities. Possible directions include risk decomposition, change of measure, conditional shift, environment heterogeneity, coupling, robust-risk duality, estimator/function stability, and impossibility results. The goal is a coherent system of objects whose interaction controls the declared target quantity.

### T4: Preserve exact structure before bounding

Prefer an identity or near-identity

\[
Q=T_1+T_2+\cdots+\mathcal R
\]

before triangle inequalities or coarse norm bounds. Mark structural, approximation, estimation, shift-uncertainty, irreducible, and proof-slack terms separately.

### T5: Judge theorem quality

A representation is useful only if it supports results that are valid, tractable, reasonably tight, statistically interpretable, compatible with DG information constraints, and explicit about assumptions. The operating rule is:

\[
\boxed{\text{Theory: tractability before microscopic fidelity.}}
\]

## Initial research question

What statistical representation of source environments and function classes is most suitable for a source-only DG generalization bound?

This question should be narrowed to one quantity and one information structure before proving anything. Candidate work should include positive bounds, tightness analyses, counterexamples, and impossibility results. A non-vacuous source-only bound may require defining an admissible target family from source environments; whether that is possible is itself a theorem question.

## Evidence standard

Primary evidence is theorem validity, counterexamples, impossibility results, assumption audits, tightness comparisons, and compatibility with source-only information. Numerical experiments can illustrate a theorem but do not substitute for a proof. No theorem claim may silently use target data, optimizer state, target-tuned representation choices, or an unproved microscopic bridge.
