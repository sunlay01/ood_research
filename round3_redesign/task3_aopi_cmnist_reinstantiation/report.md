# TASK-AOPI-CMNIST-REINSTANTIATION-AUDIT Report

## Provenance And Validity

- git head: `a973719211fd85517a7cb978dcd0d7ea5826be04`
- methods: ERM, IRMv1; seeds: 10..14
- parameter block: frozen nonlinear encoder, trainable 65D final head
- checkpoint target-gap integrity: `True`

## Why This Audit Was Run

The formal/synthetic A/O/Pi construction predates corrected neural CMNIST evidence. This audit tests a restricted frozen-encoder final-head instantiation; it does not replace the frozen 3A--3D theory.

## A Geometry

G1 stable geometry: `True`.

## O_S Geometry

O_S is the source-only concatenation of per-environment risk and IRMv1 penalty gradients. Its ranks and kernels are in `results/geometry_O.csv`.

## Pi Finite Validation

G2 finite source-perturbation prediction: `False`.

## ERM Vs IRMv1 Discrimination

G3: `NONDISCRIMINATIVE`. All paired seeds are in `results/paired_method_summary.csv`.

## Gate Verdict

`AOPI-REINSTANTIATION-PARTIAL`

## Interpretation Boundary

This result concerns a frozen-encoder, final-head local response only. It does not establish semantic mechanism discovery, causal recovery, finite-sample theory, universal DG validity, a new algorithm, or novelty.
