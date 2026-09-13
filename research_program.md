# Algorithm research program

This branch studies **local, operational training mechanisms**. Its question is:

> What information available during training can help choose better updates, reject harmful directions, or select a safer continuation?

The primary objects are network states, gradients, optimizer and algorithm state, perturbation responses, trajectories, environment-gradient relations, and short-horizon counterfactual continuations. A representation is useful here only when it supports an algorithmic decision.

The required operational chain is:

```text
probe -> observable -> estimator -> decision -> held-out intervention
```

For example, a semantic `+/-` perturbation produces functional logit displacement, which gives a finite-difference response estimate, which may score or suppress a candidate update. Abstract objects and their observable estimates must be recorded separately, together with probe coverage, conditioning, stochastic error, schedule dependence, architecture dependence, and method-identity dependence.

## A/O/Pi status

A/O/Pi remains a candidate mechanistic representation for local algorithm behavior:

\[
u \xrightarrow{O_S} O_Su \xrightarrow{\Pi_j} \Pi_jO_Su,
\]

with `A` retained as a task-relevance or oracle comparison coordinate when the experiment defines it. It is not the default theory of OOD, a proven causal mechanism, a source-only predictor of target risk, or a generalization theorem.

The current controlled evidence is deliberately narrow. Response geometry varies across algorithms, full `A` geometry can predict some finite semantic behavior, and method identity can explain substantial variation. After full `A/O` controls and independent continuation schedules, no stable incremental `Pi O_S` mechanism has been established. The correct status is therefore:

```text
A/O/Pi: mechanistic candidate representation
not a general OOD theory
not a proven causal mechanism
not a proven source-only predictor of OOD generalization
```

## Evidence standard

Every positive claim must use, where applicable:

- the same complete checkpoint state (model, optimizer, algorithm state, and logical step);
- matched forks with a single controlled intervention;
- independent continuation schedules when a predictor and outcome could share trajectory noise;
- leave-one-seed-out and leave-one-method-out checks;
- method-identity baselines and mechanism-mismatched negative controls;
- held-out intervention directions or doses;
- provenance hashes and reproducible source-only data construction.

A descriptive geometry is not called a mechanism without an intervention. A predictive signal is not called a theory. The interpretation ceiling for this branch is algorithmic/actionable evidence, not a statistical DG guarantee.

## Work plan

1. Register the algorithmic decision before defining a representation.
2. Build the smallest source-only probe that measures the proposed state variable.
3. Validate estimator fidelity and state matching.
4. Test transfer across seeds, methods, schedules, and architectures.
5. Intervene on the proposed decision and record both gains and counterexamples.
6. Retain negative results as hypothesis-space reductions.

A future rigorous bridge to statistical theory would be a new result and must not be assumed by this branch.
