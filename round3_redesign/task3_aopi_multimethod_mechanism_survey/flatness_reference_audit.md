# Flatness reference audit

This audit distinguishes definition-faithful common-harness variants from paper benchmark reproduction.

- `SAM`: implemented as the standard two-step first-order update on the same source minibatch.
- `ASAM`: implemented as adaptive SAM with elementwise parameter-scale perturbation and fixed `eta`.
- `FAD`: deferred; exact zeroth/first-order DG flatness update was not implemented rather than replaced with a SAM surrogate.
- `DISAM`: deferred; exact domain-imbalance perturbation calibration was not implemented rather than replaced with `SAM + VREX`.

Raw parameter-space flatness is not invariant under arbitrary reparameterization; comparisons here are controlled diagnostics under fixed architecture, parameterization, initialization, optimizer family, and source schedule.
