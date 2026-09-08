# 3B Results

`summary.json` is the machine-readable aggregate. `per_shift.csv` includes
both retained and first-order-null probes; an empty `blind_assignment` means
that the probe was excluded from blind discovery. `module_metrics.csv` reports
SVD internal ranks. `mixed_reconstruction.csv` reports held-out response
reconstruction and direct-sum uniqueness diagnostics. `stability.json` stores
the rotation, nuisance and bootstrap audits.

Oracle mechanism fields are post-hoc audit fields only. They are not inputs to
module selection.
