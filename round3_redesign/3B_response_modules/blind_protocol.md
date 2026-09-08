# Blind Discovery Protocol

Discovery receives only `H_S`, `g_s`, the whitened response columns, and the
intrinsic Gram matrix. It does not receive mechanism labels, intervention
families, exposure, regularizer identity, target-risk labels, or oracle module
count.

The pipeline filters first-order-null columns, normalizes response columns,
and compares point-clustering, local-PCA subspace reports, and Gram-based
spectral proxy clustering. Candidate counts from 1 through 6 are selected by
silhouette/stability diagnostics with the smaller count used on ties.

Labels are revealed only after assignments are frozen. Post-hoc metrics include
ARI, NMI, enrichment, principal angles and mixed-shift reconstruction. A
stable response module is not named as a causal mechanism.
