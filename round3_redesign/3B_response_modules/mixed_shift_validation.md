# Mixed-Shift Validation

For moment-additive probes, the quadratic population risk gradient is linear
in the moment delta.  Therefore \(g_{s+t}=g_s+g_t\) and the whitened response
is additive.  The runner checks this identity on held-out compositions
`S1+S2`, `S1+U`, `S1+N`, and `S1+S2+N`.

The module reconstruction is a diagnostic.  A small residual establishes that
the observed response lies in the joint discovered span; it does not establish
that the decomposition is unique or semantic.  The report therefore records
joint rank and a `unique_if_direct_sum` flag.
