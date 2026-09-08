# 3E-C Sharp Geometric Optimality and Helps/Hurts Benchmark

## Verdict

`3E-C-SHARP-GEOMETRY-PASS`

The finite-dimensional population theorem is evaluated for the affine policy
`z_j(u)=z_j^0+(A_irr+E_j)u`, with `P=P_ker(O_S)`, `A_irr=AP`, and
`E_j=A_rec+Pi_j O_S`.

## Sharp theorem

Because `P O_S^*=0`, `E_j P=0`, and `A_irr Q=0`, the cross operators vanish:
`A_irr E_j^*=E_j A_irr^*=0`. Thus
`(A_irr+E_j)(A_irr+E_j)^*=A_irr A_irr^*+E_j E_j^*`.

With `alpha=||A_irr||` and
`S_slack=alpha^2 I-A_irr A_irr^*`, the adaptive-only condition is exactly

`R_adap = R_info  iff  E_j E_j^* <= S_slack`.

The full affine condition is exactly

`R_j = R_info  iff  z_j^0=0 and E_j E_j^* <= S_slack`.

The proof uses finite-dimensional compactness for the top invisible singular
direction. It does not assume `alpha>0`; when `alpha=0`, the slack is zero and
the condition reduces to `E_j=0` (together with zero static steering).

## Hidden-U benchmark

The hidden-U coupled 3A/3D world has 18 table rows. ERM regret is
`0.05118145109`. All valid regularized rows satisfy the
adaptive spectral condition and adaptive regret equals the information floor;
positive lambda can still increase full regret through the static steering tax.

The tax is the symmetric-ball lower bound
`R_j >= R_info + 1/2 ||z_j^0||^2`; it is a lower bound, not a claimed equality.

## U-exposed complete information

The separately reported U-exposed table has 18 rows. Here the
information floor is zero. Nonzero `E_j` is therefore no longer absorbed by
spectral slack and produces adaptive regret. This table is a boundary audit,
not a change to the hidden-U primary benchmark.

## Helps and hurts

The hurts world is the hidden-U coupled benchmark. The legal source-design
search returned: **found**. It evaluates only worlds generated from
`ModuleEnvironment`, source environments, source task states, and the exact
population regularizer objective; it never edits `A`, `O_S`, or `Pi` directly.

The selected helps record, when present, has `source_u_exposed=true` and uses
an asymmetric source composition. Its ERM recoverable residual is nonzero, so
the improvement is a within-world regularizer comparison rather than a
comparison of different estimators or target-risk oracle selection.

## Boundaries

`E_j=0` is sufficient but not necessary for adaptive minimax optimality. Under
complete information it becomes necessary. The slack ratio is a metric-
conditional diagnostic and is infinite when support compatibility fails.
The `(b,K)` frozen quadratic pair does not determine `Pi` or the exact affine
action. No semantic labels, clusters, target-risk lower bound, causal claim,
finite-sample guarantee, deep-network result, or universal DG theorem is used.

Lean remains `LEAN-PARTIAL`: the compact algebraic theory is separate and has
no placeholders, while operator norms, top eigenvalues, and full PSD-order
formalization remain outside the retained Lean core.
