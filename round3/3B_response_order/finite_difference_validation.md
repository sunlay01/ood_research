# Finite-Difference Validation

The audit evaluates `h` in `{0.025, 0.05, 0.1, 0.2, 0.4}` and both signs. It uses central
first and second derivatives and the symmetric third derivative stencil

`[R(2h)-2R(h)+2R(-h)-R(-2h)]/(2h^3)`.

The JSON records relative changes between adjacent step sizes and analytic-vs-finite-difference
errors. An analytic zero derivative is reported with an absolute numerical residue, rather than
an unstable relative error.

