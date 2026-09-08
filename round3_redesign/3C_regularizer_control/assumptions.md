# Assumptions

This track freezes the 3A/3A-T source-risk geometry and the 3B individual response
vectors.  The main calculation is finite-dimensional population quadratic
regression with squared loss.  Source environments are the 3B relation-exposed
population design; target probes are evaluated only after the regularizer-side
objects are formed.

The primary response space has 33 relevant probes and numerical rank 5.  The
3B six-cluster output is not used as an input and is not interpreted as a
semantic module.  Results are population diagnostics, not finite-sample or
deep-network guarantees.
