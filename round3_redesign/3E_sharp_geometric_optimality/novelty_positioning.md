# 3E-C novelty positioning

## Scope

3E-C is a finite-dimensional population result about an affine response
policy. Its input is the pair of linear maps `(A, O_S)` from the coupled
3A/3D tangent, together with a static offset `z0` and a source-adaptive
operator `Pi`. The main object is the exact geometry of a policy relative to
the source-information minimax floor.

The contribution claimed here is the operational sharp comparison of four
objects:

1. the irreducible response `A_irr = A P_ker(O_S)`;
2. the recoverable residual `E = A_rec + Pi O_S`;
3. the static steering offset `z0`; and
4. the spectral slack left by the worst invisible response direction.

The benchmark is generated from legal population environments and regularizer
objectives. The benchmark does not attach semantic meaning to a singular
vector, use a mechanism label, or select a method with target-risk information.

## Relation to earlier tracks

3A supplies the source-whitened response metric and the vulnerability
interpretation. 3D supplies the task-complete source observation geometry and
the identifiable/hidden-emergent boundary. 3B response modules and 3C
regularizer geometry are used only as frozen boundary context; neither enters
the sharp theorem as an assumption. 3E-C therefore does not claim causal
identification, latent-factor recovery, universal domain generalization, or a
target-risk lower bound.

## What is and is not new

The projection decomposition and finite-dimensional trust-region facts are
standard mathematical machinery. The specific result audited here is their
coupled use for concrete source-derived regularization policies, including
the necessary-and-sufficient spectral-slack condition and the separately
audited legal helps/hurts worlds. The spectral slack is metric-conditional:
it is invariant under response-space isometries and depends on the declared
standardized world metric. It is not a coordinate-free claim under arbitrary
world reparameterizations.

## Negative claims

The results do not establish that a regularizer is universally helpful or
harmful. They do not turn source exposure into relevance, and they do not
provide finite-sample, deep-network, causal, semantic, or target-risk
guarantees. A failed helps search would be a statement about the registered
population family only. In the current registered family a legal helps world
was found, and the report records its exact search protocol and post-hoc
diagnostics.
