# CMNIST-VIS-001 Report

Status: `PARTIAL_SIGNAL / EXPLORATORY`

Independent code audit: `AUDIT_PASS`.

Independent result validation: `PARTIAL_SIGNAL`.

## Question

Can paired task-preserving color counterfactuals make it possible to observe
what a regularizer changes in the learned feature space, while separating
latent color retention from color actually used by the prediction head?

## Run

The run used the binary CMNIST task `Y=1[digit>=5]`, source color-label
correlations `(0.9, 0.8)`, and a target sign flip with correlation `0.1`.
There were 3 seeds, 7 method/strength paths per seed, and therefore 21 models:
ERM, L2 at `0.0001/0.001`, IRMv1 at `1/10`, and CORAL at `0.1/1`.
The target was evaluated only after training. No strength was selected using
target results.

Artifacts:

- `artifacts/CMNIST-VIS-001-REV1/metrics.csv`
- `artifacts/CMNIST-VIS-001-REV1/paired_metrics.csv`
- `artifacts/CMNIST-VIS-001-REV1/aggregate_summary.csv`
- `artifacts/CMNIST-VIS-001-REV1/paired_effects.png`
- `artifacts/CMNIST-VIS-001-REV1/counterfactual_paths.png`
- `artifacts/CMNIST-VIS-001-REV1/counterfactual_examples.png`
- `artifacts/CMNIST-VIS-001-REV1/feature_atlas/`

The REV1 rerun added fixed source-only diagnostic penalties and checkpoints.
All shared numeric metrics matched the first run exactly.

## What The Plots Show

The paired path plot is a supervised display: each line connects the red and
green representation of the same grayscale digit. Its axes are ridge probe
coordinates for task and color; they are not intrinsic latent coordinates.
The numeric quantities that support interpretation are the whitened latent
color response and the head-used logit response.

## Direct Penultimate-Layer Visualization

The feature atlas directly evaluates the 32-dimensional vector returned by
`model.encode(x)`, which is the penultimate representation for this model. For
each neuron it shows the eight fixed-probe inputs with the largest activation,
and a separate panel overlays the absolute input gradient of that neuron. The
CSV records activation mean/std, red/green means, color selectivity, and task
class means for all neurons. The images are therefore evidence about feature
preferences, not a semantic label assigned by PCA or by visual inspection.

In the seed-0 atlas, ERM, CORAL(1), and IRMv1(1) have top units that repeatedly
respond to recognizable digit shapes, especially green 8/5/7 and red 3/2.
Their gradients are concentrated on digit strokes rather than only the black
background. CORAL(1) has essentially the same top-shape inventory as ERM, so
the earlier reduction in head-used color response is not evidence that CORAL
removed color information from the latent vector. It is consistent with a
change in how the head uses the representation.

IRMv1(10) is visibly different: its top units are dominated by red 3/2/0/9,
and the gradient overlays follow those strokes. Only 20 of 32 units are
nonzero on the probe for every displayed checkpoint, while the average color
selectivity among active units is approximately 0.98 for ERM/CORAL(1), 1.04
for IRMv1(1), and 1.80 for IRMv1(10). This is direct visual support for the
previously observed task-collapse/color-entanglement failure at high IRMv1
strength, not proof that individual units have intrinsic causal meanings.

Activation maximization was also attempted with fixed red/green channels. The
unconstrained version produced high-frequency adversarial textures. A
low-resolution and data-anchored version reduced this effect but still did not
consistently produce clean digits. This is an expected identifiability limit:
a scalar post-flattening neuron has no unique image inverse. The synthesized
images are therefore retained as a diagnostic of the network's optimization
preferences, not as semantic evidence. The real top-activation montage is the
primary visualization. Spatial feature visualization should use the preceding
convolutional feature map, which still has a 7x7 spatial grid.

## Paired Results Relative To Same-Seed ERM

| method | own source penalty / ERM | target accuracy change | balanced accuracy change | latent color response / ERM | head-used color response / ERM | task signal / ERM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CORAL 0.1 | 0.920 | -0.13 pp | -0.03 pp | 1.019 | 0.994 | 1.004 |
| CORAL 1 | 0.609 | +0.33 pp | +0.27 pp | 1.069 | 0.918 | 1.027 |
| IRMv1 1 | 0.846 | -3.57 pp | -1.40 pp | 0.978 | 1.089 | 0.966 |
| IRMv1 10 | 1.733 | -40.27 pp | -21.93 pp | 1.691 | 1.005 | 0.659 |
| L2 0.0001 | 0.992 | +0.13 pp | +0.07 pp | 0.996 | 0.994 | 1.000 |
| L2 0.001 | 0.956 | -1.33 pp | -0.30 pp | 1.102 | 1.044 | 0.979 |

The source diagnostic is recomputed for every candidate penalty on the same
fixed source-only subset after training. It is distinct from the last training
minibatch penalty and is the quantity used in this table.

## Mechanism Reading

### CORAL

At strength `1`, CORAL reduces its source covariance discrepancy to about 61%
of the paired ERM diagnostic. The latent color response does not decrease; it
increases to about 107% of ERM. The head-used color response decreases to about
92% while task signal is retained. The narrow supported interpretation is:

```text
CORAL: head rejection / covariance-mediated prediction change
not feature removal.
```

The target improvement is only about `0.33` percentage points and is not enough
to claim OOD control. CORAL 0.1 gives no meaningful effect.

### IRMv1

At strength `1`, the source IRMv1 diagnostic decreases on average, but the
head-used color response increases to 109% of ERM and sign-flip accuracy falls
by 3.57 percentage points. Thus a lower IRMv1 penalty does not imply lower
color sensitivity in this probe.

At strength `10`, task signal falls to 66% of ERM, balanced accuracy falls by
21.93 percentage points, and target accuracy falls by 40.27 percentage points.
The latent color response rises to 169%. This is a task-signal/representation
collapse or optimization-dominated failure, not successful invariant feature
removal.

### L2

The small L2 setting is nearly indistinguishable from ERM. The larger setting
slightly reduces its parameter penalty but increases latent and head-used color
responses on average and slightly reduces task signal. It does not show
selective OOD feature control here.

## Current Conclusion

This experiment validates the observation strategy, not a general theorem.
The useful distinction is:

```text
latent feature response != prediction-head use != target error.
```

In this CMNIST probe, CORAL provides the cleanest example of reducing
prediction-use of a color response while leaving color information in the
latent representation. IRMv1 provides the contrasting failure: penalty
reduction can coexist with increased head-used color response, and strong
penalty can destroy task signal. L2 is mostly nonselective at these settings.

The results do not establish a universal semantic orthogonal decomposition or
an Omega-only generalization bound. They support a revised research object:
measure paired intervention responses in feature space, measure which of those
responses the head uses, and connect the two to source-fit and target transport
error through an explicit accounting identity.

## Limitations

- one binary MNIST task and one target color sign flip;
- three seeds and a small method/strength grid;
- no mean/covariance image interventions yet;
- the 2-D path coordinates are supervised displays, not intrinsic axes;
- the full repository test suite retains one unrelated existing numerical-root
  failure in `scalar_irmv1_zero_candidates`; all seven CMNIST-specific tests
  pass.
