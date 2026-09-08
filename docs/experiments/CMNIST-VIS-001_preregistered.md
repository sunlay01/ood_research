# CMNIST-VIS-001: Counterfactual Feature-Response Probe

Status: `PROBE_AUTHORIZED / EXPLORATORY`

This experiment tests whether paired, task-preserving color interventions can
make regularization-induced representation changes directly observable. It is
not a novelty claim, benchmark comparison, or continuation of `LATENT-001`.

## Question

For a fixed grayscale MNIST image and binary shape task, does adding a source
regularizer change:

1. the latent displacement caused only by swapping red and green;
2. the amount of that displacement used by the prediction head;
3. balanced-color task information; and
4. error under a frozen color-correlation sign flip?

## Data

The binary task is `Y = 1[digit >= 5]`. A color bit equals the task label with
probability `p_e`. The two source environments use `p=(0.9, 0.8)` and the frozen
target uses `p=0.1`. Labels and grayscale pixels are unchanged by colorization.

Every feature probe contains exact counterfactual pairs made from the same
grayscale image:

```text
x_red(g) <-> x_green(g).
```

No target quantity enters training. All method/strength paths are reported;
target performance is not used to select a strength.

## Methods

- ERM;
- joint parameter L2;
- scalar-scale IRMv1 with cross-entropy risk;
- CORAL on penultimate-layer covariance.

All methods use the same CNN, source samples, initialization seed, optimizer,
batch schedule, and epoch count for a given seed. The representation is the
penultimate activation `z`; the head is a linear two-class classifier.

## Measurements

Let `z_i^r,z_i^g` be the paired representations and let `v=W_1-W_0` be the
binary head direction. Report:

```text
latent color response     E ||Whiten(z_i^r-z_i^g)||^2
prediction color response E [(z_i^r-z_i^g)^T v]^2
probability response      E ||softmax(h_i^r)-softmax(h_i^g)||^2
task signal               whitened distance between balanced class centroids
task/color overlap        fraction of color-response energy on task direction
balanced accuracy         average accuracy over both colors for each image
source/target accuracy    observational source and frozen sign-flip target
```

Whitening is fitted on the balanced counterfactual probe only and is used for
descriptive latent geometry. Prediction response and risks are invariant to an
invertible latent reparameterization when the head is transformed with it.

The two-dimensional feature-path view uses source-probe ridge coordinates:
one coordinate predicts the task label and the other predicts the color bit.
It is a supervised display, not a claim that the representation is intrinsically
two-dimensional or orthogonally decomposed.

## Interpretation Rules

- `feature removal`: latent and prediction color responses both decrease while
  balanced accuracy is retained;
- `head rejection`: prediction response decreases but latent response remains;
- `nonselective collapse`: task signal and balanced accuracy decrease with the
  color response;
- `uncontrolled color`: prediction response and sign-flip error remain high;
- `descriptive only`: a 2-D trajectory changes without matching quantitative
  response metrics.

No causal or generalization-control claim follows from correlation alone.

## Probe Success And Stop Rules

The probe succeeds as an observation instrument if all paired metrics are
finite, counterfactual identity checks pass, repeated seeds preserve qualitative
ERM behavior, and the figures distinguish latent retention from head use.
It fails if color rendering changes labels/grayscale support, methods receive
different source data, response metrics are coordinate artifacts, or plots
cannot be reconciled with saved numeric values.

## Outputs

- `metrics.csv` with every seed/method/strength;
- all-method source-only diagnostic penalties and model checkpoints;
- `counterfactual_paths.png`;
- `feature_response_summary.png`;
- `counterfactual_examples.png`;
- `config.json`, `environment.json`, and a result report.
