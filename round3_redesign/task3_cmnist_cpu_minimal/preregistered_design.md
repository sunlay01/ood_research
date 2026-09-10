# TASK3-CMNIST-CPU-MINIMAL Preregistered Design

git commit/HEAD before run: `9ad56c0ca05f08ae471592c97a8008bcf4346d21`

git diff --stat before run:

```text
src/ood_repr_reg/run_task3_cmnist_cpu_minimal.py | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
```

config SHA256: `a1083f49401660ecbb1f1b19e75848ad24612750b34e5255131945ebbbbbe60d`

## Exact Question

Does `LOCAL_RESPONSE` using damped detached `H^-1` carry more signal than `GRAD` using identity metric when both use the same source head-gradient disagreement?

## Exact Config Values

```json
{
  "data": {
    "image_subsample": 2,
    "label_definition": "digit < 5",
    "label_noise": 0.25,
    "normalize_pixels": true,
    "source_color_flip_probs": [
      0.2,
      0.1
    ],
    "target_color_flip_prob": 0.9
  },
  "device": "cpu",
  "irmv1": {
    "penalty_anneal_iters": 100,
    "penalty_weight": 10000.0,
    "whole_loss_rescale_after_anneal": true
  },
  "model": {
    "activation": "relu",
    "hidden_dim": 64,
    "input_dim": 392,
    "linear_layers": 3
  },
  "response": {
    "calibration_update_ratio": 0.1,
    "damping_epsilon": 0.01,
    "hessian_stop_gradient": true,
    "parameter_block": "final_linear_head_weight_and_bias",
    "solver": "cholesky",
    "use_explicit_inverse": false
  },
  "stage_a": {
    "erm_max_mean_target_accuracy": 0.35,
    "irm_min_mean_advantage_over_erm": 0.2,
    "irm_min_mean_target_accuracy": 0.5,
    "methods": [
      "ERM",
      "IRMv1"
    ],
    "seeds": [
      0,
      1,
      2
    ]
  },
  "stage_b": {
    "methods": [
      "ERM",
      "IRMv1",
      "GRAD",
      "LOCAL_RESPONSE"
    ],
    "seeds": [
      10,
      11,
      12,
      13,
      14
    ]
  },
  "task_id": "TASK3-CMNIST-CPU-MINIMAL",
  "training": {
    "batch_size_per_environment": 512,
    "checkpoint_steps": [
      0,
      100,
      200,
      300,
      400,
      500
    ],
    "l2_regularizer_weight": 0.001,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "steps": 501
  }
}
```

## Stage A Gate

Stage A runs only `ERM` and `IRMv1` on seeds `0,1,2`. It passes only if IRMv1 mean target accuracy is at least `0.50`, ERM mean target accuracy is at most `0.35`, and IRMv1 minus ERM is at least `0.20`.

## Stage B Methods

Stage B runs seeds `10,11,12,13,14` with `ERM`, `IRMv1`, `GRAD`, and `LOCAL_RESPONSE`. There is no beta grid, checkpoint selection, or target-based tuning.

## GRAD Formula

`GRAD` is `0.5 * sum_e ||g_e - g_bar||^2`, where `g_e` is the final-head weight-and-bias gradient of the source environment BCE risk.

## LOCAL_RESPONSE Formula

`LOCAL_RESPONSE` uses the same `g_e` and replaces identity with damped detached Cholesky-solved `H^-1`, where `H` is the analytic pooled-source BCE head Hessian over 65 augmented coordinates.

## Calibration Formula

For `GRAD` and `LOCAL_RESPONSE`, step-0 source batches set `c = ||G_R|| / (||G_P|| + 1e-12)`, and training uses `R + 1e-3 W + 0.10 c P`.

## Verdict Rules

Allowed verdicts are `CPU-MINIMAL-INVALID`, `CPU-MINIMAL-SIGNAL`, and `CPU-MINIMAL-NO-SIGNAL`. Signal requires the preregistered LR-vs-ERM, LR-vs-GRAD, seed-win, source-accuracy, and mechanism-direction gates.

## Forbidden Target Uses

Target metrics cannot affect training, calibration, checkpointing, method choice, or any threshold. Target is evaluation-only.

## Interpretation Ceiling

This is an exploratory CPU signal probe only. It does not establish theory support, causal recovery, SOTA, finite-sample guarantees, or paper-level readiness.
