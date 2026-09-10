# TASK3-CMNIST-CPU-MINIMAL Report

## 1. Exact question
Does damped inverse-Hessian weighting improve the same head-gradient disagreement signal relative to identity weighting in this fixed CPU ColoredMNIST probe?

## 2. Exact CPU protocol
CPU-only, 3 linear layers `392->64->64->1`, batch size 512 per source environment, Adam, 501 steps, no beta grid, no checkpoint selection, target evaluation only.

## 3. Stage A benchmark calibration
ERM mean target accuracy: `0.10726666698853175`
IRMv1 mean target accuracy: `0.6746666431427002`
IRMv1 minus ERM: `0.5673999761541685`
Stage A passed: `True`

## 4. Stage B results
{
  "ERM": {
    "source_mean": 0.8497520089149475,
    "source_std": 0.0003981458099071671,
    "target_mean": 0.10979999899864197,
    "target_std": 0.00485540983389911
  },
  "GRAD": {
    "source_mean": 0.8499080061912536,
    "source_std": 0.0005715995787778023,
    "target_mean": 0.12025999873876572,
    "target_std": 0.005827776669594319
  },
  "IRMv1": {
    "source_mean": 0.5995999991893768,
    "source_std": 0.019630488160651802,
    "target_mean": 0.6691400051116944,
    "target_std": 0.013166369985302455
  },
  "LOCAL_RESPONSE": {
    "source_mean": 0.8498359978199005,
    "source_std": 0.00042811246240158574,
    "target_mean": 0.11126000136137008,
    "target_std": 0.0063775401752472795
  }
}

## 5. GRAD vs LOCAL_RESPONSE
Mean LR-ERM target difference: `0.001460002362728119`
Mean LR-GRAD target difference: `-0.00899999737739563`
LR wins vs GRAD: `0`

## 6. Mechanism diagnostics
{
  "final_P_LR_lower_than_ERM": 4,
  "target_prediction_color_agreement_lower_than_ERM": 3
}

## 7. Counterexamples / failure cases
Any failed signal criterion is retained in `results/summary.json`; no config rescue or hyperparameter search was run.

## 8. Verdict
CPU-MINIMAL-NO-SIGNAL

## 9. What this does NOT establish
This does not establish theory support, causal recovery, target-risk lower bounds, finite-sample guarantees, universal DG, SOTA, or paper-level readiness.

Historical reopen: none.
