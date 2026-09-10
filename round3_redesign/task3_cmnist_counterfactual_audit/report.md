# TASK3-CMNIST-COUNTERFACTUAL-DIAGNOSTIC-PORT Report

## 1. Validity and provenance
Git HEAD: `50d435b403bbda00427b9cce38258066eacc40db`
Config SHA256: `a1083f49401660ecbb1f1b19e75848ad24612750b34e5255131945ebbbbbe60d`
Checkpoint source: `reconstructed`
Reconciliation status: `PASS`
Diagnostic rows: `10`
Test results: `pre-run targeted audit pytest: 10 passed, 1 skipped; post-run targeted audit pytest: 11 passed; project-state checker: ok true; project-state pytest: 5 passed; CPU-minimal pytest: 26 passed`
Errors: `none`

## 2. Existing corrected performance gap
ERM mean target accuracy: `0.1098`
IRMv1 mean target accuracy: `0.66914`
IRMv1 minus ERM: `0.55934`
Paired target-accuracy deltas: `['0.5476', '0.5541', '0.5682', '0.5732', '0.5536']`

## 3. Representation content
latent_color_response: ERM mean `92.6933`, IRMv1 mean `125.524`, delta mean `32.8306`, ratio mean `1.35695`
task_signal: ERM mean `2.74367`, IRMv1 mean `1.85515`, delta mean `-0.888518`, ratio mean `0.676156`
task_color_overlap: ERM mean `0.00248385`, IRMv1 mean `0.00276057`, delta mean `0.000276719`, ratio mean `1.12307`
balanced_clean_accuracy: ERM mean `0.50618`, IRMv1 mean `0.76516`, delta mean `0.25898`, ratio mean `1.51166`

## 4. Head usage
prediction_color_response: ERM mean `12.4897`, IRMv1 mean `0.266303`, delta mean `-12.2234`, ratio mean `0.0214241`
probability_color_response: ERM mean `0.410207`, IRMv1 mean `0.0134009`, delta mean `-0.396806`, ratio mean `0.0327256`
task_head_margin: ERM mean `1.45068`, IRMv1 mean `0.736835`, delta mean `-0.713841`, ratio mean `0.507974`
counterfactual_prediction_consistency: ERM mean `0.01244`, IRMv1 mean `0.80036`, delta mean `0.78792`, ratio mean `73.2784`
counterfactual_prediction_flip_rate: ERM mean `0.98756`, IRMv1 mean `0.19964`, delta mean `-0.78792`, ratio mean `0.202135`

## 5. Paired ERM-IRM decomposition
Is the large target-accuracy gap accompanied mainly by representation differences, head-use differences, or both? The per-seed deltas and ratios are saved in `results/paired_effects.csv`; this report uses only those saved values.

## 6. Verdict
MIXED-DECOMPOSITION

## 7. What this does not establish
This does not establish causality, source identifiability, a new objective, frozen theory validation, novelty, target-risk lower bounds, finite-sample guarantees, or restoration of old CMNIST empirical results.

## 8. Next experiment only if justified
A later causal intervention must independently alter encoder content and head use to separate the explanations.

Historical reopen: none for empirical evidence; old diagnostic code was formula reference only.
