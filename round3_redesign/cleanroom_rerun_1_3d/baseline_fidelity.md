# Baseline Fidelity

Gate: `BASELINE-FIDELITY-PASS`

This stage freshly trains ERM and IRMv1 on the corrected CPU ColoredMNIST protocol. Target accuracy is evaluation-only.

ERM mean target accuracy: `0.10697000026702881`
IRMv1 mean target accuracy: `0.6781499981880188`
IRMv1 minus ERM: `0.57117999792099`

```json
{
  "all_runs_finite": true,
  "enough_optimizer_steps": true,
  "erm_color_shortcut": true,
  "irmv1_distinct_from_erm": true,
  "irmv1_not_ten_percent": true,
  "not_all_irmv1_equal_erm": true,
  "reference_protocol_recorded": true,
  "target_not_used_for_training_or_selection": true
}
```
