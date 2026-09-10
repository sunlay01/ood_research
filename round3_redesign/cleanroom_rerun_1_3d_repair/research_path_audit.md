# Research Path Audit

## Old Premise

The 3C-to-Task3 path treated the old CMNIST `helps` rows as evidence that existing source regularizers were already moving the recoverable response residual `E` in a favorable direction, motivating a local-response algorithmic objective.

## Repaired Evidence

Under the source-only bridge and common source-only projection, `12` of `16` old stable `helps` claims have no repaired family counterpart, `4` keep only the sign but fall below 10% of the old magnitude, and `0` persist as non-attenuated repaired claims.

## Impact On Task3

This does not prove `LOCAL_RESPONSE` failed only because the path was contaminated. It does show that the empirical premise used to move from 3C response geometry toward Task3 algorithm design is not cleanly reproduced after the bridge repair. The scientifically correct status is therefore `RERUN-INCONCLUSIVE`, with Task3 algorithm interpretation reopened.

## Affected Claims

| Claim | Change | Old mean E change | New mean E change | New/old magnitude |
|---|---:|---:|---:|---:|
| `declared_source_target_coupled_correlation:IRMV1:lambda=0.001` | `HELP_DIRECTION_PERSISTED_BUT_ATTENUATED_BELOW_10PCT` | -0.000331334 | -8.36891e-06 | 0.02526 |
| `declared_source_target_coupled_correlation:VREX:lambda=0.001` | `HELP_DIRECTION_PERSISTED_BUT_ATTENUATED_BELOW_10PCT` | -0.000118128 | -8.99372e-06 | 0.07614 |
| `declared_source_target_coupled_correlation:VREX:lambda=0.01` | `HELP_DIRECTION_PERSISTED_BUT_ATTENUATED_BELOW_10PCT` | -0.0011783 | -8.97628e-05 | 0.07618 |
| `declared_source_target_coupled_correlation:VREX:lambda=0.1` | `HELP_DIRECTION_PERSISTED_BUT_ATTENUATED_BELOW_10PCT` | -0.0114937 | -0.00088459 | 0.07696 |
| `irrelevant_source_diversity:VREX:lambda=0.001` | `OLD_HELP_NOT_REPRODUCED_NO_REPAIRED_FAMILY_COUNTERPART` | -0.00011213 | NA | NA |
| `irrelevant_source_diversity:VREX:lambda=0.01` | `OLD_HELP_NOT_REPRODUCED_NO_REPAIRED_FAMILY_COUNTERPART` | -0.00111842 | NA | NA |
| `irrelevant_source_diversity:VREX:lambda=0.1` | `OLD_HELP_NOT_REPRODUCED_NO_REPAIRED_FAMILY_COUNTERPART` | -0.0109044 | NA | NA |
| `mechanism_defined_exposed:L2:lambda=0.001` | `OLD_HELP_NOT_REPRODUCED_NO_REPAIRED_FAMILY_COUNTERPART` | -0.0411602 | NA | NA |
| `mechanism_defined_exposed:L2:lambda=0.1` | `OLD_HELP_NOT_REPRODUCED_NO_REPAIRED_FAMILY_COUNTERPART` | -0.564881 | NA | NA |
| `mechanism_defined_exposed:VREX:lambda=0.001` | `OLD_HELP_NOT_REPRODUCED_NO_REPAIRED_FAMILY_COUNTERPART` | -0.000263336 | NA | NA |
| `mechanism_defined_exposed:VREX:lambda=0.01` | `OLD_HELP_NOT_REPRODUCED_NO_REPAIRED_FAMILY_COUNTERPART` | -0.00262773 | NA | NA |
| `mechanism_defined_exposed:VREX:lambda=0.1` | `OLD_HELP_NOT_REPRODUCED_NO_REPAIRED_FAMILY_COUNTERPART` | -0.0257268 | NA | NA |
| `mechanism_defined_hidden:IRMV1:lambda=0.001` | `OLD_HELP_NOT_REPRODUCED_NO_REPAIRED_FAMILY_COUNTERPART` | -0.000331334 | NA | NA |
| `mechanism_defined_hidden:VREX:lambda=0.001` | `OLD_HELP_NOT_REPRODUCED_NO_REPAIRED_FAMILY_COUNTERPART` | -0.000118128 | NA | NA |
| `mechanism_defined_hidden:VREX:lambda=0.01` | `OLD_HELP_NOT_REPRODUCED_NO_REPAIRED_FAMILY_COUNTERPART` | -0.0011783 | NA | NA |
| `mechanism_defined_hidden:VREX:lambda=0.1` | `OLD_HELP_NOT_REPRODUCED_NO_REPAIRED_FAMILY_COUNTERPART` | -0.0114937 | NA | NA |

