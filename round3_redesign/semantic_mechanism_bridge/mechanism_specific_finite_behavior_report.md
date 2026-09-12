# Mechanism-specific finite behavior audit

Local response is estimated with epsilon=0.01 on fixed CMNIST semantic coordinates. Outcomes are evaluated at held-out finite shifts alpha in {0.1,0.2,0.3}; target accuracy is post-hoc metadata only.

The response geometry is evaluated per semantic mechanism, rather than against one pooled target-accuracy scalar.

## source_color_common

Empty DataFrame
Columns: [(method, ), (accuracy_change, mean), (accuracy_change, std), (loss_change, mean), (loss_change, std)]
Index: []

## source_color_contrast

Empty DataFrame
Columns: [(method, ), (accuracy_change, mean), (accuracy_change, std), (loss_change, mean), (loss_change, std)]
Index: []

## source_label_noise

Empty DataFrame
Columns: [(method, ), (accuracy_change, mean), (accuracy_change, std), (loss_change, mean), (loss_change, std)]
Index: []

## Verdict

This first behavior audit reports a mechanism-specific state representation test. BIRM and LoRA-BIRM are included using their official final checkpoints. A positive mechanism claim requires cross-method prediction that beats method identity and a held-out intervention; this artifact does not grant that claim automatically.