# Controlled Harness

The controlled harness uses the Facebook Colored MNIST identity: `digit < 5`, label noise `0.25`, source color flip probabilities `[0.2, 0.1]`, target flip `0.9`, 2-channel 14x14 inputs, BCE-with-logits, Adam, L2 `0.001`.

The local run is bounded: `501` steps on `256` examples per source environment. It is a port-fidelity smoke, not a performance benchmark.

Controlled result rows are in `results/controlled_run_table.csv`; aggregate means are in `results/controlled_summary.csv`.
