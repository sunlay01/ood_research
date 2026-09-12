# Fishr five-seed local C/K analysis

Frozen factorial outputs at rho=1e-3; all 5 seeds and all four Fishr cells pass the prescribed gates.

The mean contrast norms are:
- delta_C: norm 0.001009 (sd 0.000331), source 0.000571, counterfactual 0.000759, clean 0.000335, cosine-to-R00 mean -0.472 (sd 0.825).
- delta_K: norm 0.000277 (sd 0.000058), source 0.000157, counterfactual 0.000209, clean 0.000089, cosine-to-R00 mean -0.788 (sd 0.323).
- delta_interaction: norm 0.000815 (sd 0.000253), source 0.000468, counterfactual 0.000608, clean 0.000269, cosine-to-R00 mean 0.462 (sd 0.816).

Delta_C is largest on average, Delta_K is smaller, and Delta_CK is substantial. The interaction is therefore not negligible. Direction cosines have large seed dispersion, so no single seed-invariant response direction is established. Bank allocation is source-dominant, with counterfactual response larger than clean response.

These are numerical properties of the selected local projection and checkpoint. They do not identify semantic forcing/filtering or establish target-risk transfer.
