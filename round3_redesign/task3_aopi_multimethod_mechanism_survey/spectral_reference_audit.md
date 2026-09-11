# Spectral reference audit

This audit distinguishes definition-faithful common-harness variants from paper benchmark reproduction.

- `SPECTRAL_NORM_REG`: implemented as source risk plus `lambda * sum_l sigma_1(W_l)^2`; this is a regularizer, not `torch.nn.utils.spectral_norm`.
- `SPECTRAL_REG_2024`: implemented as `sum_l ((sigma_1(W_l)^k - 1)^2 + ||b_l||^(2k))` with fixed `k=2`.
- `SVB_ORTHDNN`: implemented as post-optimizer singular-value bounding with band `[1/(1+factor), 1+factor]`.
- `STABLE_RANK_NORM`: implemented as post-optimizer tail singular-value projection to a fixed stable-rank target followed by sigma_1 normalization.
- `SVD_SPARSE`: deferred because true SVD parameterization/singular-value sparsification would change the fixed model parameterization and continuation state.

No spectral method uses target data for coefficient selection or admission.
