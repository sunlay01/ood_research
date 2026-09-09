# IFT Bridge

At a regularized head solution `w_j`, the source first-order condition is
`F_j(w,psi)=0`.  The implementation computes `D_w F_j` and `D_psi F_j` by
Torch double autodiff and forms `Pi=-D_wF_j^dagger D_psiF_j`.  Retrained-head
central differences are compared with `Pi O_S` over three steps.

Rows with non-positive or non-invertible local metrics are invalid and do not
enter a pass claim.  No damping is added to rescue such rows.  The result is a
local empirical audit for the frozen head.
