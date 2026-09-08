# C002-IRM: Standard IRMv1 Has a Source-Optimal Blind Direction

## Status

`DISPROVED_AS_GENERAL_CALIBRATION / LITERATURE_COLLISION_PENDING`

## Claim Tested

Can the standard scalar-scale IRMv1 penalty, together with vanishing source
observational excess, certify robust risk under the C001 correlation
intervention family?

## Counterexample

Take the scalar C001 model:

\[
L=\beta=1,\quad \operatorname{Var}(\xi)=2,
\quad \operatorname{Var}(\eta)=\tfrac1{50},
\quad \sigma_Y=\tfrac1{10}.
\]

The two source nuisance relations are

\[
r_1=\tfrac7{10},\qquad r_2=-\tfrac1{10}.
\]

The source-mixture feature moment and feature-label moment are

\[
\Sigma_S=
\begin{pmatrix}3&3/10\\3/10&27/100\end{pmatrix},
\qquad c_S=(1,3/10)^\top.
\]

Therefore the population affine source ERM is

\[
w_{\rm ERM}=(w_U,w_A)=\left(\tfrac14,\tfrac56\right).
\]

For zero intercept, write the unscaled scalar-scale derivative as

\[
q_r(w)=w^\top\Sigma_rw-w^\top c_r.
\]

At this ERM point,

\[
q_r=-\tfrac7{144}-\tfrac5{12}r+\tfrac{25}{36}r^2.
\]

Thus \(q_{7/10}=q_{-1/10}=0\), and the standard penalty

\[
\Omega_{\rm IRMv1}(w)=\tfrac12\sum_{e=1}^2[2q_{r_e}(w)]^2
\]

is exactly zero.  Source risk equals the source observational oracle risk:

\[
R_S(w_{\rm ERM})=R_S^{X,*}=\tfrac{51}{100}.
\]

However, \(r_T=-1\) belongs to the correlation ball
\(\{r:|r|\le1\}\), retains the same task mechanism, and has

\[
R_T(w_{\rm ERM})-R_C^*=\tfrac{127}{48}>0.
\]

## Conclusion

There is no function \(\psi\) with \(\psi(0,0)=0\) such that, over this
model class,

\[
R_S(w)-R_S^{X,*}\le\varepsilon,
\quad\Omega_{\rm IRMv1}(w)\le\rho
\quad\Longrightarrow\quad
R_{\mathcal I_{\rm corr}}(w)-R_C^*\le\psi(\varepsilon,\rho).
\]

The counterexample has \(\varepsilon=\rho=0\), so this is not a vacuous
zero-level failure.  It is a regularizer-specific blind direction despite
nonzero source gradient-response observability.

Moreover, because this predictor simultaneously minimizes source risk and has
nonnegative penalty zero, it is a global minimizer of
\(R_S+\lambda\Omega_{\rm IRMv1}\) for every \(\lambda\ge0\) in the fixed
identity-representation affine class.

## Scope Boundary

This is a statement about the standard scalar-scale IRMv1 penalty, not full
shared-head gradient matching.  It is not yet a novelty claim: known IRMv1
failure work is a high-risk collision and must be audited before any paper
claim.

## Verification

`python -m ood_repr_reg.intervention_report` emits the full population report;
the counterexample is unit-tested in `tests/test_intervention_linear.py`.
