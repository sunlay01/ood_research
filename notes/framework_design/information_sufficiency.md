# Phase IV — Information-sufficiency tests

The test asks whether a retained formal object can distinguish worlds that have
different target risk. Equality of the source summary is not treated as proof
of equality of the target mechanism.

## Test 1: marginal alignment / source risks

Let `X` be binary and let all source environments have the same marginal on
`X`. Construct two admissible target worlds with the same source laws and the
same representation marginal, but with opposite target label conditionals:
`P_1(Y=X)=1` and `P_2(Y=1-X)=1`. A predictor using `X` has target risks 0 and
1, while source risks and marginal MMD/CORAL summaries are identical.

**Conclusion:** a marginal IPM or source-risk vector cannot by itself imply
conditional target alignment. The desired theorem is impossible without a
conditional/joint-error term or a target-family restriction. This is a missing
distinction in the theorem query, not a reason to append an arbitrary new
representation coordinate.

## Test 2: finite source environments / arbitrary target

Take two source worlds that agree on every observed source distribution and
algorithm statistic, then let their unobserved target conditionals disagree.
Any source-only procedure has the same output in both worlds but can incur a
constant-separated target risk. The construction is the standard
indistinguishability obstruction represented in the repository by Zhao-style
conditional conflict and Wang et al. (2024) lost-domain lower bounds.

**Conclusion:** arbitrary point-target guarantees are impossible from finite
source exposure. The correct output is a coverage assumption, an ambiguity
term, a quantile under a meta-law, or a lower bound.

## Test 3: gradient statistics

IRMv1 classifier gradients and Fishr gradient covariances can agree at a
stationary or matched-statistic solution while the conditional mechanism or
target family differs. No general injectivity from these optimizer statistics
to `P_T(Y|X)` is established.

**Conclusion:** retain them as algorithm-level quantities and require a local
translation theorem. Do not promote them to a universal target-risk
representation.

## Representation decision

The failed distinctions are conditional mechanism, target-family membership,
and fresh-domain tail information. Each can be represented by an existing
conditional/SCM object, uncertainty set, or meta-law term. Therefore the tests
do not justify a new universal representation. They justify explicit
assumption-controlled or irreducible terms in any future theorem.
