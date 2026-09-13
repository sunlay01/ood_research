# Generalization test

Take a new regularizer not used to define the clusters: a bounded-kernel conditional mean-embedding penalty.

1. Algorithmic object: empirical conditional embedding discrepancy across source domains.
2. Required new lemma: penalty `<= eps` implies a population conditional witness radius `rho_cond <= eps + concentration`.
3. Existing outer theorem: apply Pattern 4's covariate-plus-conditional decomposition, or Pattern 1 when the loss section lies in the conditional witness class.
4. Result: the target bound is reusable without changing the outer theorem, but only if the new penalty-to-radius lemma is proved and overlap/complexity assumptions hold.

This passes a limited generalization test. It does not show that Fishr or IRMv1 automatically fit; their derivative statistics require separate translation lemmas.
