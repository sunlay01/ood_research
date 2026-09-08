# Residual Coupling

Let `r* = Y - X'w*_S`. The source normal equation gives `E_S[X r*] = 0`.
Consequently,

```text
g_s = -2 (E_T[X r*] - E_S[X r*]) = -2 E_T[X r*].
```

This is the central diagnostic. A zero source coefficient is not sufficient
for zero OOD relevance: a target-emergent feature may couple to the residual.
Conversely, a large common burden can be model-independent and therefore have
zero leading discrimination.
