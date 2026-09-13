# Route comparison

| Criterion | A: IPM local lemma | B: conditional identification | C: DRO target set | D: meta-domain / impossibility |
|---|---:|---:|---:|---:|
| Scientific adequacy | high for marginal shifts with conditional budget | high for mechanism shifts | high for declared robust shifts | high when target is narrowed honestly |
| Formalization fidelity | high for MMD/CORAL | high for ideal IRM/ICP; medium for IRMv1 | high for GroupDRO/DRO | high for expected-domain or lower-bound query |
| Target-risk relevance | medium-high | high under SCM | high inside `U` | medium for point target, high for impossibility |
| Information sufficiency | conditional term required | strongest under coverage; fails outside SCM | sufficient only inside `U` | explicitly exposes insufficiency |
| Source-only compatibility | conditional | conditional | strong | strong |
| Algorithm fidelity | exact for MMD; translated CORAL | exact ideal; translated IRMv1 | exact | exact at risk/meta level |
| Assumption cost | medium-high | high | medium | medium |
| Irreducible ambiguity | explicit conditional radius | non-identifiability/rank failure | out-of-set misspecification | tail or lower bound |
| Proof tractability | high in RKHS/finite classes | medium in restricted models | high | high |
| Reuse of existing theory | high | high | high | high |
| Added machinery | low | medium | low | low |
| Scientific value if solved | one alignment family | mechanism-level restricted result | robust deployment guarantee | honest scope or impossibility |

No route is selected by novelty. Route C is the cleanest control comparison; Route A is the most direct MMD problem; Route B is the highest-value but highest-assumption IRM problem; Route D is the fallback when the bridge cannot be proved.
