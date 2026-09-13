# Extractor A — Discrepancy / alignment

| paper | Omega | bridge chain | target | fidelity | irreducible |
|---|---|---|---|---|---|
| BenDavid2010 | HΔH empirical discrepancy | `d_HΔH(S,T) -> uniform loss-class difference -> R_T <= R_S+d+lambda*` | target risk | EXACT theorem | target samples and joint-error oracle `lambda*` |
| Mansour2009 | loss-class discrepancy | `disc_L(P,Q) -> loss expectation difference -> transfer bound` | target risk | EXACT | target discrepancy / ideal joint hypothesis |
| Zhao2019 | marginal feature alignment | `d(P_S^Z,P_T^Z) small ->? conditional mismatch` | target risk | COUNTEREXAMPLE to bridge | conditional-label mismatch |
| Gretton2012 | MMD | `RKHS mean distance -> IPM expectation gap` | distribution gap | EXACT | witness-class limitation |
| Shui2022 | INV representation | `TV(source representation) -> Dobrushin contraction -> target risk` | unseen-domain risk | POPULATION-ABSTRACTION | invariant conditional and contraction assumptions |

Repeated objects: IPM/discrepancy, loss-class witness, target discrepancy, joint/conditional residual, contraction coefficient. Alignment-to-target is not source-only unless a target family or contraction assumption is supplied.

## Exact chain ledger (Ω → B, intermediate objects, edges)

| paper | exact Ω→B chain | intermediate objects | reusable edge | OOD difficulty term | evidence pointer |
|---|---|---|---|---|---|
| BenDavid2010 | `D_S^Φ,D_T^Φ → Ω=d_{HΔH}; ε_T(h)≤ε_S(h)+½Ω+λ*`; empirical `d̂` + `4√((2d log(2m')+log(2/δ))/m')` (Thm 2) | `HΔH`, pushed-forward marginals, ideal joint `h*` | risk triangle + `|ε_S(h,h')−ε_T(h,h')|≤½d_{HΔH}` + VC concentration | `λ*=min_h(ε_S+ε_T)`; target samples required for Ω | `BenDavid2010.pdf`, Thm 2, Defs 2–3, Lemma 3; `literature/ledger/details/BenDavid2010.md` |
| Mansour2009 | `Ω=disc_L(P,Q)=sup_{h,h'}|L_P−L_Q|`; Thm 8 `L_P(h,f_P)≤L_P(h_P*,f_P)+L_Q(h,h_Q*)+Ω+min{L_Q(h_Q*,h_P*),L_P(...)}`; Thm 9 adds empirical/Rademacher terms | loss class `L_H`, `h_P*,h_Q*`, empirical distributions | loss triangle + loss-class uniform convergence | ideal-hypothesis/conditional mismatch; target discrepancy in DA | `Mansour2009.pdf`, Def. disc_L, Props 2, Cor 6–7, Thms 8–9; ledger card |
| Gretton2012 | `Ω=MMD[F,p,q]=||μ_p−μ_q||_H`; Thm 7 `|MMD̂−MMD|=O(√(K/m)+√(K/n))`; if risk witness `g_h∈H`, `|E_p g_h−E_q g_h|≤||g_h||_HΩ` and then insert into a transfer bound | RKHS mean embeddings `μ`, witness `g_h`, bounded-kernel U/V statistic | RKHS duality + bounded-kernel concentration | no risk B from Gretton alone; target samples and conditional/joint mismatch remain | `Gretton2012.pdf`, Eq. 1, mean-embedding identity, Thm 7; `generalization_proof_techniques.md` |
| Zhao2019 | `Ω=d_{\tilde H}(D_S^g,D_T^g)`; Thm 4.1 `ε_T≤ε_S+Ω+min{E_{D_S}|f_S−f_T|,E_{D_T}|f_S−f_T|}`; Thm 4.2 empirical + `2Rad(H)+4Rad(\tilde H)+O(√(log(1/δ)/n))` | representation `Z`, `\tilde H`, source/target labelers, JS label marginals | change-of-measure decomposition; JS data processing for lower bound | explicit conditional-label shift; aligned marginals can still yield joint error 1 (counterexample) | `Zhao2019.pdf`, Sec. 4.1, Thms 4.1–4.3, Lemma 4.8; ledger card |
| Shui2022 | `Ω=INV` with latent conditional TV `κ`; nearest-source raw TV `ε`; channel contraction `α_TV(Φ)`; Prop. 1 `BER_T≤T^{-1}Σ BER_{S_t}+κ+α_TV(Φ)ε` | latent `S_t(z|Y)`, nearest `S*`, `κ,ε`, channel `Φ`, Dobrushin coefficient | TV triangle + strong data processing + bounded-loss conversion | `κ+α_TV(Φ)ε`; ε unobserved, nearest-source family required | `Shui2022.pdf`, Eq. 1, Prop. 1/proof; ledger card |

### Minimal bridge retained by extractor A

Discrepancy/IPM regularization contributes a shift term only through an explicit witness class and a proved inequality.  A valid OOD claim must additionally expose either a joint/conditional mismatch (`λ*`, `disc_L` oracle, Zhao labeler gap), or a declared target-family/coverage and contraction term (Shui `ε,κ,α_TV`).  Marginal MMD/alignment by itself is not a conditional-invariance certificate.
