# Prompt：理论建模与表示层误差分解

```text
请在已完成文献与新颖性闸门后，为一个明确、狭窄的候选问题推进理论工作。只处理一个正则化方法和一个简化模型；不把探索性的代数直觉写成定理。

先阅读 PROJECT_PROMPT_CN.md、CONTEXT_MANAGEMENT.md、docs/research 的 evidence/literature/decision 文件，以及 docs/theory/00_theory_ledger.md。若 prior-art verdict 不是 DISTINCT_BUT_RISKY、CLEAR_NOVELTY_GAP，或明确范围内的 PROBE_AUTHORIZED，先停止并指出原因。

使用 $ars-codex:academic-research-suite 对将要依赖的数学先例做 targeted three-way-scan 或 fact-check；引用只允许使用已核验来源。然后按以下顺序工作：

1. 选择唯一目标：例如“在线性 core/spurious 多环境模型中分析 IRMv1 的 feature-moment 约束”，而不是泛称“解释 IRM”。
2. 固定损失、环境分布、`x→z=Bx→ŷ=wᵀz` 参数化、最优解选择规则和所有可辨识性条件。
3. 明确选用一种 construction：representation-conditioned Bayes / hybrid distribution、nested oracle、direct-sum latent decomposition，或局部 implicit-differentiation。解释它解决何种混淆，而不只是罗列名词。
4. 先写 definition 与 assumptions，再写一个最小命题。把结论准确标为 exact equality、conditional theorem、local result、bound 或 conjecture。
5. 为每个关键步写 proof obligation；主动构造至少一个反例、退化情形或替代解释，例如 head mismatch、support loss、uniform shrinkage、collapse、非唯一最优解。
6. 若要用 `S_{j,k}(λ)`，先定义 `E_k` 的可估计性和其与 target risk 的具体关系。`S<0` 只能说明已定义响应的变化，不能自动宣称因果机制。

写回 docs/theory/00_theory_ledger.md：新增或更新定义、假设、命题和 DERIVATION 条目；必要时新增一个 Markdown 推导文件。最后报告：已证实部分、未闭合步骤、最小数值检验，以及什么结果会推翻该路线。
```
