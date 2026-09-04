# 课题简报：OOD 正则 → 表示空间 → 误差

> 状态：`WORKING_BRIEF`。本文件只固定初始范围，所有文献事实与新颖性主张须在 Stage 1–2 后更新。

## 1. 母题

给定多环境训练目标

\[
(\theta_\lambda,w_\lambda) \in \arg\min_{\theta,w}
R_S(w\circ\Phi_\theta)+\lambda\Omega(\theta,w),
\qquad Z_\lambda=\Phi_{\theta_\lambda}(X),
\]

研究 \(\Omega\) 如何通过参数响应诱导 \(Z_\lambda\) 的结构变化，并影响 OOD / target risk。

## 2. 需要回答的三个层次

1. **表示层**：是否能以 \(Z\) 而非参数表示原正则，例如定义诱导正则 \(\Psi(Z,w)=\inf_{\theta: \Phi_\theta(X)=Z}\Omega(\theta,w)\)？
2. **误差层**：能否从 hybrid distribution、nested oracle、direct-sum 或局部响应构造 representation-level quantities \(E_k(Z)\)，使 OOD 风险可被等式、局部展开或上界表达？
3. **归因层**：对于方法 \(j\)，能否通过 \(S_{j,k}(\lambda)=E_k(Z_{\lambda,j})-E_k(Z_0)\) 说明它改变了什么，并区分 selective suppression、uniform shrinkage 和 representation collapse？

## 3. 首批可证伪研究问题（待查重）

| ID | 研究问题 | 可反驳标准 |
| --- | --- | --- |
| RQ-A | 在线性 core/spurious 多环境模型中，IRMv1 诱导的表示约束是否等价于某个特征 moment discrepancy？ | 找到相同参数正则但不同 moment，或反例破坏等价。 |
| RQ-B | 哪些正则通过降低 spurious loading 改善 target risk，哪些只是整体 shrinkage？ | 对 core/spurious loading 和 OOD risk 的正则路径不支持此区分。 |
| RQ-C | 能否以一组 representation-level counterfactual errors 比较 ERM、IRMv1、谱正则的作用机制？ | 分量无法定义、不可估计、不可区分或不能预言任何现象。 |

这些 RQ 不能绕过新颖性闸门：若相同对象、机制、目标和主张已被文献覆盖，必须 `STOP` 或提出机制级转向。

## 4. 初始建模边界

- 从线性表示 \(z=Bx\) 与线性 head \(\hat y=w^\top z\) 开始；允许在证据支持下扩展到二层网络。
- 生成结构必须显式写出：例如 \(x=A_c z_c+A_s z_s+\epsilon\)，以及每个环境中 \(P_e(z_s\mid y)\) 如何变化、何者保持稳定。
- ERM 是基线；首批正则限于一到两种，以避免“算法罗列”。
- 每一条 theorem 要标记为 `exact equality`、`conditional theorem`、`local result`、`bound` 或 `conjecture`。
- 每一个实验结论要分别报告预测性能、spurious 依赖诊断、表示几何量与对照结果。

## 5. 不在当前范围

- 一开始分析任意深度、非凸网络的全局最优性。
- 声称解释所有 OOD/DG 算法，或以最终准确率给出因果机制结论。
- 未经检索就以“没人做过”或“顶会潜力”作为项目理由。
- 用难以复现的大规模模型或私有数据掩盖理论不可辨识性。

## 6. 立即执行的证据工作

1. 以方法而非单一关键词检索：representation risk decomposition、hybrid distribution、oracle DG error、IRM objective reinterpretation、gradient/Hessian/moment alignment、regularization path、feature-mode dynamics。
2. 对每个近邻工作做 `problem / mechanism / objective / assumptions / claim / evaluation` 六轴比较。
3. 记录相反证据、失败模式和已有负结果；它们与支持性论文同样重要。
4. 在 `01_evidence_register.md` 中登记原始链接和核验状态，再更新 `02_literature_matrix.md`。
