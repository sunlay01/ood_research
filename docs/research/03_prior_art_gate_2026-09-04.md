# 2026-09-04 表示层分解与算法归因先验闸门

> 本文件把附件和历史讨论当作研究上下文，不把其中的建议自动视为结论。当前实际课题是表示层分解先行的跨域算法归因；跨任务路线暂时搁置。

## 当前裁决

- 总状态：`PIVOT_REQUIRED / NO_EXPERIMENT_AUTHORIZED`
- 当前候选：对同一算法及其 \(\lambda\) 路径，先在每个严格 representation-level construction 内定义响应，再只在可证明可比的分量间审计归因的一致、冲突或 `UNDEFINED`。候选 construction 图谱已登记在 `docs/research/04_latent_space_decomposition_taxonomy.md`。
- 研究定位：**跨分解归因审计候选**，不是新的正则算法、不是新的单一分解，也不是已经证明的 target-risk bound。
- `AUDIT-001`：`BLOCKED_BY_PRIOR_ART_GATE`。

## 已有路线的裁决

| 路线 | 代表证据 | 六轴比较后的裁决 | 处理方式 |
| --- | --- | --- | --- |
| gradient/Hessian discrepancy -> target/transfer risk | P-005、P-007 | `CLOSE_EQUIVALENT` | 保留为 exact identity、conditional bound 的审计基线，不作为论文主贡献 |
| IRMv1 scalar projection failure | P-006、P-011 | `CROWDED_INCREMENTAL` | 保留为 negative control，不能宣称首次发现 |
| `Omega-only` target-risk certificate | P-003、P-013 及成熟 DA/DG impossibility 族 | `CROWDED_INCREMENTAL` | 保留最小反例与必要条件，不作为独立新颖性主张 |
| 单一 representation decomposition / nested oracle 下的算法比较 | P-001、P-002 | `CLOSE_EQUIVALENT` | 不以新分解或 lambda sweep 推进论文 |
| objective-to-representation regularization path | P-009 | `CLOSE_EQUIVALENT` 高风险近邻 | 不以 "正则改变 mode" 作为独立贡献 |
| 多个严格 decomposition 的 crosswalk 与 attribution agreement/disagreement | P-001、P-002、P-008、P-009；当前定向检索 | `INSUFFICIENT_PRIOR_ART_EVIDENCE` | 只允许补充原始论文检索和六轴比较，尚无实验授权 |

## 六轴比较

| 轴 | 当前课题的固定版本 | 必须与近邻区分的内容 |
| --- | --- | --- |
| `problem` | 同一算法的 representation-level attribution 是否依赖所选 decomposition | 是否已有工作系统量化跨 construction 的一致/冲突 |
| `mechanism` | construction-specific components + pre-registered semantic crosswalk | 是否已有工作给 crosswalk、尺度与 `UNDEFINED` 规则，而不只是并列多个 decomposition |
| `objective` | 不改变算法目标；使用 \(S_{j,k}^{(d)}(\lambda)\) 作为响应 | 是否已有算法的同路径跨分解比较，而非比较不同方法 |
| `assumptions` | 各 construction 的反事实成立；共同模型类；crosswalk 与正尺度预先固定 | crosswalk 是否偷用 target outcome 或不可识别 latent rotation |
| `evaluation` | exact toy model 优先；只比较 compatible components；冲突需替代解释 | 是否已有同等诊断，或只是 endpoint OOD accuracy correlation |
| `claim` | 一致、冲突或不可定义的可反驳分类；风险界为第四步 | 是否有新的、重要的 theorem 或机制预测；命名一个 dashboard 不构成贡献 |

## 重点证据判断

1. P-001 的 representation-level risk decomposition 与 P-002 的 nested-oracle/failure-mode 分析说明“误差分量化”本身不是空白；尤其 P-002 已比较算法、regularization strength 与训练时长下的分量响应。新分解或 regularization sweep 是 close equivalent。
2. P-003 表明 target-risk bound 需要 smoothness、related-target 或类似 coverage 条件，并警告 constant/collapsed representation。它直接约束 `Omega-only` 和无 remainder 的证书主张。
3. P-005 与 P-007 已覆盖 derivative/moment alignment 到 transfer 或 multi-source target-error 的重要部分。因此 gradient/Hessian 只能作为审计基线，除非发现不同的误差对象、tightness 或 impossibility 结论。
4. P-008 的 latent fine-grained bound 与 P-009 的 regularization-path 工作进一步压缩“表示分解”或“正则改变 mode”作为独立贡献的空间。
5. 截至 2026-09-04 的精确标题/关键词定向检索未检出以 "cross-decomposition attribution agreement/disagreement" 为主张的直接论文；该结果只说明检索尚不充分，绝不支持 novelty claim。下一轮必须用原始索引与引文链逐篇核验。
5. P-014 只作为跨任务历史近邻保存，第一阶段不据此恢复 cross-task 研究，也不把它当作当前证书路线的差异点。

## taxonomy 完成后的范围判断

当前已明确区分八类对象：

1. 条件期望/Bayes projection；
2. representation-conditioned risk transport；
3. nested-oracle failure decomposition；
4. latent direct-sum/factor decomposition；
5. latent smoothness/coverage transport；
6. approximation-estimation-optimization-shift 外层脚手架；
7. feature deficiency/decision-risk comparison；
8. objective-to-mode/spectral regularization path。

这一步只是 `CANDIDATE_TAXONOMY_COMPLETE`，不是 prior-art 通过，也不是新增 theorem。它修正了此前只用 latent oracle/head 两项记账的范围错误。首轮实验诊断应优先实现 1、3、4、5，并以 ERM 作为固定跨算法 anchor；2、6、7、8 作为 transport、学习误差、信息论和路径解释的补充层。没有完成 crosswalk 和原始引文链核验前，`AUDIT-001` 继续保持 `BLOCKED_BY_PRIOR_ART_GATE`。

## 本轮检索记录与边界

- 直接原始来源复核：P-001 的摘要、§3.1、§4.1 和其与已有 decomposition 的比较；P-002 的摘要、§3–5、算法列表和 Fig. 3。
- exact-overlap 查询：`domain generalization multiple error decompositions regularization path`、`representation decomposition algorithm attribution`、`nested oracle regularization strength domain generalization`、`decomposition disagreement representation learning`。
- 结果：没有检出以同算法、同一路径、多个严格 constructions 的 attribution agreement/disagreement 为主张的直接论文。P-001 比较 decomposition，P-002 比较算法路径，但两者的组合不是缺口证明。
- 限制：Semantic Scholar 的匿名接口 rate-limited，Crossref 的关键词结果噪声较高；当前没有完成 P-001/P-002/P-008/P-009 的前向/后向引文链逐篇筛查。因此裁决必须维持 `INSUFFICIENT_PRIOR_ART_EVIDENCE`。

## 授权条件

只有同时满足以下条件，才可以把 `AUDIT-001` 改为 `PROBE_AUTHORIZED`：

1. 六轴核验确认现有工作没有完整覆盖同算法、同 \(\lambda\) 路径、两个严格 construction 和预注册 crosswalk 的 attribution comparison；
2. 在一个可解模型中给出至少一个 nontrivial crosswalk，或严格证明其不能存在；
3. 该对象给出与 P-001/P-002/P-008/P-009 不同的、可反驳预测，不是多画一组曲线；
4. 预注册的最小 probe 能区分 "agreement is artefact" 和 "conflict identifies a construction assumption/head-collapse issue"。

否则：停止该候选，不通过增加 benchmark、lambda、分解名称或模型规模来规避闸门。target-risk certificate 的历史材料只保留为第四步风险检查工具。

## 证据边界

`VERIFIED` 只表示原始来源、版本和记录的直接主张已被定向核验，不表示该工作逐页审阅或已经完成当前六轴比较。`PARTIALLY_VERIFIED` 不可单独支撑正式新颖性结论。具体来源见 `01_evidence_register.md`，逐项比较见 `02_literature_matrix.md`。
