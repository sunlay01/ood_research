# 第三轮补充实验报告：风险响应结构重做

## 裁决：`LOW-RANK-BUT-NONSEMANTIC`

本报告是对旧第三轮 empirical discovery 的补充重做。旧结果保留为历史记录，不再被视为机制分解证据。

## 设计核验

- 真实训练模型：`660` 个；方法：`5`；每种方法 seed：`20`；正则强度：`8`。
- source environments：`7`；shift probes：`300`。
- response matrix：`[660, 300]`，主要分析对象是 shift columns。
- discovery 使用 target 风险：`False`；使用机制标签：`False`。

## Blind discovery

| 输入 | effective rank | selected k | silhouette | bootstrap stability |
|---|---:|---:|---:|---:|
| raw | 17 | 3 | 0.6922 | 0.9694 |
| model-normalized | 17 | 2 | 0.8761 | 1.0000 |
| shift-normalized | 16 | 2 | 0.4483 | not primary |

`effective rank` 只说明响应矩阵的数值维数，不自动说明存在机制语义。

## Post-hoc validation

机制 enrichment 和 counterfactual validation 只在 blind discovery 完成后运行。加权 cluster purity 为 `0.2133`，pair cluster consistency 为 `0.8867`。这些数值都不是机制识别定理，必须与跨 seed、跨方法稳定性和 hidden-composition 检验一起解释。

## 正则与风险响应

| method | mean source risk | mean absolute response | response by shift kind |
|---|---:|---:|---|
| CORAL | 0.26456 | 0.00808 | {'compound': 0.0028436331566854345, 'core_covariance': 0.0011069199360985622, 'core_mean': 0.0012018038205511695, 'nuisance_covariance': 0.002226247467424983, 'nuisance_mean': 0.0014997743486269302, 'observation': 0.003436679088436022, 'relation': 0.0022844278007312637, 'task': 0.049510674357505585} |
| ERM | 0.25000 | 0.00611 | {'compound': 4.502571155424246e-16, 'core_covariance': 5.142085587737567e-16, 'core_mean': 3.9734297723426655e-16, 'nuisance_covariance': 5.258951169277057e-16, 'nuisance_mean': 5.142085587737567e-16, 'observation': 5.366077952354923e-16, 'relation': 5.667980704665273e-16, 'task': 0.04822280005467389} |
| IRMv1 | 0.25000 | 0.00611 | {'compound': 1.3442983608996033e-12, 'core_covariance': 7.119569188582107e-13, 'core_mean': 9.1420697766406e-13, 'nuisance_covariance': 1.0415981673663872e-12, 'nuisance_mean': 1.176734952169692e-12, 'observation': 1.770655420125284e-12, 'relation': 1.057993203280068e-12, 'task': 0.048222800054674296} |
| L2 | 0.48598 | 0.02409 | {'compound': 0.01787835916840936, 'core_covariance': 0.03616145963707658, 'core_mean': 0.017533102899870694, 'nuisance_covariance': 0.003710910962655728, 'nuisance_mean': 0.002705504390638236, 'observation': 0.003039635466870315, 'relation': 0.012784361789079778, 'task': 0.09746919283930067} |
| V-REx | 0.25000 | 0.00611 | {'compound': 1.9881420583918915e-12, 'core_covariance': 1.0075934969418878e-12, 'core_mean': 1.249258299146642e-12, 'nuisance_covariance': 1.8717658271281017e-12, 'nuisance_mean': 1.5198929468797144e-12, 'observation': 2.6428526805282675e-12, 'relation': 1.5469658448966814e-12, 'task': 0.04822280005467435} |

相同 shift-side clustering 在 raw 与 model-normalized 输入上的 pairwise 一致率为 `0.8990`，raw 与 shift-normalized 的一致率为 `0.5260`。因此 raw 中的稳定簇不能直接解释为机制簇；需要先排除响应尺度、正则强度和模型表型因素。

## 当前结论

本次实验修复了旧实验的四个结构性问题：模型由目标函数训练、使用二维 core/nuisance、以 shift columns 为发现对象、并扩大了模型和 shift 数量。当前裁决是 `LOW-RANK-BUT-NONSEMANTIC`：稳定性存在，但机制 enrichment、归一化一致性和 cluster 几何不足以支持新的语义分解候选。响应结构更像“风险幅度/模型表型分层”，而不是 core、nuisance、relation 等生成机制的自然分解。

完整机器结果见 `results/round3_retry_results.json`，响应矩阵见 `results/response_*.csv`。
