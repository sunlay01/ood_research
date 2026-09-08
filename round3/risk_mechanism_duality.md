# Risk-mechanism duality

predictor side 是 \(Q_f=vv^T\)，environment side 是 structural tangent。二者通过 \(DM\) 配对。对 mechanism tangent \(\mathcal T_k\)，定义 \(S_k(f)=dR_f|_{\mathcal T_k}\)。其 dual norm 是模型对该生成机制局部变化的 sensitivity。

该对象比任意 latent basis 更严格，但仍依赖声明的 structural family 和 tangent norm。
