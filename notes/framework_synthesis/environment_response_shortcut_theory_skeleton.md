# 环境—响应表示与 shortcut 残留动力学：形式化理论骨架

日期：2026-11-06  
状态：**候选理论骨架；含本文内证明的命题与明确列出的待闭合环节**。  
所属分支：ood-theory。本文不修改 STATUS，不将结论提升为已接受的统一理论。

## 0. 问题、贡献边界与依赖

目标是把本分支的环境状态—风险响应配对，与有限样本 toy 中的补偿、回填和稳定残留接起来：

环境族 → 风险响应表示 → 可实现响应集合 → 约束下的优化选择 → 目标风险配对 → 分类 margin。

既有表示接口：

- [Stage 11R：原始—对偶表示与源暴露](stage11r_metric_gauge_revision.md)
- [Stage 12：population transfer](stage12_population_master_theorem.md)
- [Stage 13：算法翻译](stage13_regularizer_translation.md)
- [Stage 13R.2：raw risk 与 excess risk 的接口限制](stage13r2_transfer_measure_source_identifiability.md)

本文不引入 main 分支的 A/O/Pi 作为基础，不宣称新的 transfer measure，不把信息不可识别性等同于算法失败。核心新增接口是：响应不是可自由选择的对偶向量，而必须由预测器参数实现；源风险在这些可实现响应的等观测集合中选择解。

记号约定：本文用 c 表示源风险对颜色参数的梯度，用 n 表示风险差梯度；不把这些对象与环境状态混用。shortcut 比例用 rho_sc，目标族半径用 r。

## 1. 数据、损失与环境族

基础变量 X0=(D,Y,Z) 的分布 mu 在各环境相同。D∈{-1,+1} 等概率，
Y=DN，Pr(N=-1)=eta∈(0,1/2)。Z∼N(0,1)，独立于 (D,N)。

环境 e 改变 F 的条件分布：
Pr_e(F=1 | X0=x)=p_e(x)∈[0,1]，
C=Y(1-2F)。

预测器与 logistic margin loss：
f_q(D,C,Z)=u_q(D)+C v_q(Z)，
u_q=aD+d，v_q=b+hZ，
q=(a,d,b,h)，ell(t)=log(1+exp(-t))。

假设所用风险可积。对有限参数及高斯 Z 此条件满足。
预测器的形状、颜色系数与环境的条件概率是不同对象。

### 1.1 两个必须分开的环境族

原始统一翻转族：p_e(x)=epsilon_e。

用于检验异质迁移的预先声明二维族：
p_(epsilon,tau)(z)=epsilon+tau sign(z)，
0<epsilon<1，|tau|≤min(epsilon,1-epsilon)。

第二个族允许不同 Z 子群的颜色可靠性改变。它是原 toy 的环境扩展，不是原来 epsilon_T=0.9 的统一翻转目标。族在训练前定义；tau 的正负都在不确定集中，最坏目标可以依赖最终预测器，固定目标的伤害不因此自动为正。

## 2. 精确的环境—响应表示

定义
gamma_q(x)=ell(Yu_q-v_q)-ell(Yu_q+v_q)，
B(q)=E_mu ell(Yu_q+v_q)。

**命题 1（精确配对）。**
对任何可行 p，
R_p(q)=B(q)+E_mu[p gamma_q]。
因此 R_p(q)-R_pbar(q)=E_mu[(p-pbar)gamma_q]。

证明：给定 X0，F=0 与 F=1 时的 signed margins 分别是 Yu+v 和 Yu-v。对 Bernoulli F 条件平均即可。没有 Taylor 余项。□

环境是原始对象 p，风险响应 gamma_q 是对偶对象。B(q) 虽不影响同一预测器的环境风险差，却参与训练中的预测器选择，不能从优化问题中删掉。

### 2.1 有限维、算法独立的总体实例

在二维族中，预先取环境状态 Psi(P_(epsilon,tau))=(epsilon,tau)，
beta_0(q)=E gamma_q，
beta_1(q)=E[sign(Z)gamma_q]。

则
R_(epsilon,tau)(q)=B(q)+epsilon beta_0(q)+tau beta_1(q)。

这是精确的二维 affine raw-risk 表示；Psi 不依赖 q 或算法。两个 tau=0 的源环境仅暴露 epsilon 方向：
Var_e R_e(q)=Var_e(epsilon_e) beta_0(q)^2。

环境状态的维数与网络参数的维数不必相同。一般条件概率族对应函数空间配对；本文不把一般族冒充固定有限维表示。

### 2.2 原配对有限样本的精确实例

固定基础样本 x_i=(D_i,Y_i,z_i)，两个环境的颜色实现为 F_ei∈{0,1}。
记 <r,s>_n=n^{-1}sum_i r_i s_i，
gamma_q=(gamma_q(x_i))_i，
d_F=F_0-F_1，Fbar=(F_0+F_1)/2。

精确有
Rhat_e(q)=Bhat(q)+<F_e,gamma_q>_n，
Deltahat(q)=Rhat_0(q)-Rhat_1(q)=<d_F,gamma_q>_n，
Rhat(q)=(Rhat_0+Rhat_1)/2=Bhat(q)+<Fbar,gamma_q>_n。

样本条件下这是 n 维表示。它用于原训练目标的精确分析，不等于总体二维状态；F_ei 的随机实现包含采样扰动。经验硬约束不能直接替换为总体风险差为零。

## 3. 可实现响应与算法约束

定义响应像 M={gamma_q:q∈Theta}。不预设 M 是全局光滑流形或线性空间。
定义源观测算子 O gamma=(<p_e-pbar,gamma>)_e。
有限配对两环境中可以使用标量 O_n gamma=<d_F,gamma>_n。

算法只通过参数 q 访问 M；gamma 与 B(q) 由相同参数共同产生。单独在整个对偶空间上自由优化 gamma 通常不是原问题。

### 3.1 精确原生 V-REx 与归一化

采用训练目标
J_lambda(q)=Rhat(q)+lambda Deltahat(q)^2。

两个等权环境的原生风险方差为 Deltahat^2/4，所以原生 V-REx 权重 Lambda 对应 lambda=Lambda/4。
如果旧实验使用风险和 Rhat_0+Rhat_1，应先统一总体目标缩放及时间尺度，再比较 lambda 与动力学数值。

### 3.2 补偿的定义

记 n_q=grad Deltahat(q)=Dgamma_q^* d_F，伴随按样本配对定义。

局部参数方向 v 是一阶补偿方向，当且仅当
d Deltahat(q)[v]=<d_F,Dgamma_q[v]>_n=0。

它不要求 Dgamma_q[v]=0，也不要求颜色输出变化为零。
固定 a,d，若 partial_b Deltahat≠0，则隐函数定理给出局部补偿曲线 b=b(h)，满足
db/dh=-(partial_h Deltahat)/(partial_b Deltahat)。

这是风险差补偿，不是 b 与 h 在参数范数中的相互抵消。

## 4. 动力学与回填

令 M_q 为光滑正定参数迁移率矩阵；欧氏梯度流 M_q=I。考虑
qdot=-M_q grad J_lambda
     =-M_q grad Rhat-2lambda Deltahat M_q n_q。

于是精确有
Deltahat_dot=-n_q^T M_q grad Rhat-2lambda Deltahat n_q^T M_q n_q，
gamma_dot=Dgamma_q[qdot]。

这给出了源风险与惩罚对响应变化的精确分解，但一般不能只用 gamma 闭合方程，因为 B、M_q 和 Dgamma_q 未必由 gamma 唯一确定。

### 4.1 硬约束慢流

在 Deltahat=0、n_q≠0 处，保持约束的投影流为
qdot=-P_M grad Rhat，
P_M=M_q-M_q n_q(n_q^T M_q n_q)^{-1}n_q^T M_q。

因此
Deltahat_dot=0，
Rhat_dot=-(grad Rhat)^T P_M grad Rhat≤0。

推导：从 qdot=-M_q(grad Rhat+mu n_q) 与 n_q^T qdot=0 解出 mu。
该投影流本身精确；把有限 lambda 轨迹近似为它，需要紧性、约束正则性、法向吸引与初始层控制，本文不省略这些条件。

**回填不是任意切向运动的别名。**
定义颜色能量 S(q)=E_n(b+hZ)^2。只有
Sdot=-grad S^T P_M grad Rhat>0
时，才称该段为能量回填。
对目标风险 H(q)，有害回填还要求
Hdot=-grad H^T P_M grad Rhat>0。
两个符号不等价，需分别计算。

### 4.2 去色点排斥条件：从“允许”到“选择”

令 q_g=(a_g,d_g,0,0) 为经验去色模型的源风险驻点。
在 b=h=0 时 gamma=0，且 gamma 对形状参数的偏导为零，故 n_q 的形状块为零。

设
c=grad_(b,h) Rhat(q_g)，
k=grad_(b,h) Deltahat(q_g)。

**命题 2（非驻点判据）。**
若 k≠0 且 c不属于span{k}，则 q_g 不是 Deltahat=0 上的约束驻点，也不是任何有限 lambda 的 J_lambda 驻点。

证明：约束 KKT 的颜色块要求 c+mu k=0，与假设冲突。有限 lambda 时 Deltahat(q_g)=0，惩罚梯度为零，而 c≠0。□

欧氏约束流在该点的颜色速度为
-(I-kk^T/||k||^2)c。
若经验颜色设计 Gram G_z=E_n[(1,Z)^T(1,Z)] 正定，则从 q_g 出发，
S(q_t)=t^2 v_c^T G_z v_c+o(t^2)>0，
其中 v_c 为上述非零颜色速度。这证明局部颜色生成，不证明全程单调回填。

### 4.3 条件性的长期残留接口

**命题 3（局部稳定分支的有限惩罚延拓）。**
假设 q_* 满足 Deltahat(q_*)=0、
grad Rhat(q_*)+mu_* n_*=0、n_*≠0。
若拉格朗日 Hessian
H_L=Hess Rhat(q_*)+mu_* Hess Deltahat(q_*)
在 ker n_*^T 上正定，则对足够大的 lambda，存在邻近驻点
q_lambda=q_*+O(lambda^{-1})，
2lambda Deltahat(q_lambda)=mu_*+O(lambda^{-1})。
这些点是严格局部极小点；若 S(q_*)>0，则其颜色残留对足够大 lambda 有统一正下界。

证明概要：对方程
grad Rhat(q)+mu grad Deltahat(q)=0，
Deltahat(q)-mu/(2lambda)=0
在 1/lambda=0 应用隐函数定理。LICQ 与切向正定使 bordered KKT 矩阵可逆。Hess J_lambda=H_L(q,mu)+2lambda nn^T，法向强正曲率与切向正定给出局部正定性。连续性保留 S>0。□

局部稳定分支不自动证明 ERM warm-start 进入其吸引域，更不证明任意初始化高概率进入。该入口定理单列为待证义务。上述 M_q 正定流的结论不自动适用于带动量与历史状态的 Adam。

## 5. 从颜色残留到环境响应：可证明的桥

以下总体结论使用 Z 对称独立、b=d=0、a>0。
令
gamma(A,v)=ell(A-v)-ell(A+v)，A=Yu。

它关于 v 为奇函数，且
partial_v gamma(A,v)=sigmoid(v-A)+sigmoid(-A-v)>0。

当 |A|≤a 时，
partial_v gamma(A,v)≥2sigmoid(-a)。
对 A≥0，此导数关于 |v| 在零处取最小值；对 A≤0，它至少为1。因此
|gamma(A,v)|≥2sigmoid(-a)|v|。

**命题 4（源不可见与子群响应）。**
beta_0=0，
sign(beta_1)=sign(h)（h≠0），
|beta_1|≥2sigmoid(-a)|h|sqrt(2/pi)。

证明：Z 对称性与 gamma 关于 v 的奇性给出 beta_0=0。
sign(Z)gamma(A,hZ) 与 h 同号；逐点积分上述下界即可。□

这不是“颜色对所有环境变化都不可见”：它对统一翻转方向不可见，对已声明的子群翻转方向可见。若环境族只有统一 epsilon，则不应声称异质残留必然造成跨环境风险增量。

### 5.1 非对称训练终点的连续性桥

固定 a,h，比较 q=(a,d,b,h) 与 q0=(a,0,0,h)。
由于 |partial_v gamma|≤2、|partial_A gamma|≤1，
|beta_1(q)-beta_1(q0)|≤2|b|+|d|。

因此
|beta_1(q)|≥[2sigmoid(-a)|h|sqrt(2/pi)-2|b|-|d|]_+。

这是可检验的充分条件，可能很松。一般 S=b^2+h^2>0 不保证该下界为正，也不能从 S 单独推出 beta_1 下界。实际补偿分支需要控制残留的组成与方向。

## 6. 从响应到 BCE 与分类伤害

### 6.1 目标族内最坏 raw-risk 增量

固定 epsilon，预先声明 |tau|≤r≤min(epsilon,1-epsilon)。精确有
sup_tau [R_(epsilon,tau)(q)-R_(epsilon,0)(q)]
=r|beta_1(q)|。

在命题4条件下，下界为
2r sigmoid(-a)|h|sqrt(2/pi)。

若 rho_sc=h^2/(a^2+h^2)，则 |h|=a sqrt(rho_sc/(1-rho_sc))。
要得到沿轨迹统一的数值下界，还需对 a 的尺度作上下界控制；仅有 rho_sc 下界不够。

### 6.2 精确的干净分类错误桥

仍取 b=d=0、a>0、h≠0。以 D 为干净标签；分类边界等号在高斯 Z 下概率为零。
令 m=1-2eta，
A_flip=Pr(|hZ|>a)=2Phi(-a/|h|)。

逐样本上，|hZ|<a 时形状决定分类；|hZ|>a 时颜色决定分类。
对固定 z，后一区域内的干净错误率为
1/2 + m(p(z)-1/2) sign(hz)。

对称性给出
Err_clean_(epsilon,tau)
= A_flip/2 + m tau sign(h) A_flip。
故
sup_|tau|≤r [Err_clean_(epsilon,tau)-Err_clean_(epsilon,0)]
= r(1-2eta) A_flip。

相对于去色分类器 sign(aD)=D 的绝对目标伤害是
A_flip/2 + m tau sign(h) A_flip。
前者是迁移增量，后者包含共同基线伤害；两者不可混称。

在此切片中
A_flip=2Phi(-sqrt((1-rho_sc)/rho_sc))。
因此小残留可以有正但指数小的分类效应；本公式不能证明 CMNIST 量级的伤害。推广到非恒定形状 margin，必须保留 margin 与颜色响应的联合分布。

### 6.3 excess risk 的限制

定义 E_p(q)=R_p(q)-R_p^*，则
E_T(q)-E_S(q)
=[R_T(q)-R_S(q)]-[R_T^*-R_S^*]。

本文精确配对的是 raw risk。最优风险是对预测器取下确界形成的包络，不自动对环境状态仿射。本文不宣称消除了 Stage13R.2 的 optimum-risk 修正，也不把单个训练解的结论升级为整个预测器类的 transfer-measure 上界。

## 7. 待闭合的主定理

拟证明的主定理应在一个统一概率空间中包含：

1. 有限训练集及随机初始化；ERM warm-start 进入非退化补偿分支吸引域的概率。
2. 有限惩罚或指定优化器下的长期收敛/留存，给出 h 而非仅总颜色能量的下界。
3. 分支上的 a,d,b,h 控制，使第5.1节的响应下界非平凡，或推导更紧的直接 beta_1 下界。
4. 在预先声明的物理目标族上推出总体风险及分类伤害；目标族不按训练结果事后构造。
5. 经验源配对到总体响应的误差控制，处理 q 依赖训练集的问题。
6. 若接到 CMNIST，建立语义明确的形状子群与颜色干预状态，验证相同配对结构和动态选择；不能把 Z 子群的意义凭空移植。

候选结论形式：
以至少1-delta的概率，训练在长期窗口内满足
S(q_t)≥s_*，
|beta_1(q_t)|≥b_*，
sup_T∈U [R_T(q_t)-R_ref(q_t)]≥r b_*；
分类下界另外依赖 margin 条件。

其中第一条不自动蕴含第二条；第二条只对指定目标族给第三条。
本文没有完成以上联合高概率主定理。

## 8. 本轮验证与可复现入口

配套脚本：
[environment_response_shortcut_checks.py](experiments/environment_response_shortcut_checks.py)。

检查内容：

- 两环境配对样本的精确风险分解；
- 补偿曲线一阶切向与风险差导数；
- 二维总体环境族的积分与配对恒等式；
- 对称切片 beta_0=0、beta_1 的符号与定量下界；
- 干净分类错误的直接有限标签求和与闭式公式；
- 非对称响应的 Lipschitz 桥。

本轮运行结果：PASS。经验风险配对误差 1.11e-16；补偿切向的中心差分误差 4.12e-13；分类闭式公式与直接积分的误差 2.17e-18。
示例 a=log(3), h=0.4, eta=0.25, epsilon=0.15, tau=0.1：
beta_0=0，beta_1=0.2409000572，BCE 迁移增量=0.02409000572；
颜色敏感质量=0.00602292449，干净源错误=0.00301146224，干净目标错误=0.00331260847。

这些是解析恒等式与定理实例的数值检查，不是新训练、不是 checkpoint 重放，也不是 Lean 验证。
已发表相关工作的等价性与 novelty 本轮未审计。本文不新增算法，不改变分支既有的研究状态。
