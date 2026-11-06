# 原配对 logistic toy 的长期 shortcut 残留定理

日期：2026-11-06。状态：完整的条件性解析论证，附数据条件与轨迹的数值复核；未作 Lean 或区间算术认证。

本文补充 [环境—响应理论骨架](environment_response_shortcut_theory_skeleton.md) 中尚未写入的长期结论。证明不假设存在非零颜色的稳定分支，不假设初始化已在某个吸引域，也不依赖特定分支的快慢进入论证。

## 1. 结论与精确量词

固定一份满足第3节条件 A–C 的配对数据。存在只依赖该数据的常数 c_n>0，使得：

- 对每个有限 lambda≥0，目标 J_lambda 的**每一个驻点**均满足 S≥c_n；
- 对任意有限初值、任意有限 ERM warm-up 时间 T≥0，之后固定任意有限 lambda≥0 的欧氏梯度流满足
  $$\liminf_{t\to\infty}S(q(t))\ge c_n>0;$$
- 利用实解析梯度流的标准收敛性，还可加强为 q(t)→q_infty，且 S(q_infty)≥c_n。

常数 c_n 对初始化、T 和 lambda 一致；达到下界邻域的时间不声称对这些量一致。更精确地，对每条轨迹和每个 0<c<c_n，存在有限 t_0，使所有 t≥t_0 均有 S(q(t))≥c。

这是**固定数据、固定四参数模型、固定有限惩罚权重的连续梯度流定理**。不包括任意步长离散 GD、Adam、时变无界 lambda、非配对环境或 CMNIST/ReLU。

## 2. 数据与归一化

固定 n 个基础样本 (D_i,y_i,z_i)，D_i∈{−1,1}、y_i∈{0,1}、z_i∈R。
令 Y_i=2y_i−1，两个环境共享这些基础样本，仅颜色不同：
C_ei=Y_i(1−2F_ei)，F_ei∈{0,1}。

设
$$q=(a,d,b,h),\quad f_{ei}=aD_i+d+C_{ei}(b+hz_i),$$
$$R_e(q)=\frac1n\sum_i\{\log(1+e^{f_{ei}})-y_i f_{ei}\},$$
$$R=R_0+R_1,\quad \Delta=R_0-R_1,\quad J_\lambda=R+\lambda\Delta^2.$$

**本文使用风险之和 R，不使用平均风险。**
若使用原生目标 (R_0+R_1)/2+Lambda Var(R_0,R_1)，将整个目标乘以2后得到本文目标，lambda=Lambda/2，梯度流时间相应缩放。
骨架中的平均风险版本 Rbar+lambda_skel Delta² 对应本文 lambda=2lambda_skel。

颜色能量：
$$S(q)=\frac1n\sum_i(b+hz_i)^2=v^\top Gv,\quad v=(b,h)^\top,$$
$$G=\frac1n\sum_i(1,z_i)^\top(1,z_i).$$

它是反事实颜色 logit 差平方均值的四分之一；它不是分类翻转率，也不是 OOD 错误。

## 3. 可在训练前检查的数据条件

### A. 四单元标签凸包重叠

对每个 (D,C)∈{−1,1}²，汇集两个环境的记录。
该单元内 y=0 与 y=1 两组 z 均非空，且两组 z 凸包的交集具有严格正长度。

此条件是强制性/非弱可分性的充分条件，不是必要条件。

### B. 颜色翻转差的分组正性

记 I_s={i:D_i=s}，n_s=|I_s|>0，
$$p_s=\frac1{n_s}\sum_{i\in I_s}y_i\in(0,1),$$
$$k_{st}=\sum_{i:D_i=s,y_i=t}(F_{0i}-F_{1i})>0,\quad s\in\{-1,1\},\ t\in\{0,1\}.$$
定义
$$\kappa=\frac2n\sum_s\min(k_{s0},k_{s1})>0.$$

### C. 灰度最优点的梯度非共线性

令 u_s=d+sa，灰度最优点 q_g=(a_g,d_g,0,0) 满足
u_s(q_g)=logit(p_s)。

定义颜色块梯度
$$A_c=\nabla_{b,h}R(q_g),\quad B_c=\nabla_{b,h}\Delta(q_g),$$
$$r=A_c-\frac{A_c^\top B_c}{\|B_c\|^2}B_c,\quad \varrho=\|r\|>0.$$
另要求 G 正定。B 将保证 B_c 的 b 分量至少为 kappa，故上述投影有定义。

条件 C 是灰度点上可计算的导数条件，**不是假设最终残留**。

## 4. 引理：强制性与 ERM 唯一性

记设计向量 x_ei=(D_i,1,C_ei,C_ei z_i)。
若存在非零 q 使所有 signed margins Y_i x_ei^T q≥0，则在固定 (D,C) 单元内，仿射函数
aD+d+Cb+Chz
在 y=1 的凸包上非负、在 y=0 的凸包上非正。由 A，它在一个正长度区间为零。因此 h=0，且四个单元均满足 aD+d+Cb=0。这迫使 a=d=b=0，矛盾。

故任意单位向量 theta 至少有一个严格负 signed margin。连续函数
$$L(\theta)=\frac1n\sum_{e,i}[-Y_i x_{ei}^\top\theta]_+$$
在单位球上严格为正，紧性给出 l_*>0。由 log(1+exp(−m))≥[−m]_+，
$$R(q)\ge l_*\|q\|.$$
因此 R 强制有界。

同一论证排除设计矩阵的非零零空间。有限 q 下 logistic 权重 sigma(f)(1−sigma(f))>0，所以 Hess R(q) 正定。R 有唯一有限全局最优点。□

## 5. 关键定理：排除所有 lambda 的近去色驻点

**定理 1。** 条件 B–C 下存在 c_n>0，使任意 lambda≥0 与任意 grad J_lambda(q)=0 都满足 S(q)≥c_n。条件 A 仅用于之后的轨迹紧性与驻点存在，不参与本节排除论证。

### 第一步：去色附近乘子不能随 lambda 爆炸

记 t=max_i|b+hz_i|。对任意形状参数 u=(u_−,u_+)，精确有
$$\Delta_b(u,0)=\frac2n\sum_s[k_{s0}\sigma(u_s)+k_{s1}(1-\sigma(u_s))]\ge\kappa.$$

由 sigmoid 的 1/4-Lipschitz 性，
$$|\Delta_b(u,v)-\Delta_b(u,0)|\le t/2.$$
若 t≤kappa，则 Delta_b≥kappa/2。

在驻点令 mu=2lambda Delta(q)。b 方程为
R_b+mu Delta_b=0。
由于 |R_b|≤2，
$$|\mu|\le 4/\kappa=:M.$$

这里对所有形状参数统一成立；没有预先假设 a,d 有界。

### 第二步：近去色驻点的形状必须接近灰度最优

在 u_s 坐标下，
$$\left|R_{u_s}-2\frac{n_s}{n}(\sigma(u_s)-p_s)\right|\le\frac{n_s}{2n}t,$$
$$|\Delta_{u_s}|\le\frac{n_s}{2n}t.$$
后式使用配对结构：颜色为零时两个环境的该偏导完全相同。

驻点形状方程遂给出
$$|\sigma(u_s)-p_s|\le\frac{1+|\mu|}{4}t.$$

### 第三步：紧化到灰度点并得到矛盾

反设存在驻点序列 (q_j,lambda_j)，S(q_j)→0。
G 正定意味着 (b_j,h_j)→0；固定有限数据使 t_j→0。
第一步保证 mu_j 有界。第二步与 p_s∈(0,1) 保证
u_s(q_j)→logit(p_s)，即 q_j→q_g。

取 mu_j 的收敛子列，其颜色驻点方程极限为
$$A_c+\mu_\infty B_c=0,$$
与 varrho>0 矛盾。

所以所有 lambda 的所有驻点的 S 下确界严格为正。该论证不依赖预先选择一个局部稳定分支，也不要求驻点集合全局有界。□

### 5.1 一个可计算但非常保守的下界

为让存在性结论可核对，给出显式保守常数。记
$$p_*=\min_s\{p_s,1-p_s\},\quad g_*=\lambda_{\min}(G),\quad W_*=\max_i(1+z_i^2),$$
$$L=\frac12(3+\overline{z^2}),\quad M=4/\kappa,$$
$$H_u=\frac{(1+M)/4}{(p_*/2)(1-p_*/2)},\quad H_q=\sqrt{H_u^2+g_*^{-1}},$$
$$t_*=\min\left\{\kappa,\frac{2p_*}{1+M},
\frac{\varrho}{L(1+M)H_q}\right\},\qquad
c_{\rm explicit}=\frac{g_*}{W_*}t_*^2.$$

则可取 c_n=c_explicit>0。

证明：若驻点 t≤前两个阈值，第二步保证 sigma(u_s) 位于
[p_*/2,1−p_*/2]，故逆 logit 的导数界给
max_s|u_s−u_s(q_g)|≤H_u t。
形状参数变换满足
||(a−a_g,d−d_g)||²=(|u_+−u_+g|²+|u_−−u_−g|²)/2。
又 S≤t²，故 ||(b,h)||≤t/sqrt(g_*)，从而 ||q−q_g||≤H_q t。

Hess R 与 Hess Delta 的算子范数均不超过 L。
把颜色驻点方程投影到 B_c 的正交补，得
varrho≤L(1+M)||q−q_g||≤L(1+M)H_q t。
所以任意驻点都有 t≥t_*。最后
t²≤W_*||(b,h)||²≤W_* S/g_*，给出结论。□

这个下界为了全 lambda、全初始化的一致性牺牲了数值紧度。它不能代替实际终点的残留量级或分支的渐近统计尺度。

## 6. 从训练流程到长期残留

先在 [0,T] 使用 qdot=−grad R，之后使用 qdot=−grad J_lambda。
q(0) 有限，T 有限，lambda 固定有限。

**定理 2（无需选定吸引域）。**
条件 A–C 下，该分段流全局存在，且 liminf S(q(t))≥c_explicit。

证明：

1. ERM 段有 Rdot=−||grad R||²≤0。强制性使轨迹留在紧子水平集，因此无有限时间爆炸，q(T) 有限。
2. 切换时 J_lambda(q(T)) 可能增大，但仍有限。切换后
   Jdot=−||grad J||²≤0，且 J≥R；故轨迹仍留在紧集并全局存在。
3. J 有下界，因此积分 int_T^infty ||grad J(q(t))||² dt 有限。
   在该紧集上 grad J 与 Hess J 有界，
   d/dt ||grad J||²=−2 grad J^T Hess J grad J 有界。
   非负、一致连续且可积的函数必趋零，所以 grad J(q(t))→0。
4. 任意时间序列的收敛子列极限都是驻点。若 liminf S<c_explicit，
   可以选取相应序列并用紧性提取收敛子列；连续性给出一个 S<c_explicit 的驻点，
   与定理1矛盾。□

此证明已经得到长期残留，不依赖 Łojasiewicz、Fenichel 或隐函数定理。

**单点收敛补充。**
有限样本 logistic J_lambda 是实解析的。标准 Łojasiewicz 梯度不等式在任一聚点附近给出
|J−J_*|^theta≤C||grad J||，theta∈[1/2,1)。
结合能量下降得到局部剩余弧长界
int ||qdot||dt≤C'|J−J_*|^(1−theta)。
选取足够靠近聚点且能量差足够小的进入时刻，弧长界阻止轨迹离开该邻域，并使其为 Cauchy，故 q(t) 收敛到单一驻点。该标准补充不用于上面的 liminf 主结论。

ERM warm-up 不是本模型残留存在的必要条件；T=0 同样适用。不同初始化可能收敛不同驻点，不宣称唯一全局吸引子。lambda=0 的 ERM 终点也受定理覆盖。

## 7. 补偿、回填与表示骨架的连接

在 q_g，Delta=0 且形状梯度为零。等风险约束的欧氏切向流在颜色块上是
$$\dot v(0)=-r,\qquad \dot R(0)=-\|r\|^2,$$
因此
$$S(q(t))=t^2r^\top G r+o(t^2)>0.$$

这给出“完全去色点有源风险下降的补偿方向”的局部机制。
本定理进一步用乘子界把局部非共线性升级为**全体驻点的统一排除**，再用能量下降接到全体有限初值的长期轨迹。这是长期论证的实质。

在骨架的环境—响应表示中，Delta=<F_0−F_1,gamma_q>_n。
对任何有限 A，gamma(A,v)=ell(A−v)−ell(A+v) 对 v 严格递增且零点唯一为0。
所以 S>0 确实意味着经验响应向量 gamma_q 非零。
但本定理不保证该响应在某个指定目标方向上的投影具有统一下界。

它不证明 S 全程单调增加，不证明 h 单独有统一下界，不证明特定目标风险一定增加。分类伤害和目标方向响应仍需要骨架中的额外桥梁。

## 8. 数据条件为何并非终点假设

在独立生成模型中，eta=.25，epsilon_0=.2>epsilon_1=.1，
各 (D,C,y) 单元均有正概率，Z 独立标准高斯。
因此 A 的凸包重叠、B 的四组计数正性、G 正定随 n→∞ 以概率趋1。

条件 C 的非共线性也有直接解释。给定离散数据，定义
alpha_i=(p_Di−y_i)(C_0i+C_1i)/n，
beta_i=(p_Di−y_i)(C_0i−C_1i)/n。
则
A_c=(sum alpha_i,sum alpha_i z_i)，
B_c=(sum beta_i,sum beta_i z_i)。

其行列式是
sum_i[(sum alpha) beta_i−(sum beta) alpha_i] z_i。
若至少有一个样本 C_0i=C_1i，则该样本 beta_i=0、alpha_i≠0；
由 B，sum beta_i≥kappa>0，所以该线性式非退化。
独立连续高斯 Z 下其为零的概率是0。
出现至少一个同色样本的概率趋1。

这些陈述说明条件的可实现性与渐近典型性，不给出 n=512 的显式概率保证。固定 n 的结论首先是条件于实际数据检查的确定性定理。

## 9. 可复现检查、文件与证据等级

配套目录：[long_term_shortcut_toy/](experiments/long_term_shortcut_toy/)。

- model.py：原四参数 logistic 实现与原数据生成顺序。
- audit.py：重算 A–C、显式下界、导数界与留存终点的驻点残差。
- reference_full_flow.json：此前9条 Radau 轨迹的终点记录；不是本轮新训练。
- reproduce_flow.py：从有限随机初始化重跑 ERM 100 + 惩罚2000，lambda=10000。
- audit_results.json：本轮 audit.py 的结果。

运行：
python notes/framework_synthesis/experiments/long_term_shortcut_toy/audit.py
python notes/framework_synthesis/experiments/long_term_shortcut_toy/reproduce_flow.py

原数据 seed21–23 各3个初始化的既有终点经验颜色能量约为
0.02632698、0.11439440、0.06852200。
本轮独立重算导数与数据条件；有限时间记录只作数值佐证，所有时间、初始化和 lambda 的量词来自解析证明。

本轮条件检查和证明中的导数界检查全部通过。9个既有终点重算的最大梯度范数为 6.46e-13。

| seed | kappa | varrho | 显式统一 S 下界 | 既有轨迹终点 S |
|---|---:|---:|---:|---:|
|21|0.03125|0.05305|3.29e-14|0.02632698|
|22|0.02734375|0.11276|1.18e-13|0.11439440|
|23|0.04296875|0.09091|3.51e-13|0.06852200|

确定性显式下界很保守；不会把“严格正”伪装成已经证明了实验量级。
本文仅提交长期残留的固定数据主论证，不把旧稿中的分支进入高概率论证或 chi-square 渐近式一并提升为本次已复核结果。
