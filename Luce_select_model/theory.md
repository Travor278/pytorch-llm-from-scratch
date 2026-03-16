# Softmax 背后的选择理论：从 Luce 公理到嵌套 Logit

>你每天都在用 Softmax，但它并不是凭空设计的。它是 1959 年一个心理学公理的直接推论——而这个公理，有一个让它几乎被遗忘了二十年的致命缺陷。

---

## 一切从"效用"开始

离散选择理论的出发点极其朴素：人面对一组备选项 $C = \{1, 2, \ldots, J\}$，选最好的那个。

问题在于"最好"本身是随机的。观测者无法看到所有影响决策的因素，所以用随机变量来建模：

$$U_j = V_j + \varepsilon_j$$

$V_j$ 是可以观测和建模的确定性效用，$\varepsilon_j$ 是误差项，捕捉不可观测的个人偏好。决策者选择使 $U_j$ 最大的那个备选项。

**不同的 $\varepsilon$ 分布假设，对应不同的选择模型。** 这是整个理论框架的核心。

---

## Luce 公理：一个优雅的出发点

1959 年，心理学家 Robert Duncan Luce 提出了一个关于选择概率的公理：

> 若 $P(i \mid C)$ 是从集合 $C$ 中选择备选项 $i$ 的概率，则对任意子集 $S \subseteq C$：
> $$P(i \mid C) = P(i \mid S) \cdot P(S \mid C)$$

这条公理说的是：选 $i$ 这件事，可以分解为"先选到子集 $S$，再从 $S$ 中选 $i$"，并且这两步互相独立。直觉上很自然。

从这个公理出发，可以严格推导出唯一的概率形式：每个备选项具有正的吸引力 $s(j) > 0$，选择概率为：

$$P(i \mid C) = \frac{s(i)}{\sum_{j \in C} s(j)}$$

令 $s(j) = e^{V_j}$，就得到**多项 Logit 模型（Multinomial Logit, MNL）**：

$$P(i \mid C) = \frac{e^{V_i}}{\sum_{j \in C} e^{V_j}}$$

这就是 **Softmax**。它不是设计出来的，是从一条心理学公理推导出来的。

### 概率论的背书

从随机效用角度，若各 $\varepsilon_j$ 独立同分布于 Gumbel(0, 1)（I 型极值分布），则：

$$P(i \mid C) = P\!\left(V_i + \varepsilon_i > V_j + \varepsilon_j,\ \forall j \neq i\right) = \frac{e^{V_i}}{\sum_j e^{V_j}}$$

Gumbel 分布之所以会出现，是因为它是最大值运算的稳定分布——选择本质上是求最大值，而最大值的极限分布正是 Gumbel。

---

## IIA：既是优点，也是原罪

Luce 模型有一条内置性质，叫做**无关备选项独立性（Independence from Irrelevant Alternatives, IIA）**：

$$\frac{P(i \mid C)}{P(j \mid C)} = \frac{e^{V_i}}{e^{V_j}}$$

任意两个备选项的概率之比，只取决于它们自身的效用，与集合里有什么其他备选项完全无关。

IIA 让估计变得极其方便：想知道苹果和香蕉的偏好比，只需要苹果-香蕉的选择数据，不需要知道有没有橘子在场。McFadden 凭借这个模型获得了 2000 年诺贝尔经济学奖。

**但 IIA 同时意味着一件荒谬的事。**

### 红色巴士 / 蓝色巴士悖论

想象通勤者只有两种出行方式：开车（Car）和坐红色巴士（Red Bus），各占 50% 概率，比率 1:1。

现在交通部门新增了一条蓝色巴士（Blue Bus）路线，路线完全相同，只是车身颜色不同。

IIA 的预测是：
$$P(\text{Car}) = P(\text{Red Bus}) = P(\text{Blue Bus}) = \frac{1}{3}$$

但任何有常识的人都知道，蓝色巴士只会从红色巴士那里分走乘客：
$$P(\text{Car}) = 0.5,\quad P(\text{Red Bus}) = P(\text{Blue Bus}) = 0.25$$

**问题的根源**在于：Luce 模型假设所有 $\varepsilon_j$ 相互独立，但红色巴士和蓝色巴士共享着大量未被观测的共同因素（"它们都是巴士"）。这种相关性在独立 Gumbel 假设下完全消失了。

---

## 嵌套 Logit：在树上做两次 Softmax

修复思路很直接：**把相似的备选项归为一组（巢，Nest），让同巢内的备选项共享误差相关结构。**

选择过程变成两步：先选巢，再在巢内选具体备选项。

```bash
所有备选项
├── 巢 B（巴士类）
│   ├── 红色巴士
│   └── 蓝色巴士
└── 巢 C（非巴士）
    └── 小汽车
```

### 数学推导

设备选项 $i$ 属于巢 $k$，效用为 $V_{ik}$，巢 $k$ 的尺度参数为 $\lambda_k \in (0, 1]$。

**第一步——巢内条件概率（下层 Logit）：**

$$P(i \mid k) = \frac{e^{V_{ik}/\lambda_k}}{\displaystyle\sum_{j \in k} e^{V_{jk}/\lambda_k}}$$

**第二步——包含价值（Inclusive Value）：**

$$\text{IV}_k = \lambda_k \ln \sum_{j \in k} e^{V_{jk}/\lambda_k}$$

包含价值是一个 LogSumExp，代表巢 $k$ 内的"集体吸引力"——巢内备选项越好、越多，IV 越高。

**第三步——巢间概率（上层 Logit）：**

$$P(k) = \frac{e^{\alpha_k + \text{IV}_k}}{\displaystyle\sum_{m} e^{\alpha_m + \text{IV}_m}}$$

**联合概率：**

$$\boxed{P(i) = P(i \mid k) \cdot P(k)}$$

### 尺度参数 $\lambda_k$ 的意义

$\lambda_k$ 控制着巢内备选项的误差相关程度。同巢内任意两个备选项 $i, j$ 的误差相关系数为：

$$\text{Corr}(\varepsilon_i, \varepsilon_j) = 1 - \lambda_k^2$$

| $\lambda_k$ | 误差相关性 | 含义 |
| :---: | :---: | :--- |
| 1 | 0 | 巢内无相关，退化为普通 MNL（Luce 模型） |
| 0.7 | 0.51 | 同巢备选项有中等相关 |
| → 0 | → 1 | 完全相关，巢内备选项本质上是一个 |

回到红色巴士例子：把两辆巴士归入同一巢，$\lambda_{\text{Bus}} < 1$ 捕捉了它们的相关性，蓝色巴士进入后只影响巢内概率，不再"抢走"小汽车的份额。

### GEV 框架的统一视角

嵌套 Logit 是**广义极值（Generalized Extreme Value, GEV）**模型家族的成员。GEV 用一个生成函数 $G$ 统一描述所有允许误差相关的离散选择模型：

$$P(i) = \frac{e^{V_i} \cdot \partial G / \partial y_i}{G(e^{V_1}, \ldots, e^{V_J})}$$

嵌套 Logit 对应的生成函数为：

$$G(\mathbf{y}) = \sum_{k} \left(\sum_{j \in k} y_j^{1/\lambda_k}\right)^{\lambda_k}$$

IIA 成立当且仅当 $G$ 是各 $y_j$ 的线性函数，即 $\lambda_k = 1$ 时退化为 Luce 模型。

---

## 更进一步的泛化

嵌套 Logit 要求每个备选项属于唯一的巢，这仍然是一个限制。

### 交叉嵌套 Logit（Cross-Nested Logit）

允许一个备选项同时属于多个巢，用隶属度 $\alpha_{ik} \geq 0$（$\sum_k \alpha_{ik} = 1$）表示归属程度：

$$G(\mathbf{y}) = \sum_k \left(\sum_j \alpha_{jk} \cdot y_j^{1/\lambda_k}\right)^{\lambda_k}$$

例如"连衣裙"可以 60% 归属"上装"巢，40% 归属"正装"巢，比强制二选一更合理。

### 混合 Logit

允许参数 $\beta$ 在个体间随机变化，通过对 $\beta$ 的分布积分：

$$P(i) = \int \frac{e^{\beta' x_i}}{\sum_j e^{\beta' x_j}}\ f(\beta \mid \theta)\, d\beta$$

在理论上可以近似任意的 RUM 模型，完全克服 IIA，但需要蒙特卡洛数值积分。

### 多项 Probit

直接假设 $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \Sigma)$，允许完整的协方差结构。理论上最灵活，但需要高维正态分布积分，计算代价随备选项数量急剧上升。

### 一个信息论的视角

Matějka & McKay（2015）的理性忽视模型给了 Softmax 另一个解释：

$$\max_P \left[\mathbb{E}[U] - \mu \cdot \mathrm{KL}(P \| Q)\right]$$

在信息处理成本（KL 散度）约束下最大化期望效用，最优解恰好是以先验 $Q$ 为基础的 Luce 模型。**Softmax 是信息受限决策者的最优策略。**

将 KL 散度替换为 $\alpha$-Tsallis 散度，得到 $\alpha$-entmax：

- $\alpha = 1$：Softmax（Luce 模型）
- $\alpha = 2$：Sparsemax，输出稀疏概率分布

---

## 与深度学习的对应

标准分类网络的输出层就是 Luce 模型，IIA 的所有问题它都有。

嵌套 Logit 在深度学习里对应的是**层次 Softmax（Hierarchical Softmax）**，最早由 Morin & Bengio（2005）用于解决大词表语言模型的计算瓶颈：将词汇表组织成二叉树，沿路径做二元分类，复杂度从 $O(V)$ 降至 $O(\log V)$，这正是嵌套 Logit 在 $\lambda=1$、二分巢情形下的特例。

更一般地，嵌套结构给深度学习输出层带来了三点好处：

1. **建模类别相关性**：真实世界的类别存在天然的语义层次（ImageNet 的 WordNet 树），嵌套结构与数据本质契合；
2. **更好的概率校准**：IIA 会对相似类别给出不合理的均等概率，嵌套结构让同巢类别在竞争时互相"谦让"；
3. **包含价值即注意力**：$\text{IV}_k = \lambda_k \cdot \log \sum_{j \in k} e^{V_j / \lambda_k}$ 本质上是 LogSumExp，和 Transformer 注意力机制里的归一化因子形式相同。

| 离散选择理论 | 深度学习对应 |
| :--- | :--- |
| 备选项效用 $V_j$ | 分类头的 logit |
| MNL（Luce 模型） | 标准 Softmax 输出层 |
| 嵌套 Logit | 层次 Softmax / 树形分类头 |
| 包含价值 IV | LogSumExp 聚合 |
| 混合 Logit | 变分推断 / 集成 |
| 尺度参数 $\lambda_k$ | 可学习的层次解耦系数 |
| 理性忽视（KL 约束） | Temperature Scaling / 熵正则化 |

---

## 延伸阅读

- Luce, R.D. (1959). *Individual Choice Behavior: A Theoretical Analysis*
- McFadden, D. (1978). Modeling the choice of residential location. *Transportation Research Record*
- Train, K. (2009). *Discrete Choice Methods with Simulation*
- Matějka & McKay (2015). Rational Inattention to Discrete Choices. *AER*
- Morin & Bengio (2005). Hierarchical Probabilistic Neural Network Language Model. *AISTATS*
