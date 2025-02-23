# RLHF

---

资料：

- [https://datawhalechina.github.io/easy-rl/#/](https://datawhalechina.github.io/easy-rl/#/)

# 时间线

- **2017**: Proximal Policy Optimization Algorithms
- **2023**: Direct Preference Optimization: Your Language Model is Secretly a Reward Model
- **2024**: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

# 基本概念

| **概念** | **英文** | **符号** | **解释** |
| --- | --- | --- | --- |
| 智能体 | Agent |  |  |
| 环境 | Environment |  |  |
| 状态 | State | $s_t$ |  |
| 状态空间 |  | $S$ | 所有可能的状态集合 |
| 动作 | Action | $a_t$ | 智能体在每个状态$s_t$下可以选择一个动作$a_t$，以期望获得最大的长期奖励。 |
| 动作空间 |  | $A$ | 所有可能的动作集合 |
| 奖励 | Reward | $r_t=R(s_t,a_t,s_{t+1})$
$R(s, a)=\mathbb{E}\left[r_{t+1} \mid s_t=s, a_t=a\right]$ | 注意，这里不同来源的下标可能有些差异，本质上是想说$s_t,a_t$并成功到达$s_{t+1}$时获得的奖励。 |
| 策略 | Policy | $a_t = \pi_\theta(s_t)\\
a_t \sim \pi_\theta(\cdot|s_t)$ | 在$s_t$下智能体应该选取下一步动作的准则，分为**确定性策略(deterministic)**和**随机性策略(stochastic)**。$\theta$是神经网络的参数。 |
| 运动轨迹 | Trajectory | $\tau = (s_t, a_t, s_{t+1}, a_{t+1},....)$ |  |
|  | Transition | $p(s_{t+1}|s_t,a_t)$ | 因为环境存在一定随机性，在$s_t$下采取动作$a_t$可能会到$s_{t+1}$ |
| 累积奖励 | Cumulative Reward | $R(\tau) = \sum_{t=0}^{T-1} r_t$ |  |
| 折扣奖励 | Discounted Cumulative Reward | $R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_t$ | 因为越远的奖励跟当前动作的关系越低 |
| **回报** | Return | $G_t=\sum_{k=0}^\infty \gamma^k r_{t+k}$ |  |
| **状态价值函数** | State-value function | $V_\pi(s)=E_{\pi}[G_t|S_t=s]$ | 从某个状态$s$开始，按照策略$\pi$能得到的累计奖励的期望值。 |
| **动作价值函数** | Action-value function | $Q_\pi(s,a) = E_\pi[G_t|S_t=s, A_t=a]$ | 从某个状态$s$且当前执行动作$a$，按照策略$\pi$能得到的累计奖励的期望值。 |
| **优势函数** | Advantage Function | $A^\theta\left(s_t, a_t\right)$ | 一般由Critic网络来估计。 |
| 基于策略的方法 | Policy-based Method |  | Policy Gradient
支持连续空间，但高方差、样本效率低、训练不稳定 |
| 基于价值的方法 | Value-based Method |  | Q-learning, DQN
样本效率高、训练稳定，较难处理连续动作。 |
| Actor-Critic | Actor-Critic |  | **Actor**: Policy-based Method，决定智能体如何行动，即$\pi_\theta$
**Critic**: 判断当前动作的好坏，即$Q_\pi(s,a)$或$V_\pi(s)$ |
| 同策略/在线 | On-Policy |  | 要学习的智能体和与环境交互的智能体是相同的 |
| 异策略/离线 | Off-Policy |  | 要学习的智能体和与环境交互的智能体不是相同的 |
| 时序差分 | Temporal Difference |  |  |

## Actor-Critic

### Actor (Policy Gradient, REINFORCE)

![$p_\theta$即上述的$\pi_\theta$，是我们要训练的Policy Model
$R(\tau^i)$: 第$i$个episode得到的总奖励
实际实现中，可以使用pytorch的梯度下降优化这个损失。
$Loss = -\frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} R\left(\tau^n\right) \log \pi_\theta\left(a_t^n \mid s_t^n\right)$](RLHF%20180e56f56d3c80ce9fbec0232dc4f1e8/image.png)

$p_\theta$即上述的$\pi_\theta$，是我们要训练的Policy Model
$R(\tau^i)$: 第$i$个episode得到的总奖励
实际实现中，可以使用pytorch的梯度下降优化这个损失。
$Loss = -\frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} R\left(\tau^n\right) \log \pi_\theta\left(a_t^n \mid s_t^n\right)$

- 证明：为什么是这个梯度（基于最大化Reward的假设）
  
    ![image.png](RLHF%20180e56f56d3c80ce9fbec0232dc4f1e8/image%201.png)
    

![image.png](RLHF%20180e56f56d3c80ce9fbec0232dc4f1e8/image%202.png)

<aside>
💡

改进方法1：**添加基线$b$，一般取$b = E[R(\tau)]$，实践中记录每个采样得到Reward的平均即可。**

</aside>
$$
\boxed{\nabla \bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n}\left(R\left(\tau^n\right)-b\right) \nabla \log p_\theta\left(a_t^n \mid s_t^n\right)}
$$

- 理由：因为我们是采样来更新的，如果Reward设置都是正的（例如0-5分），则被采样到的动作都会提升其对应的概率，但这不意味着没被采样到的动作就不好。
  
    ![截屏2025-02-22 下午3.57.43.png](RLHF%20180e56f56d3c80ce9fbec0232dc4f1e8/%E6%88%AA%E5%B1%8F2025-02-22_%E4%B8%8B%E5%8D%883.57.43.png)
    
- 通过基线，我们可以只对$R(\tau)>b$提升$\log p_\theta\left(a_t^n \mid s_t^n\right)$。

<aside>
💡

改进方法2：更精细化地控制每个更新的权重 → Actor-Critic

</aside>

$$
\boxed{\nabla \bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n}A_\phi(s_t, a_t) \nabla \log p_\theta\left(a_t^n \mid s_t^n\right)}
$$

- 一局赢了，则所有的动作都是优秀的动作吗？未必 → 更精细化地控制梯度的权重。
- Advantage Function: $A_\phi(s_t, a_t)$
    - 一种估计方法：$\sum_{t^{\prime}=t}^{T_n} \gamma^{t^{\prime}-t} r_{t^{\prime}}^n-b$，从当前时刻t之后所能获得的奖励减去一个基准值。

根据对于梯度更新前面系数的选择有不同的方法，比如**REINFORCE**选择的是

$$
\nabla \bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} G_t^n \nabla \log \pi_\theta\left(a_t^n \mid s_t^n\right)\\
G_t=\sum_{k=t+1}^T \gamma^{k-t-1} r_k
$$

![截屏2025-02-22 下午4.14.57.png](RLHF%20180e56f56d3c80ce9fbec0232dc4f1e8/%E6%88%AA%E5%B1%8F2025-02-22_%E4%B8%8B%E5%8D%884.14.57.png)

![截屏2025-02-22 下午4.17.02.png](RLHF%20180e56f56d3c80ce9fbec0232dc4f1e8/%E6%88%AA%E5%B1%8F2025-02-22_%E4%B8%8B%E5%8D%884.17.02.png)

![REINFORCE是online的](RLHF%20180e56f56d3c80ce9fbec0232dc4f1e8/%E6%88%AA%E5%B1%8F2025-02-22_%E4%B8%8B%E5%8D%884.20.03.png)

REINFORCE是online的

### Critic (Bellman Equation)

**Bellman Equation**

$$
\boxed{V(s)=\underbrace{R(s)}_{\text {即时奖励 }}+\underbrace{\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s\right) V\left(s^{\prime}\right)}_{\text {未来奖励的折扣总和 }}}
$$

- $p(s'|s)$: 从当前状态转移到某个状态的概率（包含策略以及环境转移2部分）
- 推导Bellman Equation，用到**全期望公式**：$\mathbb{E}[X]=\sum_i \mathbb{E}\left[X \mid A_i\right] p\left(A_i\right)$
  
    ![image.png](RLHF%20180e56f56d3c80ce9fbec0232dc4f1e8/image%203.png)
    
- 例子：如果有Transition可以直接有解析解
  
    $$
    \left(\begin{array}{c}V\left(s_1\right) \\V\left(s_2\right) \\\vdots \\V\left(s_N\right)\end{array}\right)=\left(\begin{array}{c}R\left(s_1\right) \\R\left(s_2\right) \\\vdots \\R\left(s_N\right)\end{array}\right)+\gamma\left(\begin{array}{cccc}p\left(s_1 \mid s_1\right) & p\left(s_2 \mid s_1\right) & \ldots & p\left(s_N \mid s_1\right) \\p\left(s_1 \mid s_2\right) & p\left(s_2 \mid s_2\right) & \ldots & p\left(s_N \mid s_2\right) \\\vdots & \vdots & \ddots & \vdots \\p\left(s_1 \mid s_N\right) & p\left(s_2 \mid s_N\right) & \ldots & p\left(s_N \mid s_N\right)\end{array}\right)\left(\begin{array}{c}V\left(s_1\right) \\V\left(s_2\right) \\\vdots \\V\left(s_N\right)\end{array}\right)
    $$
    
    $$
    \begin{aligned}\boldsymbol{V} & =\boldsymbol{R}+\gamma \boldsymbol{P} \boldsymbol{V} \\\boldsymbol{I} \boldsymbol{V} & =\boldsymbol{R}+\gamma \boldsymbol{P} \boldsymbol{V} \\(\boldsymbol{I}-\gamma \boldsymbol{P}) \boldsymbol{V} & =\boldsymbol{R} \\\boldsymbol{V} & =(\boldsymbol{I}-\gamma \boldsymbol{P})^{-1} \boldsymbol{R}\end{aligned}
    $$
    
- 例子：可以用**蒙特卡洛法直接估计某个状态的价值**
  
    ![截屏2025-02-23 上午11.11.00.png](RLHF%20180e56f56d3c80ce9fbec0232dc4f1e8/%E6%88%AA%E5%B1%8F2025-02-23_%E4%B8%8A%E5%8D%8811.11.00.png)
    
- 例子：**动态规划** (Bootstrapping + Bellman update)
  
    ![截屏2025-02-23 上午11.12.13.png](RLHF%20180e56f56d3c80ce9fbec0232dc4f1e8/%E6%88%AA%E5%B1%8F2025-02-23_%E4%B8%8A%E5%8D%8811.12.13.png)
    

**Q函数和V函数的关系 （Q函数的Bellman Equation）**

# PPO

> 论文：[Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347)
OpenAI
> 

## 预备知识

### Policy Gradient Methods

$$
\boxed{\hat{g}_t=\hat{\mathbb{E}}_t\left[\nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right) \hat{A}_t\right]}
$$

- $\hat{g}_t$: policy gradient estimator (t时刻)
- $\pi_\theta$: 待优化的policy, $\theta$是对应网络的参数
- $\hat{A}_t$: estimator of advantage function
- $\mathbb{E}_t[\ldots]$: 一个batch的平均值（empirical average）
- 证明：为何这个梯度方向是我们希望的更新方向。
  
    ![image.png](RLHF%20180e56f56d3c80ce9fbec0232dc4f1e8/image%201.png)
    

**具体实践的时候可以直接优化下面这个损失(我们希望其最大化)**

$$
\boxed{L^{P G}(\theta)=\hat{\mathbb{E}}_t\left[\log \pi_\theta\left(a_t \mid s_t\right) \hat{A}_t\right]}
$$

- 论文里说明了如果直接用同一批采样数据（trajectory）来多次更新可能会导致破坏性的很大的更新。
- **因为这个函数的梯度就是上式的$\hat{g}$，换言之这个函数取到极值的梯度方向就是我们上面希望优化的梯度方向。**
  
    Implementations that use automatic differentiation software work by constructing an objective function whose gradient is the policy gradient estimator
    
- 参考：[https://zhuanlan.zhihu.com/p/31278940](https://zhuanlan.zhihu.com/p/31278940)

### Importance Sampling

### Importance Sampling

- 因为加了Importance Sampling才需要控制两个策略分布不能差太远

## 具体实践

# DPO

> 论文：[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290)
> 

![image.png](RLHF%20180e56f56d3c80ce9fbec0232dc4f1e8/image%204.png)

- 原始的RLHF通常包含3个环节：
    1. **Supervised fine-tuning (SFT)** → $\pi^{\mathrm{SFT}}$: 在高质量的数据上训练（对话、摘要等）
    2. **Preference Sampling & Reward Learning**: 
        1. 给定prompt $x$，采样模型输出：$\left(y_1, y_2\right) \sim \pi^{\mathrm{SFT}}(y \mid x)$
        2. 人工挑选哪个输出更好：$y_w \succ y_l \mid x$，$y_w$是两者之间更好的那个。
        3. 
    3. **RL optimization**: 
- 对PPO的改进，去掉RM训练和RL环节。只需要加载一个推理模型和一个训练模型，直接在偏好数据上进行训练即可。

## 预备知识

### KL散度

> 衡量2个概率分布之间的差异，值越小，两者之间的差异越小。
> 

$$
D_{\mathrm{KL}}(P \| Q)=\int P(x) \log \frac{P(x)}{Q(x)} d x
$$

- $P(x)$：目标分布
- $Q(x)$：近似分布，KL散度衡量的是

性质：

1. **非负性**：$D_{\mathrm{KL}}(P \| Q) \geq 0$，当且仅当$P(x)=Q(x)$时取到最小值0
2. **非对称性**：$D_{\mathrm{KL}}(P \| Q) \neq D_{\mathrm{KL}}(Q \| P)$

- 非负性证明：
  
    ![image.png](RLHF%20180e56f56d3c80ce9fbec0232dc4f1e8/image%205.png)
    

### Bradley-Terry 模型

> 对于胜率进行建模，论文里也提到还有其他模型 (Plackett-Luce ranking models)
> 
- 对象$O_i$的强度：$\lambda_i$
- $O_i$对$O_j$的胜率

$$
P(i>j)=\frac{\lambda_i}{\lambda_i+\lambda_j}
$$

- 可以用极大似然法来估计对应的强度参数$\lambda_i$的值：

$$
L(\boldsymbol{\lambda})=\prod_{(i, j) \in \text { pairs }}\left(\frac{\lambda_i}{\lambda_i+\lambda_j}\right)^{x_{i j}}\left(\frac{\lambda_j}{\lambda_i+\lambda_j}\right)^{1-x_{i j}}\\

x_{ij} = 1, 如果i赢\\
x_{ij} = 0, 如果i输
$$

- $L(\boldsymbol{\lambda})$：当前参数下，观察到所有两两比赛(pairs)的胜率。所以我们希望能够通过调整$\lambda_i$来使得估计得到的胜率最大化。

个人感觉：$\lambda_i \geq 0$，否则胜率有可能会超过1。

### DPO

1. **Reward Model**优化目标推导

$$
\boxed{P\left(y_w \succ y_l \mid x\right)=\sigma\left(r\left(x, y_w\right)-r\left(x, y_l\right)\right) \quad\quad \text{(1)}}
$$

- $y_w$: 模型输出的**更优**的答案
$y_l$: 模型输出的**次优**的答案
- $\sigma(x)$: sigmoid函数，$\sigma(x)=\frac{1}{1+e^{-x}}$
- 推导过程：用到2个假设(Bradley-Terry模型 & $\lambda_i = e^{r(x, y_i)}$)
  
    $$
    \begin{aligned}
    P\left(y_w \succ y_l \mid x\right)&=\frac{e^{r\left(x, y_w\right)}}{e^{r\left(x, y_w\right)}+e^{r\left(x, y_l\right)}}\\
    &=\frac{1}{1+e^{r\left(x, y_l\right)-r\left(x, y_w\right)}}\\
    &=\sigma\left(r\left(x, y_w\right)-r\left(x, y_l\right)\right)
    \end{aligned}
    $$
    

由此，对于一个偏好数据集

$$
\mathcal{D}=\left\{x^{(i)}, y_\omega^{(i)}, y_l^{(i)}\right\}_{i=1}^N
$$

我们可以根据极大似然估计构建一个reward model: $r_\phi(x, y)$，其目标是使得在这个数据上偏好概率的乘积最大（参考BT模型一节）。

$$
\mathcal{L}(r_\phi, \mathcal{D})=\prod_{\left(x, y_w, y_l\right) \sim \mathcal{D}} \sigma\left(r_{\phi} \left(x, y_w\right)-r_{\phi}\left(x, y_l\right)\right)
$$

如果我们要得到一个最小化的loss，可以写作：

$$
\boxed{\mathcal{L}_{\mathcal{R}}\left(r_\phi, \mathcal{D}\right)=-\mathbb{E}_{\left(x, y_\omega, y_l\right) \sim \mathcal{D}} \log \sigma\left(r_\phi\left(x, y_w\right)-r_\phi\left(x, y_l\right)\right)\quad\quad \text{(2)}}
$$

- 一般reward model $r_\phi(x, y)$从SFT结束后的模型$\pi^{\mathrm{SFT}}(y \mid x)$初始化，并在最后的transformer layer上增加一层linear层来得到一个reward的值。

1. DPO损失函数推导

先从之前的做法开始， 假设我们已经有了一个reward model $r_\phi(x,y)$，我们可以用如下的损失来优化我们的模型（Policy model）$\pi_\theta(y \mid x)$。

$$
\boxed{\max _{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y \mid x)}\left[r_\phi(x, y)\right]-\beta \mathbb{D}_{\mathrm{KL}}\left[\pi_\theta(y \mid x) \| \pi_{\mathrm{ref}}(y \mid x)\right] \quad\quad\text{(3)}}
$$

- $\pi_\theta(y \mid x)$：当前待优化模型(llm)，也被称为Policy Model
- $\pi_{\mathrm{ref}}(y \mid x)$：参考的策略，一般就是SFT之后的模型 $\pi^{\mathrm{SFT}}$
- $\beta$：超参数，我们希望训练完的模型和训练之前模型的概率分布比较接近。
    - 论文里解释：防止模型偏离奖励模型准确的分布太远，以及保持生成多样性和防止模式崩溃为单个高奖励答案。
      
        The added constraint is important, as it prevents the model from deviating too far from the distribution on which the reward model is accurate, as well as maintaining the generation diversity and preventing mode-collapse to single high-reward answers. 
        
    - 因为之前的$r_\phi(x,y)$是根据$\pi_{\mathrm{ref}}(y \mid x)$（即$\pi^{\mathrm{SFT}}$）训练得到的，如果换个policy，对应的奖励函数可能差距比较大。所以为了能复用这个奖励函数，得加上这个KL限制。
- **之前常见做法**：
    - $r(x, y)=r_\phi(x, y)-\beta\left(\log \pi_\theta(y \mid x)-\log \pi_{\mathrm{ref}}(y \mid x)\right)$
    - 用PPO最大化(3)式，从而得到最后的policy model $\pi_\theta$

---

那么DPO是怎么干掉reward model $r_\phi(x,y)$的呢？分2步

**第一步：证明（3）式要求的最优策略是有显式解的**。

$$
\boxed{\pi(y \mid x)=\pi^*(y \mid x)=\frac{1}{Z(x)} \pi_{r e f}(y \mid x) e^{\frac{1}{\beta} r_\phi(x, y)} \quad\quad\text{(4)}}
$$

$$
Z(x)=\sum_y \pi_{r e f}(y \mid x) e^{\frac{1}{\beta} r_\phi(x, y)}
$$

- 证明：核心是凑出一个概率分布，再利用KL散度非负性得到最优的策略。
  
    ![image.png](RLHF%20180e56f56d3c80ce9fbec0232dc4f1e8/image%206.png)
    

**第二步：直接计算每个样本的胜率，回归到BT模型，从而得到DPO损失。**

$$
\boxed{\mathcal{L}_{D P O}\left(\pi_\theta ; \pi_{\mathrm{ref}}\right)=-\mathbb{E}_{\left(x, y_w, y_l\right) \sim D}\left[\log \sigma\left(\beta \log \frac{\pi_\theta\left(y_w \mid x\right)}{\pi_{\mathrm{ref}}\left(y_w \mid x\right)}-\beta \log \frac{\pi_\theta\left(y_l \mid x\right)}{\pi_{\mathrm{ref}}\left(y_l \mid x\right)}\right)\right]}
$$

- $\pi_\theta(y \mid x)$: 当前待优化的模型。

- 证明：核心是既然reward和policy存在一定的关系，与训练一个reward model不如直接训练policy model $\pi_\theta(y|x)$
  
    ![image.png](RLHF%20180e56f56d3c80ce9fbec0232dc4f1e8/image%207.png)
    

### DPO 梯度的解释

![截屏2025-02-16 下午2.54.16.png](RLHF%20180e56f56d3c80ce9fbec0232dc4f1e8/%E6%88%AA%E5%B1%8F2025-02-16_%E4%B8%8B%E5%8D%882.54.16.png)

$$
\hat{r}_\theta(x, y)=\beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}
$$

- 直观理解是：更新会增大prefered答案的概率，减小不喜欢的答案的概率，以及当reward估计有误的时候会有较大更新。
- 梯度的推导：
  
    ![image.png](RLHF%20180e56f56d3c80ce9fbec0232dc4f1e8/image%208.png)
    

### DPO 实现步骤

Case1: 如果从0开始（没有对应的偏好数据）

1. $y_1, y_2 \sim \pi_{\mathrm{ref}}(\cdot \mid x)$ 采样，人工标注得到偏好数据$\mathcal{D}=\left\{x^{(i)}, y_w^{(i)}, y_l^{(i)}\right\}_{i=1}^N$。此时$\pi_{ref}$就是待优化的模型。
2. 优化DPO损失。

Case2: **复用开源的DPO数据集**

1. **如果开源的数据集有公开采样的模型权重，则用对应的模型权重初始化**$\pi_{\mathrm{ref}}=\pi^{\mathrm{SFT}}$
2. 通过训练待优化的模型来逼近采样用的模型$\pi_{\mathrm{ref}}$
   
    $$
    \pi_{\mathrm{ref}}=\arg \max _\pi \mathbb{E}_{x, y_w \sim \mathcal{D}}\left[\log \pi\left(y_w \mid x\right)\right]
    $$
    
3. 利用$\pi_{ref}$进行DPO训练。

## 具体实践

- 数据格式：
    - prompt: 上下文输入
    - chosen: 更好的回答
    - rejected: 更差的回答
- 例子：[jingyaogong/minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset?row=33)
  
    ```python
    {
    	"chosen": [
    	{"content": "How many moles of HBr are required to react with 2 moles of C2H6 to form 2 moles of C2H5Br along with 2 moles of H2?", "role": "user"}, 
    	{"content": "To determine the number of moles of HBr required to react with 2 moles of C2H6 (ethane) to form 2 moles of C2H5Br (ethyl bromide) and 2 moles of H2, we need to write the balanced chemical equation for the reaction.\n\nThe reaction is a halogenation reaction, specifically the reaction of ethane with hydrogen bromide (HBr) to form ethyl bromide and hydrogen gas. The balanced chemical equation for this reaction is:\n\n\\[ \\text{C}_2\\text{H}_6 + \\text{HBr} \\rightarrow \\text{C}_2\\text{H}_5\\text{Br} + \\text{H}_2 \\]\n\nHowever, this equation is not balanced as written. To balance it, we need to ensure that the number of atoms of each element on the reactant side is equal to the number of atoms of that element on the product side. The balanced equation is:\n\n\\[ \\text{C}_2\\text{H}_6 + \\text{HBr} \\rightarrow \\text{C}_2\\text{H}_5\\text{Br} + \\text{H}_2 \\]\n\nUpon closer inspection, we see that the equation is already balanced as written, with 1 mole of C2H6 reacting with 1 mole of HBr to produce 1 mole of C2H5Br and 1 mole of H2.\n\nGiven that you have 2 moles of C2H6, and the stoichiometry of the reaction is 1:1 for C2H6 to HBr, you would need 2 moles of HBr to react with 2 moles of C2H6 to form 2 moles of C2H5Br and 2 moles of H2.\n\nTherefore, the number of moles of HBr required is **2 moles**.", "role": "assistant"}], 
    	"rejected": [
    	{"content": "How many moles of HBr are required to react with 2 moles of C2H6 to form 2 moles of C2H5Br along with 2 moles of H2?", "role": "user"}, 
    	{"content": "To answer this question, we need to write down the chemical equation representing the reaction between hydrogen bromide (HBr) and ethane (C2H6):\n\nC2H6(g) + 2HBr(g) \u2192 C2H5Br(l) + H2(g)\n\nNow, we can set up a mole ratio based on the coefficients in front of each substance in the balanced equation:\n\nMoles of HBr : Moles of C2H6 :: 2 : 1\n\nWe also know that there are two moles of C2H6 present in our starting mixture, so we can substitute this value into the above mole ratio to determine the amount of HBr needed:\n\n(2 mol C2H6)/(1 mol C2H6) x 2 mol HBr = 4 mol HBr\n\nSo, four moles of HBr would be required to completely react with two moles of C2H6 according to the given stoichiometric relationship.", "role": "assistant"}]}
    ```
    
- 训练代码：来自论文
  
    ```python
    import torch.nn.functional as F 
    
    def dpo_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta): 
    	""" 
    	pi_logps: policy logprobs, shape (B,) 
    	ref_logps: reference model logprobs, shape (B,) 
    	yw_idxs: preferred completion indices in [0, B-1], shape (T,) 
    	yl_idxs: dispreferred completion indices in [0, B-1], shape (T,) 
    	beta: temperature controlling strength of KL penalty 
    	
    	Each pair of (yw_idxs[i], yl_idxs[i]) represents the indices of a single preference pair. 
    	""" 
    	pi_yw_logps, pi_yl_logps = pi_logps[yw_idxs], pi_logps[yl_idxs] 
    	ref_yw_logps, ref_yl_logps = ref_logps[yw_idxs], ref_logps[yl_idxs] 
    	
    	pi_logratios = pi_yw_logps - pi_yl_logps 
    	ref_logratios = ref_yw_logps - ref_yl_logps 
    	losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios)) 
    	rewards = beta * (pi_logps - ref_logps).detach() 
    	
    	return losses, rewards
    ```
    
    - $\beta = 0.1, BS=65, RMSProp, lr=1e-6$
- 训练代码：trl库
  
    ```python
    
    ```
    
- 训练代码：手搓

## 参考

- [https://blog.csdn.net/qq_36803941/article/details/142251643](https://blog.csdn.net/qq_36803941/article/details/142251643)
- [https://zhuanlan.zhihu.com/p/644911957](https://zhuanlan.zhihu.com/p/644911957)

# GRPO

> 论文：[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)