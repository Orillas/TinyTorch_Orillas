这篇由 [Jeremy Bernstein](https://jeremybernste.in/writing/deriving-muon)（深度学习优化领域的知名学者、Muon 优化器的核心理论贡献者之一）撰写的博客文章 **"Deriving Muon"**，系统地阐述了 Muon 优化器背后的严谨数学推导过程。

文章的核心主旨是：**Muon 并非像 Adam 那样基于工程直觉或启发式（Heuristic）规则构建，而是基于“度量深度学习（Metrized Deep Learning）”的第一性原理，通过严格的数学推导得出的。**

文章将 Muon 的推导过程拆解为逻辑严密的四个步骤，以下为您详细客观地进行解析：

### 背景设定
推导的对象是神经网络中的**隐藏层（Hidden Layers）**，即执行 $y = Wx$ 线性变换的层。文章假设输入 $x$ 和输出 $y$ 都是“稠密（Dense）”的激活向量（即向量元素的平均绝对值或 RMS 均方根值约为 1）。

---

### Step 1: Metrizing the linear layer（度量化线性层）
**核心概念**：为输入、输出和权重引入“尺度（Size）”的度量标准。
*   在优化过程中，我们需要控制变量的大小。对于输入向量 $x$，文章采用了 **RMS 范数（均方根范数）**，并假设 $\|x\|_{\text{RMS}} \le 1$。
*   对于权重矩阵 $W$ 以及它的更新量 $\Delta W$，我们需要一种能将输入尺度映射到输出尺度的度量方式，即**算子范数（Operator Norm）**。在这里，具体表现为 **RMS-to-RMS 算子范数**（本质上是缩放后的谱范数）。

### Step 2: Perturbing the linear layer（扰动线性层）
**核心概念**：建立权重更新（$\Delta W$）与输出变化（$\Delta y$）之间的定量边界。
*   神经网络通过更新权重矩阵来训练：$W_{new} = W + \Delta W$。
*   权重的更新会直接导致层输出的改变：$\Delta y = (W + \Delta W)x - Wx = \Delta W x$。
*   利用第一步定义的算子范数，可以得出一个严格的边界：**输出的 RMS 变化量，受限于权重更新的 RMS-to-RMS 算子范数**。即我们要控制输出不发生剧烈震荡，就必须控制 $\|\Delta W\|_{\text{RMS} \to \text{RMS}}$。

### Step 3: Dualizing the gradient（对偶化梯度）—— 理论核心
**核心概念**：在约束条件下寻找使损失下降最快的更新方向。
文章将寻找最优 $\Delta W$ 转化为一个带约束的优化问题：
1.  **目标**：最大化损失函数的线性下降量（即最小化内积 $\langle \nabla_W L, \Delta W \rangle$）。
2.  **约束**：权重更新的算子范数不能超过某个步长 $\eta$（即 $\|\Delta W\|_{\text{RMS} \to \text{RMS}} \le \eta$），以保证输出变化可控。

在数学上，求解这个特定范数下的最速下降方向，被称为求梯度的**对偶（Dualizing）**。
*   如果梯度矩阵的奇异值分解（SVD）为 $\nabla_W L = U \Sigma V^\top$。
*   该优化问题的严格数学解为：**$\Delta W \propto - U V^\top$**。
*   **物理意义**：保留梯度矩阵的奇异向量（方向信息），但**舍弃所有奇异值的大小（幅度信息）**。这在数学上被称为对梯度矩阵进行**正交化（Orthogonalizing）**。
*   **附带结论**：文章指出，在这个范数下对偶化梯度，其推导出的更新规则天然包含了 **$\mu$P（最大更新参数化）** 所需的学习率缩放规律。这意味着 Muon 在理论上能实现完美的跨模型宽度学习率迁移（Learning Rate Transfer）。

### Step 4: Dualizing fast via Newton-Schulz（通过牛顿-舒尔茨法快速对偶化）
**核心概念**：将理论解转化为 GPU 友好的高效算法。
*   第三步证明了需要计算 $UV^\top$。但标准的 SVD 计算复杂度极高（$O(N^3)$），在 GPU 上运行极其缓慢。
*   求 $UV^\top$ 本质上是将所有奇异值强行设置为 1。由于奇异值都是正数，这等价于对奇异值对角矩阵 $\Sigma$ 施加符号函数（`sign` function）。
*   符号函数不是多项式，无法直接通过矩阵乘法计算。但可以通过**迭代奇数次多项式**来逼近它。
*   文章引入了 **Newton-Schulz 迭代**（例如基于三次多项式 $p_3(\Sigma) = \frac{3}{2}\Sigma - \frac{1}{2}\Sigma^3$ 的矩阵形式：$X_{k+1} = \frac{1}{2}X_k(3I - X_k^\top X_k)$）。
*   这种迭代完全由纯矩阵乘法（MatMul）组成，极度契合现代 GPU 的 Tensor Core 架构。通常只需 5 次迭代，即可高精度、低延迟地完成正交化操作。

---

### 总结

Jeremy Bernstein 的这篇文章展示了 Muon 并非偶然的工程发现。其完整逻辑链条为：
**控制输出稳定性 $\rightarrow$ 采用 RMS-to-RMS 算子范数约束 $\rightarrow$ 求解带约束的最速下降方向 $\rightarrow$ 得出必须对梯度进行奇异值正交化 $\rightarrow$ 利用 Newton-Schulz 迭代实现工程加速。**

这种基于泛函分析和对偶范数的推导范式（Metrized Deep Learning），为深度学习优化器设计提供了比传统启发式方法更坚实的理论支撑。


### Muon优化器介绍
**Muon (Momentum Orthogonalized)** 优化器的原理。

Muon 是在 2024 年下半年由 Keller Jordan 等人在 `modded-nanogpt` 项目中推出的一种**新型优化器**。它在训练 Transformer 模型（尤其是大语言模型）时，展现出了比 AdamW 更快的收敛速度和更高的样本效率，被认为是目前训练 LLM 的 **SOTA（State-of-the-Art）** 优化器之一。

一句话概括 Muon 的核心思想：**它不再像 AdamW 那样对每个参数进行逐元素的缩放，而是对整个权重矩阵的动量（Momentum）进行正交化（Orthogonalization）处理。**

下面我从**背景痛点、核心数学原理、Newton-Schulz 迭代算法、以及工程实践**四个维度为你拆解。

---

### 1. 背景：AdamW 的局限性与二阶优化的梦想

#### AdamW 的问题：逐元素视角的短视
AdamW 是目前的主流。它的核心是计算梯度平方的移动平均（$v_t$），然后用它来缩放梯度（$m_t / (\sqrt{v_t} + \epsilon)$）。
这本质上是一种**对角预处理（Diagonal Preconditioning）**。它假设每一个参数 $\theta_i$ 都是独立的，忽略了参数之间的相关性。
*   但对于 Transformer 中的线性层（Linear Layer），权重是一个巨大的矩阵 $W \in \mathbb{R}^{in \times out}$。矩阵的行与列之间存在复杂的协方差关系。AdamW 这种“各自为战”的更新方式，并不是最优的。

#### 二阶优化的困境
理论上，利用海森矩阵（Hessian Matrix）或其近似（如 K-FAC, Shampoo）可以捕捉这些相关性，极大加速收敛。但这些方法通常计算量巨大（涉及矩阵求逆），显存占用高，难以在大规模 LLM 训练中落地。

**Muon 的定位**：它是一种极其巧妙的折中方案。它不计算海森矩阵，而是直接对**一阶动量**进行矩阵级的正交化变换，以极低的计算代价实现了近似二阶优化的效果。

---

### 2. Muon 的核心数学原理：正交化更新

Muon 的更新公式非常简单，但数学含义深刻。

假设我们在第 $t$ 步，动量矩阵为 $M_t$（就是 SGD 中的动量，梯度的移动平均）。
AdamW 会做 $M_t / \sqrt{V_t}$（逐元素除法）。
**Muon 则是做矩阵变换：**

$$ \text{Update}_t = \text{Orthogonalize}(M_t) \times \text{Scaling\_Factor} $$

#### 什么是 Orthogonalize（正交化）？
在数学上，对于一个矩阵 $M$，我们可以通过奇异值分解（SVD）写成 $M = U \Sigma V^T$。
其中 $\Sigma$ 是奇异值对角矩阵，代表了矩阵在各个方向上的“能量”或“尺度”。

Muon 的操作本质上是**强制将所有奇异值变为 1**。即：
$$ \text{Orthogonalize}(M) = U V^T $$

**物理意义：**
1.  **去除曲率影响（Whitening/Sphering）**：这一步消除了梯度矩阵在不同方向上的尺度差异（Condition Number 变为 1）。这类似于 Shampoo 优化器的极限情况，让优化器在所有特征方向上以相等的步长前进。
2.  **保留方向，重置模长**：它保留了动量矩阵最核心的旋转/方向信息，但丢弃了幅度信息。

最终，Muon 结合 Nesterov 动量，将更新量缩放到特定的**RMS（均方根）谱范数**，确保更新步长稳定。

---

### 3. 秘密武器：Newton-Schulz 迭代 (Newton-Schulz Iteration)

如果在 GPU 上对每个权重矩阵做 SVD 分解（$O(N^3)$），训练速度会慢得像蜗牛。Muon 之所以能落地，全靠 **Newton-Schulz 迭代**。

这是一种纯矩阵乘法（Matrix Multiplication）的算法，用于快速逼近矩阵的逆平方根，进而实现正交化。

#### 迭代公式
给定输入矩阵 $G$（即动量），我们要找它的正交化形式 $X$。
初始化 $X_0 = G / \|G\|_2$（归一化谱范数）。
迭代步骤（通常只需 5 次）：
$$ X_{k+1} = \frac{1}{2} X_k (3I - X_k^T X_k) $$
或者更高效的形式：
$$ A = X_k^T X_k $$
$$ B = 3I - A $$
$$ X_{k+1} = 0.5 \times X_k B $$

**为什么这很重要？**
1.  **全利用 Tensor Cores**：这个迭代过程只包含矩阵乘法（MatMul），这正是 NVIDIA GPU 最擅长的。在 bf16 精度下，速度极快。
2.  **无需 SVD**：避开了昂贵的特征值分解。
3.  **收敛极快**：通常 5 次迭代就能得到极高精度的正交矩阵。
``` python
# Pytorch code
def newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X
```
---

### 4. 实践中的 Muon：混合策略

你不能把整个模型的所有参数都扔给 Muon。在工程实践（如 `modded-nanogpt` 或 Llama 训练脚本）中，Muon 采用的是**分层混合策略**：

1.  **Muon 负责的大头（>95% 参数）**：
    *   **二维矩阵权重**：所有的 Linear 层（Attention 的 QKV 投影、Output 投影、FFN 的升降维投影）、Embedding 层。
    *   这些矩阵适合做正交化，Muon 能让它们迅速收敛。

2.  **AdamW 负责的小头（<5% 参数）**：
    *   **一维向量**：LayerNorm / RMSNorm 的 scale 向量、Biases（偏置项）。
    *   这些向量做“正交化”没有意义（或者是标量运算），AdamW 更加稳健。

### 5. 总结：Muon 为什么强？

*   **更佳的几何适应性**：Transformer 的权重矩阵本质上是在高维空间做线性变换，Muon 的正交化更新比 AdamW 的逐元素缩放更符合矩阵的几何性质。
*   **计算效率高**：利用 Newton-Schulz 迭代，将复杂的优化步骤转化为简单的 MatMul，几乎不增加训练时长。
*   **显存友好**：相比 Shampoo 等二阶优化器需要存储巨大的预条件矩阵，Muon 的显存占用与 AdamW 相当（甚至可以优化）。

**一句话总结**：Muon 是利用牛顿-舒尔茨迭代实现的高效**动量正交化**优化器，它专门为 Transformer 的**二维权重矩阵**设计，是目前从头预训练大模型的首选加速神器。

### Q：QK-norm之后Muon训练出现了维度坍缩，如何解决？

### Q: QK-clip Muon-clip