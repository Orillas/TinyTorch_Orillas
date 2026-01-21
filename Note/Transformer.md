
在现代深度学习，尤其是自然语言处理（NLP）和 Transformer 架构（如 BERT, GPT, Llama）中，LayerNorm 是绝对的基石。没有它，这些大模型的训练将变得极其困难甚至无法收敛。

---

### 1. 为什么需要归一化？（背景）

在深度神经网络的训练过程中，存在一个经典问题：**内部协变量偏移 (Internal Covariate Shift, ICS)**。

简单来说，当数据经过每一层网络的参数运算后，其输出分布会发生变化。这导致下一层网络必须不断适应新的输入分布，不仅拖慢了收敛速度，还容易导致梯度消失或爆炸。

**归一化 (Normalization)** 的核心目的就是：强行把神经网络每一层的输入拉回到一个相对标准的分布（通常是均值为 0，方差为 1），从而让梯度下降更加平稳、高效。

---

### 2. LayerNorm 的核心直觉

LayerNorm 由 Geoffrey Hinton 团队在 2016 年提出。它的核心思想可以用一句话概括：

> **LayerNorm 是在“单个样本”的范围内，对其所有特征进行归一化。**

它不关心你这一个 Batch 里有多少其他样本，它只关心当前这一个样本自身的数值分布。

#### 通俗类比：
假设我们在批改考试试卷：
*   **Batch Normalization (BN)** 像是**“按科目排名”**：把全班同学的“数学成绩”拉出来，算出平均分，看你在全班数学成绩中的位置。这依赖于“全班同学”（Batch）的数据。
*   **Layer Normalization (LN)** 像是**“按学生综合素质评估”**：不看别人，只看**你这一名同学**。把你自己的数学、语文、英语等所有科目的成绩拿出来，算出**你自己**的平均分和波动，然后把你的各科成绩进行标准化。以此判断你哪科相对更强，哪科相对更弱，消除了“试卷难度”（样本整体数值大小）带来的干扰。

---

### 3. LayerNorm 的数学原理

假设我们有一个输入向量 $x$（对应某一个样本的特征向量），维度为 $H$（Hidden Size）。
$x = [x_1, x_2, ..., x_H]$

LayerNorm 的计算包含三个步骤：

#### 第一步：计算均值 (Mean) 和方差 (Variance)
我们在**当前样本的所有隐藏层节点**上计算统计量：

$$ \mu = \frac{1}{H} \sum_{i=1}^{H} x_i $$

$$ \sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2 $$

*注意：这里的求和是针对特征维度 $H$ 进行的，与 Batch Size 完全无关。*

#### 第二步：归一化 (Normalization)
使用计算出的均值和方差，将 $x$ 转化为标准正态分布：

$$ \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} $$

*其中 $\epsilon$ 是一个极小的数（如 1e-5），用于防止分母为 0。*

#### 第三步：仿射变换 (Affine Transformation) —— 这一步至关重要
如果我们强行把每一层的输出都限制在 0 均值 1 方差，可能会破坏模型学到的特征（比如某些激活函数在 0 附近是线性的，我们可能需要非线性区间）。因此，LayerNorm 引入了两个**可学习参数**：缩放因子 $\gamma$ (gamma) 和 平移因子 $\beta$ (beta)：

$$ y_i = \gamma \cdot \hat{x}_i + \beta $$

*   $\gamma$ 和 $\beta$ 的维度与 $x$ 一致。
*   网络在训练过程中会自动学习这两个参数。如果网络发现不需要归一化，它可以把 $\gamma$ 学成 $\sigma$，把 $\beta$ 学成 $\mu$，从而还原回原始输入。

---

### 4. 核心对比：LayerNorm vs Batch Normalization

这是面试和理解中最关键的部分。假设我们有一个数据张量 $(N, D)$。
*   $N$: Batch Size (样本数)
*   $D$: Dimension (特征维度)

| 特性 | Batch Normalization (BN) | Layer Normalization (LN) |
| :--- | :--- | :--- |
| **归一化方向** | **纵向切分**：跨样本。在同一个特征通道上，计算所有样本的均值。 | **横向切分**：跨特征。在同一个样本内，计算所有特征通道的均值。 |
| **依赖性** | 强依赖 **Batch Size**。如果 Batch 太小，统计不准；如果是 1，无法计算。 | **独立于 Batch Size**。Batch Size 为 1 也能照常工作。 |
| **训练/推理差异** | 训练时用当前 Batch 统计量，推理时用滑动平均统计量（Running Mean/Var）。 | **训练和推理完全一致**。不需要维护全局统计量。 |
| **适用场景** | 计算机视觉 (CNN) | 自然语言处理 (RNN, Transformer) |

---

### 5. 为什么 NLP/Transformer 偏爱 LayerNorm？

在 CNN 中，不同样本的同一通道特征往往具有相似的物理意义（比如都在寻找边缘、纹理），所以跨样本归一化（BN）效果很好。

但在 NLP 中，BN 的效果往往很差，LayerNorm 占据统治地位，原因如下：

1.  **序列长度可变 (Variable Sequence Length)**：
    NLP 的输入句子长度不一（有的 10 个词，有的 100 个词）。BN 需要在固定的时间步上计算统计量，对于变长序列，必须进行 Padding（填充）。如果对 Padding 部分做 BN，统计量会完全跑偏。而 LN 是针对单个 Token 内部做计算，不受序列长度影响。

2.  **Batch Size 的限制**：
    大语言模型（LLM）通常极其庞大，显存占用高。这导致训练时 Batch Size 往往很小（甚至只有 1 或 2）。在这种情况下，BN 估算的均值方差噪声极大，导致模型崩溃。而 LN 对 Batch Size 不敏感。

3.  **特征的语义对应**：
    在 RNN/Transformer 中，同一个维度的特征在不同时间步可能代表完全不同的语义。BN 强行对不同时间步的同一维度做归一化，可能破坏了语义信息。而 LN 保证了每个时间步（Token）自身的特征分布稳定。

---

### 6. 进阶：RMSNorm (Root Mean Square Normalization)

既然你是向专家请教，我也补充一个目前的最新趋势。
在 **Llama、Gemma** 等最新的大模型中，LayerNorm 的一个变体 **RMSNorm** 变得非常流行。

**原理**：RMSNorm 认为，**中心化（减去均值 $\mu$）** 并不重要，重要的是**缩放（除以方差）**。
$$ \text{RMS}(x) = \sqrt{\frac{1}{H} \sum x_i^2 + \epsilon} $$
$$ \bar{x}_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma $$

**优点**：
*   少算了一个均值 $\mu$，计算速度更快。
*   效果与 LayerNorm 几乎持平，甚至在某些深层网络中更稳定。

LayerNorm 的本质是 **“样本内的特征标准化”**。它通过消除单个样本内部的数值尺度波动，让神经网络的训练更稳定。由于它不依赖 Batch Size 且能完美处理变长序列，它成为了 Transformer 和所有现代 NLP 模型的默认归一化方案。

### Q2: LayerNorm (LN) 是如何允许我们使用更大的学习率并加速收敛的。

从**前向传播的数值稳定性**、**反向传播的梯度动力学**以及**损失曲面的几何特性**三个核心维度，为你剖析 LayerNorm (LN) 是如何允许我们使用更大的学习率并加速收敛的。

---

### 1. 前向传播：抑制“数值爆炸”，稳定激活分布

在没有归一化的深层网络中，存在一个**乘法效应**。假设网络有 $L$ 层，每一层的权重矩阵为 $W_l$。

#### 只有 LayerNorm 之前的世界
如果每一层权重的尺度（Scale）稍微大于 1（例如初始化不当或更新导致），经过几十层的连续矩阵乘法，输出值 $x_L$ 会呈指数级增长。
$$ x_L \approx \prod W_l \cdot x_0 $$
这种**激活值幅度的剧烈波动**（Internal Covariate Shift 的一种表现）会导致两个严重后果：
1.  **落入饱和区**：如果你使用 sigmoid/tanh 等激活函数，巨大的输入值会使激活进入饱和区，梯度趋近于 0（梯度消失）。
2.  **数值不稳定**：即使是 ReLU，巨大的数值也会导致下一层的权重更新步长变得极不稳定。

#### LayerNorm 的作用
LayerNorm 强制将每一层的输出分布拉回到 $\mu=0, \sigma=1$。
$$ \text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2}} $$
（忽略 $\gamma, \beta$ 带来的仿射变换，仅看标准化过程）

这意味着，**无论前一层的权重 $W$ 变得多大，经过 LN 后，输出的激活值幅度都被限制在一个固定的范围内。** 这种确定性切断了“数值爆炸”的传播路径，保证了数据在深层网络中流动时，始终保持在激活函数的敏感区间（非饱和区），从而保留了有效的信息传递。

---

### 2. 反向传播：权重尺度的不变性与梯度的“自动调节”

这是 LayerNorm 能够允许**更大特定学习率**的最核心数学原理。我们称之为**权重尺度不变性 (Weight Scale Invariance)**。

#### 数学推导
假设某层的计算为 $y = \text{LN}(W \cdot x)$。
如果我们把权重 $W$ 放大 $\lambda$ 倍，即 $W' = \lambda W$。
观察 LN 的计算公式（分子分母同时约去了 $\lambda$）：
$$ \text{LN}(\lambda W x) = \frac{\lambda W x - \text{Mean}(\lambda W x)}{\sqrt{\text{Var}(\lambda W x)}} = \frac{\lambda (Wx - \mu)}{\lambda \sqrt{\sigma^2}} = \text{LN}(W x) $$

**结论 1：前向传播输出不变。**
权重的整体缩放不会改变 LayerNorm 的输出。

**结论 2：反向传播梯度反向缩放（关键点）。**
根据链式法则，如果输出 $y$ 对 $W$ 不变，那么损失函数 $\mathcal{L}$ 对缩放后的权重 $W'$ 的梯度会发生什么变化？
$$ \frac{\partial \mathcal{L}}{\partial W'} = \frac{\partial \mathcal{L}}{\partial (\lambda W)} = \frac{1}{\lambda} \frac{\partial \mathcal{L}}{\partial W} $$

#### 这意味着什么？
这引入了一种非常巧妙的**自我调节机制**：
*   **当权重 $W$ 很大时**（$\lambda$ 大）：梯度 $\nabla W$ 会自动变小（乘以 $1/\lambda$）。这防止了在权重本来就很大的情况下，梯度更新步长过大导致模型发散。
*   **当权重 $W$ 很小时**（$\lambda$ 小）：梯度 $\nabla W$ 会自动变大。这加速了小权重的更新，使其快速脱离微小区域。

**总结**：LayerNorm 使得梯度的幅度与权重的幅度成反比。这实际上起到了一种**自适应学习率**的效果。因此，即使你人为设置了一个较大的全局学习率，LayerNorm 也会在局部防止某些层因为参数过大而发生“梯度爆炸”。

---

### 3. 优化景观：平滑损失曲面 (Smoothing the Loss Landscape)

除了数值和梯度，LayerNorm 还改变了损失函数 $\mathcal{L}$ 的几何形状。这是由 MIT 的 Santurkar 等人在研究 BatchNorm 时提出的理论，同样适用于 LayerNorm。

#### Lipschitz 连续性
LayerNorm 限制了激活值的增长，同时也限制了损失函数梯度对输入变化的敏感度。在数学上，这降低了损失函数及其梯度的 **Lipschitz 常数**。

*   **没有 LN**：损失曲面可能非常崎岖（Jagged），存在极陡峭的悬崖和极平坦的高原。如果学习率稍大，参数更新就会直接冲出悬崖（Loss 震荡或发散）。
*   **有 LN**：损失曲面变得更加**平滑（Smooth）**。曲率（Hessian 矩阵的特征值）更加均匀。

#### 为什么能加速收敛？
想象你在下山（梯度下降）：
*   在崎岖的山路上，你必须小心翼翼，步子迈得很小（**小学习率**），否则容易摔死。
*   在平滑的斜坡上，你可以放心大胆地迈大步（**大学习率**），直线冲向谷底。

LayerNorm 将“崎岖山路”变成了“平滑斜坡”，使得优化器可以选取更大的步长，从而**显著减少了到达最优解所需的迭代步数（Epochs）**。

---

### 4. 消除自由度：降低优化难度

最后，从参数空间的角度来看。
由于 LN 具有尺度不变性（$W$ 和 $\lambda W$ 等价），参数空间中实际上有很多方向是“无效”的（即改变权重长度不改变 Loss）。

LayerNorm 隐式地将优化过程约束在一个**超球面**上。这减少了优化器需要搜索的有效参数空间的复杂性。它消除了权重向量的“长度”这一自由度带来的干扰，让优化器专注于调整权重向量的“方向”，这通常是更高效的收敛路径。

### 总结
1.  **前向稳定性**：将激活值锁定在均值 0 方差 1，防止网络深处的数值爆炸或消失。
2.  **梯度自调节 (核心机制)**：通过 $ \frac{\partial \mathcal{L}}{\partial (\lambda W)} = \frac{1}{\lambda} \frac{\partial \mathcal{L}}{\partial W} $ 的特性，使得大权重获得小梯度，天然防止参数更新过冲，允许使用更大的全局学习率。
3.  **优化景观平滑化**：降低了 Loss 曲面的 Lipschitz 常数，让曲面更平滑，使梯度下降路径更直、更可预测。

### Q3: 数学推导经过 Norm 层的权重 W 的梯度$|\nabla_W L| \propto \frac{1}{\|W \|}$

这是一个非常硬核的数学推导请求，直击 LayerNorm (及 BatchNorm) 优化机制的灵魂。

这个“刹车机制”的本质来源于归一化操作引入的 **尺度不变性 (Scale Invariance)**。

简单来说：**因为 LayerNorm 抹除了权重的“大小”信息，所以权重的“大小”反过来会制约梯度的“大小”。**

下面我为你详细推导为什么 $\nabla W$ 与 $\|W\|$ 成反比。

---

### 1. 设定符号与前置条件

假设我们关注神经网络中的某一层。
*   **输入**：$x$
*   **权重矩阵**：$W$
*   **线性变换（Pre-activation）**：$z = Wx$
*   **LayerNorm 输出**：$y = \text{LN}(z) = \frac{z - \mu}{\sigma} \cdot \gamma + \beta$
    *   这里 $\mu$ 是均值，$\sigma$ 是标准差。
*   **最终 Loss**：$\mathcal{L}$

为了简化推导并聚焦核心原理，我们**忽略**仿射变换参数 $\gamma$ 和 $\beta$（把它们看作常数或合并到下一层），并假设数据已经中心化（$\mu=0$）。此时简化的 LayerNorm 为：
$$ y = \frac{z}{\sigma} = \frac{Wx}{\sigma(Wx)} $$

其中标准差 $\sigma$ 是关于 $z$ 的函数：
$$ \sigma(z) = \sqrt{\frac{1}{H}\sum (z_i - \mu)^2} \approx \sqrt{\text{Var}(z)} $$

---

### 2. 第一步：证明前向传播的“尺度不变性”

假设我们将权重矩阵 $W$ 放大 $\lambda$ 倍（$\lambda > 0$），得到新的权重 $W' = \lambda W$。

1.  **新的线性输出**：
    $$ z' = W'x = (\lambda W)x = \lambda (Wx) = \lambda z $$
2.  **新的标准差**：
    由于标准差计算是线性的（$\sqrt{\text{Var}(\lambda z)} = \lambda \sqrt{\text{Var}(z)}$）：
    $$ \sigma' = \sigma(z') = \sigma(\lambda z) = \lambda \sigma(z) = \lambda \sigma $$
3.  **新的归一化输出**：
    $$ y' = \frac{z'}{\sigma'} = \frac{\lambda z}{\lambda \sigma} = \frac{z}{\sigma} = y $$

**结论**：
$$ \text{LN}(\lambda W \cdot x) = \text{LN}(W \cdot x) $$
这意味着：**无论你怎么缩放权重 $W$（改变其范数 $\|W\|$），LayerNorm 的输出 $y$ 保持不变，因此 Loss $\mathcal{L}$ 也保持不变。**

---

### 3. 第二步：反向传播的梯度推导（核心部分）

我们利用上述的“不变性”来推导梯度关系。

令 $W$ 为原始权重，$\widehat{W} = \lambda W$ 为缩放后的权重。
根据前向传播结论，我们有：
$$ \mathcal{L}(\widehat{W}) = \mathcal{L}(\lambda W) = \mathcal{L}(W) $$

现在，我们想知道**缩放后的权重梯度** $\frac{\partial \mathcal{L}}{\partial \widehat{W}}$ 是什么。

根据链式法则，我们将 $\mathcal{L}(\widehat{W})$ 对 $\widehat{W}$ 求导。为了看清关系，我们利用变量代换：$\widehat{W} = \lambda W$，则 $W = \frac{1}{\lambda} \widehat{W}$。

这里有一个更直观的推导路径：利用**齐次函数 (Homogeneous Function)** 的欧拉定理性质，或者直接对等式 $\mathcal{L}(\lambda W) = \mathcal{L}(W)$ 两边关于 $W$ 求导可能比较绕。

**我们采用最直接的定义法：**

假设 Loss 对输出 $y$ 的梯度为 $\delta_y = \frac{\partial \mathcal{L}}{\partial y}$。由于 $y$ 不变，$\delta_y$ 也不变。
我们需要求 $\frac{\partial \mathcal{L}}{\partial W}$。

$$ \frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial W} $$
$$ \frac{\partial \mathcal{L}}{\partial W} = \delta_y \cdot \frac{\partial (\frac{z}{\sigma})}{\partial z} \cdot x^T $$

关键在于中间项 $\frac{\partial (\frac{z}{\sigma})}{\partial z}$（LayerNorm 的雅可比矩阵）。
对于 $y = \frac{z}{\sigma}$，根据商的求导法则：
$$ \frac{\partial y}{\partial z} = \frac{1}{\sigma} I - \frac{z}{\sigma^2} \frac{\partial \sigma}{\partial z} $$
这里只要注意到**分母中包含 $\sigma$**。
所以原始梯度 $\nabla_W \mathcal{L}$ 的量级大约与 $\frac{1}{\sigma}$ 成正比。

**现在来看缩放后的梯度 $\nabla_{\widehat{W}} \mathcal{L}$：**
$$ \frac{\partial \mathcal{L}}{\partial \widehat{W}} = \delta_y \cdot \frac{\partial (\frac{z'}{\sigma'})}{\partial z'} \cdot x^T $$
注意这里的分母变成了 $\sigma'$。我们已知 $\sigma' = \lambda \sigma$。
所以：
$$ \frac{\partial (\frac{z'}{\sigma'})}{\partial z'} \approx \frac{1}{\sigma'} (\dots) = \frac{1}{\lambda \sigma} (\dots) = \frac{1}{\lambda} \cdot \left[ \frac{1}{\sigma}(\dots) \right] $$

结合起来，我们可以得出严格的数学关系：
$$ \nabla_{\widehat{W}} \mathcal{L} = \frac{1}{\lambda} \nabla_{W} \mathcal{L} $$

---

### 4. 结论与“刹车”的物理意义

如果我们将缩放因子 $\lambda$ 视为权重的范数（即 $\lambda = \|W\|$），那么上面的公式就变成了：

$$ \nabla_{W} \mathcal{L} \propto \frac{1}{\|W\|} $$

这就是所谓的 **“刹车机制”**：

1.  **权重越大，梯度越小**：
    当权重 $W$ 的范数 $\|W\|$ 变大时（例如 $\lambda=10$），根据公式，计算出的梯度 $\nabla W$ 会自动缩小 10 倍。
    $$ W_{new} = W - \eta \cdot \frac{1}{10} \nabla W $$
    这阻止了权重在本来很大的情况下继续大幅度更新，防止了“梯度爆炸”。

2.  **权重越小，梯度越大**：
    当权重 $W$ 的范数很小时（例如 $\lambda=0.1$），梯度会自动放大 10 倍。
    这帮助小权重快速增长，逃离 0 附近的区域（避免梯度消失或停滞）。

### 5. 补充：梯度的正交性 (Orthogonality)

除了反比关系，这个推导还有一个重要的推论：**梯度的方向**。
对于具有尺度不变性的函数 $\mathcal{L}(W) = \mathcal{L}(\lambda W)$，如果我们对 $\lambda$ 求导并令 $\lambda=1$，我们会得到：
$$ \nabla_W \mathcal{L} \cdot W = 0 $$
（这是欧拉齐次函数定理的一个推论）。

**几何意义**：
LayerNorm 导致 Loss 对权重 $W$ 的梯度向量，始终与权重向量 $W$ **垂直（正交）**。

*   这意味着梯度更新**只改变 $W$ 的方向，不改变 $W$ 的模长（在一阶近似下）**。
*   所有的优化实际上都是在超球面上进行的。这极大地简化了优化路径，因为优化器不需要纠结于“我要把权重伸长多少”，而只需关注“我该把权重转向哪里”。

这就是 LayerNorm 能够使用超大学习率并快速收敛的深层数学原因。

### Pre-LN 与 Post-LN 的演进

在实际架构中，LayerNorm 放哪儿至关重要：

* **Post-LN (原始 Transformer 使用)**： Norm(x + SubLayer(x))。

  * 将 LN 放在残差连接之后。这种方式模型性能上限更高，但非常难训练，通常需要严密的 Learning Rate Warm-up。

* **Pre-LN (主流模型如 GPT/Llama 使用)**： x + SubLayer(Norm(x))。

   * 将 LN 放在残差路径内部。这使得梯度流更加顺畅，训练更稳定，模型更容易收敛。


以下是将图片内容转换为 Markdown 格式的文本：

### 1. 核心数学前提：零阶齐次性

LayerNorm 具有一个关键性质：它是**零阶齐次 (Zero-order Homogeneous)** 的。这意味着如果你把输入按比例缩放，输出保持不变。

对于一个神经元的运算 $y = \text{LayerNorm}(W \cdot x)$，由于 LayerNorm 会除以标准差，所以对于任何缩放因子 $\alpha > 0$：

$$
f(\alpha W) = f(W)
$$

其中 $f$ 表示从权重到 Loss 的映射。

---

### 2. 利用欧拉齐次函数定理推导

在数学上，如果一个函数满足 $f(\alpha W) = \alpha^k f(W)$，它被称为 $k$ **阶齐次函数**。根据**欧拉齐次函数定理**：

$$
W \cdot \nabla_W f(W) = k \cdot f(W)
$$

对于 LayerNorm 后的 Loss 而言，$k=0$ （零阶齐次），所以：

$$
W \cdot \nabla_W L = 0 \cdot L = 0
$$

---

### 3. 梯度范数与权重范数的反比关系

现在我们来解释你图片中提到的公式。假设我们有两组权重，$W$（模长为 1）和 $W_{actual} = \rho W$ （模长为 $\rho$）。

根据链式法则，我们考察 Loss 对 $\alpha$ 的导数：

1.  我们已知 $L(\alpha W) = L(W)$ （Loss 不随 $\alpha$ 缩放而变化）。
2.  两边对 $W$ 求导：$\nabla_W L(W) = \frac{\partial L(\alpha W)}{\partial W}$。
3.  根据复合函数求导：
    $$
    \nabla_W L(W) = \alpha \cdot \nabla_{\alpha W} L(\alpha W)
    $$
4.  代入 $W_{actual} = \alpha W$：
    $$
    \nabla_W L(W) = \alpha \cdot \nabla_{W_{actual}} L
    $$
5.  变形得到：
    $$
    \nabla_{W_{actual}} L = \frac{1}{\alpha} \nabla_W L(W)
    $$

由于 $\alpha$ 本质上就是 $W_{actual}$ 的模长（即 $\|W_{actual}\|$），而 $\nabla_W L(W)$ 是在单位圆上的梯度（是一个相对固定的参考值），所以：

$$
\|\nabla_{W_{actual}} L\| = \frac{1}{\|W_{actual}\|} \cdot \|\text{常数梯度}\|
$$

$$
\|\nabla_{W_{actual}} L\| \propto \frac{1}{\|W_{actual}\|}
$$
深度理解：这对优化意味着什么？

这个数学结果带来了两个极其重要的物理特性：

**A. 步长的自适应调节 (Automatic Re-scaling)**

在随机梯度下降（SGD）中，权重的更新量为 $ΔW=−η∇ 
W_L$。

当权重 $W$ 很小时，梯度变大，模型迫使 $W$ 快速增长，防止学习停滞。

当权重$ W $变得很大时，梯度自动变小，模型让更新步速降下来。

结果： 即使你设置的学习率 $η$ 很大，LayerNorm 也会通过梯度的反比关系起到“限速”作用，防止模型跑飞。

**B. 优化空间从“欧几里得空间”转为“球面”**

因为 $W⋅∇_WL=0$，梯度总是垂直于权重向量。

这意味着权重的更新不会显著改变权重的模长，而主要是在改变权重的方向。

传统的神经网络优化是在广袤的平面寻找最优点，而有 LayerNorm 的模型是在一个高维球面的表层寻找最优点。这大大缩小了搜索空间，显著加快了收敛速度。
​