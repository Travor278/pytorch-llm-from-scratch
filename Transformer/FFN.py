# %% [markdown]
# # Transformer Feed Forward Network
# Transformer block 里除了注意力，还有一个非常重要的模块：
# Feed Forward Network，通常简称 FFN（本质上是position-wise MLP）。
# 它的标准形式是：
# $$
# \mathrm{FFN}(x) = W_2 \ \sigma(W_1 x + b_1) + b_2
# $$
# 其中：
# - 第一层先把维度从 $d_{model}$ 扩张到更大的隐藏维度 $d_{ff}$
# - 中间经过一个非线性激活函数 $\sigma$
# - 第二层再把维度投影回 $d_{model}$
# 在原始 Transformer 论文里，常见设置是：
# $$
# d_{ff} = 4 \times d_{model}
# $$
# 也就是先“升维扩张”，再“压回原维度”。

# %% [markdown]
# ## FFN 的作用是什么
# 注意力负责“不同 token 之间的信息交互”，
# 而 FFN 更像是“每个 token 自己做一次更复杂的非线性特征变换”。
# 所以可以粗略理解为：
# - Attention：跨 token 混信息
# - FFN：对每个 token 单独做深加工
# 它是 position-wise 的，意思是：
# - 不同位置之间不混合
# - 同一套参数对每个位置都重复使用
# 如果输入 shape 是：
# $$
# (B, S, d_{model})
# $$
# 那 FFN 会对每个位置的向量独立作用，但输出 shape 仍然保持：
# $$
# (B, S, d_{model})
# $$

# %% [markdown]
# ## 为什么要先升维再降维
# 先升维到 $d_{ff}$，可以让模型在更大的特征空间里做非线性变换，
# 表达能力更强。
# 如果只做一层线性变换：
# $$
# y = Wx
# $$
# 那么本质上还是线性的。
# 加入激活函数后：
# $$
# \sigma(W_1x)
# $$
# 再配合第二层投影回原维度，就能让每个 token 学到更丰富的局部表示。

# %%
import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    Transformer 里的 Position-wise FFN。
    对每个 token 的最后一维独立作用，挖掘更复杂的特征。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        # d_model: 输入/输出维度
        # d_ff: 中间隐藏层维度，通常比 d_model 大
        # dropout: dropout 概率
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout) # 中间层最容易co-adaption，所以在activation后加
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, d_model)
        return : (batch_size, seq_len, d_model)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# %% [markdown]
# ## 代码和公式如何对应
# 代码：
# ```python
# x = self.fc1(x)
# x = self.activation(x)
# x = self.dropout(x)
# x = self.fc2(x)
# ```
# 对应公式：
# $$
# \mathrm{FFN}(x) = W_2 \ \sigma(W_1 x + b_1) + b_2
# $$
# 其中：
# - `self.fc1` 对应第一层线性变换 $W_1x + b_1$
# - `self.activation` 对应非线性函数 $\sigma$
# - `self.dropout` 对应训练时的随机失活
# - `self.fc2` 对应第二层线性变换 $W_2(\cdot) + b_2$

# %% [markdown]
# ## 为什么叫 Position-wise
# 因为它不会像 attention 那样让不同位置互相通信。
# 比如输入是：
# $$
# x \in \mathbb{R}^{B \times S \times d_{model}}
# $$
# 那么 FFN 实际上是对每个位置单独做：
# $$
# x_{b,s,:} \rightarrow \mathrm{FFN}(x_{b,s,:})
# $$
# 所以：
# - 不同 token 之间不会直接相互影响
# - 但每个 token 自己的表示会变得更复杂
# 这也是为什么 Transformer block 里需要同时有：
# - Attention：做 token 间交互
# - FFN：做 token 内非线性变换

# %%
def demo_ffn_shapes():
    batch_size = 2
    seq_len = 5
    d_model = 8
    d_ff = 32

    x = torch.randn(batch_size, seq_len, d_model)
    ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.0)
    y = ffn(x)

    print("input shape :", x.shape)
    print("output shape:", y.shape)
    print()
    print("第一个样本第一个 token 输入向量：")
    print(x[0, 0])
    print()
    print("第一个样本第一个 token 输出向量：")
    print(y[0, 0])


# %% [markdown]
# ## 一个直观理解
# 如果把 attention 想成“每个 token 去问别人该关注谁”，
# 那 FFN 更像是：
# “每个 token 在拿到上下文信息之后，自己再做一轮更深的思考和特征提炼。”
# 所以在 Transformer 里：
# - MHA 负责信息交换
# - FFN 负责局部加工
# 两者配合起来，表达能力就比单独使用其中一个更强。

# %%
if __name__ == "__main__":
    demo_ffn_shapes()