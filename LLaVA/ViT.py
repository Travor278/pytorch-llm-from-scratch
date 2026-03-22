# %% [markdown]
# # Vision Transformer (ViT)
# ViT 可以看成是“把图片变成一串 patch token，再交给 Transformer Encoder 去处理”。
# 它的核心思路来自论文：
# "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"。
#
# 和 NLP 里的 token 类似，ViT 会先把一张图片切成很多小块（patch）：
# - 例如输入图片大小是 224 x 224
# - patch size 是 16 x 16
# - 那么一共会得到：
# $$
# \frac{224}{16} \times \frac{224}{16} = 14 \times 14 = 196
# $$
# 个 patch
#
# 接着，每个 patch 会被映射成一个向量：
# $$
# x_{patch} \in \mathbb{R}^{d_{model}}
# $$
# 然后这些 patch 向量会像文本 token 一样进入 Transformer Encoder。
#
# 对多模态模型（如 LLaVA）来说，ViT 的作用不是做分类，
# 而是把图片编码成一串视觉 token，供后续 projector / LLM 使用。

# %% [markdown]
# ## ViT 的整体流程
# 一张图片进入 ViT，通常会经历下面几步：
# 1. Patch Embedding：把图片切成 patch，并映射到 d_model
# 2. Position Embedding：告诉模型每个 patch 在图中的位置
# 3. Transformer Encoder：让 patch token 之间做全局信息交互
# 4. 输出视觉 token 序列
#
# 如果是图像分类任务，很多实现会额外加一个 CLS token；
# 但如果是给 LLaVA 这类多模态模型做视觉前缀，更常见的做法是保留整串 patch token。

# %%
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in (None, ""):
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from Transformer.MHA import MultiHeadAttention
    from Transformer.FFN import PositionwiseFeedForward
else:
    from Transformer.MHA import MultiHeadAttention
    from Transformer.FFN import PositionwiseFeedForward


class PatchEmbedding(nn.Module):
    """
    把图片切成 patch，并把每个 patch 映射成 d_model 维向量。

    输入:
    - x: (batch_size, in_channels, image_size, image_size)

    输出:
    - patch_tokens: (batch_size, num_patches, d_model)
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 768,
    ) -> None:
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size 必须能被 patch_size 整除。")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model

        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        # Conv2d(kernel=stride=patch_size) 等价于：
        # - 先按 patch_size 切块
        # - 再把每个 patch 拉平并做一次线性投影
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, in_channels, image_size, image_size)
        return : (batch_size, num_patches, d_model)
        """
        if x.dim() != 4:
            raise ValueError("输入图片张量必须是 4 维：(B, C, H, W)。")

        batch_size, _, height, width = x.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"输入图片大小必须是 ({self.image_size}, {self.image_size})，"
                f"但拿到了 ({height}, {width})。"
            )

        x = self.proj(x)
        # (B, d_model, H/patch, W/patch)

        x = x.flatten(2)
        # (B, d_model, num_patches)

        x = x.transpose(1, 2)
        # (B, num_patches, d_model)
        return x


class ViTEncoderLayer(nn.Module):
    """
    ViT 的单层 Encoder。

    为了和你当前 Transformer 目录里的实现风格保持一致，
    这里采用 Post-LN:
    - Self-Attention -> Add & Norm
    - FFN -> Add & Norm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ffn = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x shape: (batch_size, seq_len, d_model)
        """
        attn_out, attn_weights = self.self_attn(x, x, x, mask=None)
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x, attn_weights


class VisionTransformer(nn.Module):
    """
    一个适合多模态输入前缀场景的最小 ViT。

    特点：
    - Encoder-only Transformer
    - Patch Embedding
    - Learned Position Embedding
    - 可选 CLS token
    - 默认返回整串视觉 token，方便后续接 projector / GPT
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        use_cls_token: bool = False,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.use_cls_token = use_cls_token

        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=d_model,
        )

        self.num_patches = self.patch_embed.num_patches
        self.seq_len = self.num_patches + (1 if use_cls_token else 0)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        self.embed_dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList(
            [
                ViTEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

        # ViT 中 patch projection 常见做法之一是使用较小方差初始化
        if isinstance(self.patch_embed.proj, nn.Conv2d):
            fan_in = (
                self.patch_embed.proj.in_channels
                * self.patch_embed.proj.kernel_size[0]
                * self.patch_embed.proj.kernel_size[1]
            )
            std = math.sqrt(1.0 / fan_in)
            nn.init.normal_(self.patch_embed.proj.weight, std=std)
            if self.patch_embed.proj.bias is not None:
                nn.init.zeros_(self.patch_embed.proj.bias)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        images shape: (batch_size, in_channels, image_size, image_size)

        return:
        - tokens: (batch_size, seq_len, d_model)
        - all_attn_weights: 每层注意力权重列表
        """
        x = self.patch_embed(images)
        batch_size = x.size(0)

        if self.cls_token is not None:
            cls = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls, x), dim=1)

        x = x + self.pos_embedding
        x = self.embed_dropout(x)

        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            all_attn_weights.append(attn_weights)

        x = self.final_norm(x)
        return x, all_attn_weights


# %% [markdown]
# ## 代码里的形状变化
# 假设输入图片：
# $$
# x \in \mathbb{R}^{B \times 3 \times H \times W}
# $$
# 且：
# - 图片大小 $H=W=224$
# - patch size = 16
#
# 那么 Patch Embedding 之后会变成：
# $$
# (B, 3, 224, 224)
# \rightarrow
# (B, d_{model}, 14, 14)
# \rightarrow
# (B, 196, d_{model})
# $$
# 也就是 196 个 patch token。
#
# 如果打开 CLS token，那么序列长度会变成：
# $$
# 196 + 1 = 197
# $$
# 然后进入 Transformer Encoder，输出 shape 保持不变：
# $$
# (B, S, d_{model})
# $$

# %% [markdown]
# ## 为什么这里默认不用 CLS token
# 在图像分类任务里，CLS token 常用于汇聚全局信息，再接分类头。
# 但在 LLaVA 这类多模态模型里，通常更希望保留整串 patch token，
# 因为：
# - 后续 projector 可以逐 patch 投影到语言空间
# - LLM 可以对不同图像区域保留更细粒度的信息
# - 更适合做“图像前缀 token”
#
# 所以这里默认：
# ```python
# use_cls_token = False
# ```
# 如果后面想做分类实验，再把它打开也很方便。

# %%
def demo_vit_shapes() -> None:
    batch_size = 2
    image_size = 224
    patch_size = 16
    d_model = 128
    num_heads = 4
    num_layers = 3
    d_ff = 512

    images = torch.randn(batch_size, 3, image_size, image_size)

    vit = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=3,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=0.0,
        use_cls_token=False,
    )

    tokens, attn_list = vit(images)

    print("images shape      :", images.shape)
    print("num patches       :", vit.num_patches)
    print("tokens shape      :", tokens.shape)
    print("num layers        :", len(attn_list))
    print("layer1 attn shape :", attn_list[0].shape)
    print()
    print("第一个样本前两个视觉 token：")
    print(tokens[0, :2])


if __name__ == "__main__":
    demo_vit_shapes()
