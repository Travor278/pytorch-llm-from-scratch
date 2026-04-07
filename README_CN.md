[English](README.md) | 中文

# PyTorch 学习笔记

这是一个从零开始的 PyTorch 学习仓库，当前覆盖九条线：
- Autograd 核心原理（`00` 到 `08`）
- Dataset / DataLoader / TensorBoard 数据流实践
- `nn.Module` 各类层、损失函数与完整网络搭建
- 模型保存、加载与预训练模型使用
- CPU / GPU 完整训练流程
- 进阶专题模块：训练工程、损失函数、模型架构、参数高效微调、生成模型、评估推理、多模态
- Transformer（Encoder-Decoder，从论文从头实现）
- GPT（Decoder-only 语言模型，含 RoPE / KV Cache / BPE tokenizer）

## 目录结构

### 1）Autograd 系列（核心）

| 文件 | 主题 | 内容 |
|------|------|------|
| `00_interactive_playground.py` | 交互式试验场 | PyTorch 环境检查、简单 autograd 示例、Jacobian 计算 |
| `01_manual_gradient.py` | 手动梯度计算 | 手写链式法则实现梯度下降，拟合 `y = 2x + 1` |
| `02_autograd_basics.py` | Autograd 入门 | `requires_grad`、`backward()`、`no_grad()`、梯度累加陷阱 |
| `03_computational_graph.py` | 计算图机制 | `grad_fn`、叶子张量、`retain_graph`、`detach()` |
| `04_hooks.py` | Hook 机制 | Tensor/Module Hook、梯度查看与修改 |
| `05_jacobian_and_advanced_autograd.py` | Jacobian 与高阶导数 | VJP、Jacobian、Hessian 基础 |
| `06_gradient_accumulation_and_tricks.py` | 训练技巧 | 梯度累加、梯度裁剪、梯度检查、参数冻结 |
| `07_custom_autograd_function.py` | 自定义 Function | 自定义 ReLU/STE/多输入函数 + `gradcheck` |
| `08_tensor_operations_and_gpu.py` | 张量与 GPU | 创建、变形、索引、广播、GPU、NumPy 互转 |

### 2）数据流与可视化

| 文件 | 说明 |
|------|------|
| `generate_labels.py` | 为蚂蚁/蜜蜂图像生成标签 txt 文件 |
| `read_data.py` | 构建自定义 `Dataset`（hymenoptera） |
| `test_Tf.py` | `transforms.ToTensor()` 基础用法 |
| `Useful_TF.py` | 常用 transforms：`Normalize`、`Resize`、`Compose`、`RandomCrop` |
| `test_tb.py` | TensorBoard 基础：记录图像和标量 |
| `dataset_transform.py` | CIFAR-10 下载 + transforms 组合 + TensorBoard 预览 |
| `dataloader.py` | 使用 `DataLoader` 按批加载 CIFAR-10 并记录多轮图像 |

### 3）nn.Module —— 层、损失函数与完整网络搭建

| 文件 | 说明 |
|------|------|
| `nn_module.py` | 最小自定义 `nn.Module`（`forward` 逻辑示例） |
| `nn_relu.py` | `ReLU` / `Sigmoid` 激活函数演示，并对 CIFAR-10 批数据做可视化 |
| `nn_maxpool.py` | `MaxPool2d` 下采样演示（`ceil_mode=True`）并写入 TensorBoard |
| `nn_conv.py` | `F.conv2d` 的 stride / padding 对比实验 |
| `nn_conv2d.py` | `nn.Conv2d` 处理 CIFAR-10，并在 TensorBoard 可视化输入输出 |
| `nn_linear.py` | `nn.Linear` 基础用法：将 CIFAR-10 图像展平后接全连接层 |
| `nn_seq.py` | 用 `nn.Sequential` 搭建 Conv→Pool→Flatten→Linear 网络，TensorBoard 可视化计算图 |
| `nn_loss.py` | 常用损失函数：`L1Loss`、`MSELoss`、`CrossEntropyLoss` |
| `nn_loss_network.py` | 完整 CNN（Conv2d + MaxPool2d + Flatten + Linear）+ 损失计算 + `backward()` |
| `nn_optim.py` | 加入 `torch.optim.SGD`：`zero_grad` → `backward` → `step` 完整训练循环，跑 20 轮 |
| `nn_common_layers.py` | 常用层汇总：`BatchNorm2d`、`Dropout`、`AvgPool2d`、`AdaptiveAvgPool2d`、`Flatten`、`Embedding` |

### 4）模型保存、加载与预训练模型

| 文件 | 说明 |
|------|------|
| `model.py` | 独立定义 `Moli` CNN 网络结构（供 `train.py` 导入） |
| `model_save.py` | 两种保存方式：整个模型 vs `state_dict`（官方推荐）；自定义类保存陷阱演示 |
| `model_load.py` | 两种加载方式：方式1需 `weights_only=False`；方式2先建模型再 `load_state_dict`；PyTorch 2.6 注意事项 |
| `model_pretrained.py` | 加载有/无预训练权重的 VGG16；用 `add_module` 追加层或直接替换层修改分类头 |

### 5）完整训练流程

| 文件 | 说明 |
|------|------|
| `train.py` | CPU 完整训练：数据加载 → 模型 → 损失 → 优化器 → train/eval 循环 → TensorBoard → 每轮保存模型 |
| `train_gpu_1.py` | GPU 训练（`.cuda()` 写法）：硬编码 CUDA 设备，加入 `time` 计时 |
| `train_gpu_2.py` | GPU 训练（`.to(device)` 写法，推荐）：设备变量统一管理，CPU/GPU 一行切换 |

### 6）控制台速查脚本

| 文件 | 说明 |
|------|------|
| `Console.py` | 快速检查 CUDA 可用性与 `torch` API 的交互式小脚本 |

---

### 7）进阶专题模块

#### `training_engineering/`（训练工程）

| 文件 | 说明 |
|------|------|
| `optimizer_scheduler.py` | AdamW vs Adam（解耦 weight decay）、参数组差异化学习率、StepLR / CosineAnnealing / OneCycleLR、LambdaLR 实现 warmup |
| `amp_training.py` | FP16 / BF16 / FP32 格式对比、`autocast` + `GradScaler`、loss scaling 原理 |
| `gradient_checkpoint.py` | 激活重计算原理、`torch.utils.checkpoint` 用法、显存–计算权衡、分段策略 |
| `distributed_training.py` | DataParallel vs DDP 本质区别、Ring-AllReduce 梯度同步、`mp.spawn` 单机多卡启动 |

#### `loss_functions/`（损失函数）

| 文件 | 说明 |
|------|------|
| `kl_divergence.py` | KL 散度定义、正向 vs 反向 KL（均值寻找 vs 模式寻找）、温度缩放、知识蒸馏（Hinton 2015）、与 SEED 的关联 |
| `contrastive_loss.py` | InfoNCE、NT-Xent（SimCLR）、Triplet Loss、in-batch negatives 与温度参数 |
| `focal_loss.py` | Focal Loss 处理类别不平衡（Lin 2017）、调制因子 `(1−p_t)^γ`、多分类扩展 |
| `label_smoothing.py` | 标签平滑理论、KL 惩罚解释、`F.cross_entropy(label_smoothing=)`、校准与蒸馏的权衡 |

#### `model_architecture/`（模型架构）

| 文件 | 说明 |
|------|------|
| `attention_mechanism.py` | Scaled Dot-Product Attention、因果掩码、多头注意力从头实现、`nn.MultiheadAttention` |
| `positional_encoding.py` | 正弦 PE（Vaswani 2017）、可学习 PE（BERT）、RoPE（LLaMA / LLaVA） |
| `residual_connection.py` | 残差梯度流分析、BasicBlock + projection shortcut、Post-LN vs Pre-LN、RMSNorm |
| `normalization.py` | BatchNorm / LayerNorm / GroupNorm / InstanceNorm — 统计维度与选型指南 |

#### `peft/`（参数高效微调）

| 文件 | 说明 |
|------|------|
| `lora_principle.py` | LoRA 数学 `W = W₀ + (α/r)BA`、B 零初始化、`LoRALinear` 包装、替换模型层 |
| `adapter_layers.py` | 瓶颈 Adapter（Houlsby 2019）、门控 vs 串行插入、PEFT 方法横向对比 |

#### `generative_models/`（生成模型）

| 文件 | 说明 |
|------|------|
| `vae_basics.py` | ELBO 推导、重参数技巧 `z = μ + σε`、KL 解析解、β-VAE |
| `diffusion_basics.py` | DDPM 前向过程 `q(x_t\|x_0)`、噪声预测损失 `L_simple`、正弦时间嵌入、祖先采样 |
| `gan_training.py` | GAN 极大极小目标、非饱和损失、训练循环、模式崩塌分析、WGAN-GP 梯度惩罚 |

#### `evaluation_inference/`（评估与推理）

| 文件 | 说明 |
|------|------|
| `sampling_strategies.py` | 贪心、温度采样、Top-k、Top-p（Nucleus，Holtzman 2020）、重复惩罚 |
| `beam_search.py` | Beam Search 算法、长度惩罚（Wu 2016）、Diverse Beam Search 概念、解码策略选型 |
| `systematic_evaluation.py` | 系统化 LLM 评估总览：离线客观指标、judge 开放题评估、pairwise battle、胜率、Elo、bootstrap 直觉 |
| `mmlu_eval.py` | 教学版 MMLU 评估：few-shot 多选题 prompt 构造、答案解析、按学科准确率与宏平均 |
| `mt_bench_eval.py` | 教学版 MT-Bench 流程：多轮问题集、模型生成、judge 打分、turn/category 汇总 |
| `lmsys_arena_eval.py` | 教学版 LMSYS/Chatbot Arena 流程：匿名 A/B 对战、投票记录、胜率与 Elo 更新 |

#### `multimodal/`（多模态专项）

| 文件 | 说明 |
|------|------|
| `clip_contrastive.py` | CLIP 双向 InfoNCE、可学习温度 `logit_scale`、零样本分类、在 LLaVA/SEED 中的角色 |
| `cross_attention.py` | 跨模态 Cross-Attention、门控交叉注意力（Flamingo 2022）、Q-Former（BLIP-2 2023） |

---

### 8）Transformer/

从头实现 *Attention Is All You Need*（Vaswani et al. 2017）的 Encoder-Decoder Transformer，以带注释的 notebook 形式呈现。

| 文件 | 说明 |
|------|------|
| `MHA.py` | 多头注意力从头实现：Q/K/V 投影、Scaled Dot-Product、head 拆分与合并、输出注意力权重 |
| `FFN.py` | Position-wise 前馈网络：两层线性 + ReLU，维度从 `d_model` 扩张到 `d_ff` |
| `PostionalEncoding.py` | 正弦位置编码：奇偶维度分别用 sin/cos，加到 token embedding 上 |
| `create_mask.py` | 三种掩码原理与实现：源序列 padding mask、目标序列因果 mask、memory mask |
| `Encoder.py` | Encoder Layer（双向 self-attention + FFN + Add&Norm）；堆叠成完整 Encoder |
| `Decoder.py` | Decoder Layer（masked self-attention + cross-attention + FFN + Add&Norm）；堆叠成完整 Decoder |
| `Transformer.py` | 将上述模块拼接成完整 Encoder-Decoder Transformer；包含 Generator（线性 + log-softmax）输出头 |
| `train.py` | 在小型平行语料（`data/parallel_toy.tsv`）上训练：构建词表、teacher forcing 训练循环、掩码构造 |
| `test.py` | MHA、掩码工具函数及完整 Transformer 的形状与前向传播测试 |

---

### 9）GPT/

Decoder-only GPT 语言模型，在约 40 万字古典汉语文本上训练，所有组件均从头实现。

| 文件 | 说明 |
|------|------|
| `MHA.py` | 因果多头自注意力：Q/K 上可选 RoPE，返回注意力权重，支持 KV Cache（`past_kv`） |
| `RoPE.py` | 旋转位置编码：`rotate_half` 使用 LLaMA 风格配对，预计算 `cos`/`sin` 表，作用于 Q 和 K |
| `FFN.py` | 前馈模块：GELU 激活，可配置 `d_ff` 隐藏维度 |
| `GPTBlock.py` | Pre-LN Decoder Block：LayerNorm → 自注意力 → 残差；LayerNorm → FFN → 残差；透传 KV Cache |
| `GPT.py` | 完整 GPT 模型：token embedding + 可选 Learned PE（`use_rope=False` 时的 fallback），堆叠 `GPTBlock`，最终 LayerNorm，权重绑定 `lm_head` |
| `create_mask.py` | 因果掩码工具（`make_causal_mask`）：上三角 bool 掩码，用于自回归解码 |
| `tokenizer.py` | `CharTokenizer`：字符级 tokenizer，含 pad/bos/eos/unk 特殊 token，`min_freq` 过滤，保存/加载为 JSON |
| `bpe_tokenizer.py` | `BPETokenizer`：在语料上训练 BPE；增量更新 pair_counts 提升训练速度；贪心优先合并编码；保存/加载为 JSON |
| `train_gpt.py` | 训练脚本：cosine LR warmup（`get_lr`）、tqdm 进度条、char/BPE tokenizer 切换（`--tokenizer`）、checkpoint 保存与恢复、梯度裁剪 |
| `sample.py` | 生成脚本：KV Cache 自回归解码、温度缩放、top-k 过滤、top-p（nucleus）采样 |
| `test_gpt.py` | 单元测试：GELU 激活、权重绑定、Pre-LN block 行为、前向形状、RoPE 正确性、KV Cache 与全量前向等价性 |

#### 训练

```bash
# 字符级 tokenizer
python GPT/train_gpt.py --text-path GPT/_train_text_large.txt --device cpu --max-steps 2000

# BPE tokenizer（推荐）
python GPT/train_gpt.py --text-path GPT/_train_text_large.txt \
  --tokenizer bpe --bpe-vocab-size 8500 \
  --d-model 256 --num-heads 8 --num-layers 6 --d-ff 1024 \
  --device cpu --max-steps 2000
```

#### 采样

```bash
python GPT/sample.py --ckpt GPT/checkpoints/gpt_char_best.pt \
  --prompt "春风" --temperature 0.8 --top-p 0.9 --max-new-tokens 120
```

---

## 数据集与本地目录

- `data/`：MNIST
- `dataset/`：hymenoptera（ants vs bees）
- `dataset2/`：CIFAR-10
- `GPT/_train_text_large.txt`：约 40 万字古典汉语文本（四书五经、宋词、全唐诗）

数据集、TensorBoard 日志和训练生成的模型文件体积较大，已通过 `.gitignore` 排除，不会提交到 GitHub。

## 环境

```bash
pip install torch torchvision tensorboard tqdm
```

## 推荐学习顺序

1. `00` → `08`：Autograd 主线
2. transforms / dataset 脚本 → DataLoader / TensorBoard
3. `nn` 各层：激活与池化 → 卷积 → 全连接 → Sequential → 损失函数 → 优化器 → 常用层
4. 模型保存与加载：`model_save.py` → `model_load.py` → `model_pretrained.py`
5. 完整训练流程：`train.py`（CPU）→ `train_gpu_1.py` → `train_gpu_2.py`
6. 进阶专题：`training_engineering/`、`model_architecture/`、`loss_functions/`、`peft/`、`generative_models/`、`evaluation_inference/`、`multimodal/`
7. `Transformer/`：从论文从头搭建 Encoder-Decoder Transformer
8. `GPT/`：搭建 Decoder-only GPT，在真实文本上训练，生成采样
