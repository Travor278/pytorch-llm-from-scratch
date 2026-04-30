# nn.Module 总结

快速回顾 PyTorch 里的 `nn.Module`。

一句话理解：
`nn.Module` 是 PyTorch 中“带状态管理能力的可调用计算单元”基类。

它不只是一个普通 Python 类，还负责统一管理：

- 子模块
- 可训练参数
- 非训练但需要保存的状态
- 训练/推理模式
- 设备迁移
- 模型保存与加载

---

## 1. 为什么要继承 `nn.Module`

如果只是写一个普通类：

```python
class MyLayer:
    def forward(self, x):
        return x + 1
```

虽然也能做计算，但它缺少很多 PyTorch 的核心能力：

- 不能自动收集参数给优化器
- 不能直接用 `model.to("cuda")`
- 不能用 `model.train()` / `model.eval()`
- 不能方便地 `state_dict()` 保存权重
- 不能自动管理子模块

而继承 `nn.Module` 后，这些功能都会接入统一框架。

---

## 2. 一个最小例子

仓库里的 [nn_module.py](./nn_module.py) 就是一个最小例子：

```python
import torch
import torch.nn as nn


class Moli(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


moli = Moli()
x = torch.tensor(1.0)
output = moli(x)
print(output)
```

这个例子说明了 4 件最核心的事：

1. 自定义模块要继承 `nn.Module`
2. 在 `__init__()` 里先调用 `super().__init__()`
3. 在 `forward()` 中定义前向逻辑
4. 实际调用时写 `model(x)`，而不是 `model.forward(x)`

---

## 3. `super().__init__()` 到底在做什么

很多人会把它理解成“把父类功能搬过来”，这个说法不算错，但不够准确。

更准确的理解是：
`super().__init__()` 会先执行父类 `nn.Module` 的初始化逻辑，把当前对象初始化成一个“合法的 Module”。

也就是先给当前对象搭好底座，让它具备这些基础能力：

- 可以注册参数
- 可以注册子模块
- 可以注册 buffer
- 可以保存/加载状态
- 可以切换训练和推理模式
- 可以使用 `to()` 迁移到不同设备

所以可以记成一句话：

先搭好 `nn.Module` 的底座，再在上面做自定义。

---

## 4. `forward()` 和 `model(x)` 的关系

写的时候是 `forward()`，但真正用的时候通常写：

```python
y = model(x)
```

而不是：

```python
y = model.forward(x)
```

原因是：

- `model(x)` 会调用 `nn.Module.__call__()`
- `__call__()` 内部再去调用 `forward()`
- 同时还能处理 hook、trace、compile 等机制

所以推荐永远把 `forward()` 当成“定义逻辑”的地方，把 `model(x)` 当成“正式调用”的方式。

---

## 5. `nn.Module` 管理的三类重要对象

### 5.1 子模块 `submodule`

当你在 `__init__()` 里写：

```python
self.linear = nn.Linear(10, 20)
```

这个 `linear` 会被自动注册成当前模块的子模块。

好处是：

- `model.to(device)` 时它会一起迁移
- `model.train()` / `eval()` 时它会一起切换
- `state_dict()` 时它里面的参数会被一起保存

---

### 5.2 参数 `Parameter`

参数是需要训练更新的量，比如：

- `Linear.weight`
- `Linear.bias`
- 卷积核参数

如果你想让一个张量被优化器更新，它必须是 `nn.Parameter`，或者被某个 `nn.Module` 间接持有。

例如：

```python
self.weight = nn.Parameter(torch.randn(10, 10))
```

这时它会：

- 出现在 `model.parameters()`
- 参与反向传播
- 被优化器更新
- 被 `state_dict()` 保存

---

### 5.3 缓冲区 `buffer`

buffer 不是可训练参数，但也是模型状态的一部分。

典型例子：

- BatchNorm 的 `running_mean`
- BatchNorm 的 `running_var`
- 固定位置编码

注册方式：

```python
self.register_buffer("pe", pe)
```

buffer 的特点：

- 不会出现在 `model.parameters()` 中
- 不会被优化器更新
- 会跟着 `model.to(device)` 迁移
- 会被 `state_dict()` 保存

这非常适合固定位置编码。
在 [Transformer/PostionalEncoding.py](./Transformer/PostionalEncoding.py) 里就用了这个模式：

```python
self.register_buffer("pe", pe)
```

意思是：

- `pe` 是模型的一部分
- 但不是可训练参数

---

## 6. 普通 Tensor、Parameter、buffer 的区别

### 普通 Tensor

```python
self.x = torch.randn(3, 3)
```

这只是普通属性：

- 不会被优化器管理
- 不一定被 `state_dict()` 保存
- 不一定随 `to()` 自动迁移

---

### `nn.Parameter`

```python
self.w = nn.Parameter(torch.randn(3, 3))
```

这是可训练参数：

- 会被 `model.parameters()` 找到
- 会被优化器更新
- 会进入 `state_dict()`

---

### `register_buffer`

```python
self.register_buffer("mask", mask)
```

这是“非训练但需要保存和迁移的状态”：

- 不会被优化器更新
- 会进入 `state_dict()`
- 会随 `to()` 迁移

---

## 7. 为什么 `model.parameters()` 这么重要

优化器通常这样写：

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

意思是把模型中所有已注册的参数都交给优化器。

所以如果某个张量没有被正确注册为参数，它就不会训练。

这也是为什么下面这种写法常常是错的：

```python
self.w = torch.randn(10, 10)
```

虽然它是张量，但它不是 `Parameter`，优化器看不到它。

---

## 8. `train()` 和 `eval()` 到底在切什么

很多人会混淆：

- `model.train()`
- `model.eval()`
- `torch.no_grad()`

它们不是一回事。

### `model.train()`

把模块切到训练模式。

受影响的典型模块：

- `Dropout`
- `BatchNorm`

例如：

- `Dropout` 训练时会随机丢弃
- `BatchNorm` 训练时使用当前 batch 的统计量

---

### `model.eval()`

把模块切到推理模式。

例如：

- `Dropout` 在推理时不再随机丢弃
- `BatchNorm` 使用训练过程中积累的 running statistics

---

### `torch.no_grad()`

关闭 autograd 的梯度记录。

它和 `eval()` 不一样：

- `eval()` 改变模块行为
- `no_grad()` 改变是否记录计算图

推理时一般两个一起用：

```python
model.eval()
with torch.no_grad():
    y = model(x)
```

---

## 9. `to()` 为什么这么方便

写：

```python
model.to("cuda")
```

就可以把整个模型迁移到 GPU。

这是因为 `nn.Module` 会递归处理：

- 参数
- buffer
- 子模块

所以只要你的对象被正确注册进 `Module` 体系里，就能跟着一起迁移。

---

## 10. `state_dict()` 保存的是什么

`state_dict()` 返回模型状态字典，里面通常包含：

- 所有参数
- 所有持久化 buffer

例如：

```python
torch.save(model.state_dict(), "model.pth")
```

这是 PyTorch 最推荐的保存方式。

对应加载：

```python
model.load_state_dict(torch.load("model.pth"))
```

优点是：

- 更稳
- 更通用
- 不强依赖原始 Python 对象序列化

---

## 11. 一个带参数的简单例子

```python
import torch
import torch.nn as nn


class MyLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)
```

这里：

- `self.linear` 是子模块
- `self.linear.weight` 和 `self.linear.bias` 是参数
- 调用 `model.parameters()` 能拿到这些参数

---

## 12. 一个带 buffer 的简单例子

```python
import torch
import torch.nn as nn


class AddFixedBias(nn.Module):
    def __init__(self):
        super().__init__()
        bias = torch.tensor([1.0, 2.0, 3.0])
        self.register_buffer("bias", bias)

    def forward(self, x):
        return x + self.bias
```

这里：

- `bias` 不是训练参数
- 但它属于模块状态
- 它会保存、加载、迁移设备

---

## 13. 自定义 `nn.Module` 的推荐模板

```python
import torch
import torch.nn as nn


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 定义子模块
        # 2. 定义参数
        # 3. 注册 buffer

    def forward(self, x):
        # 写前向逻辑
        return x
```

一个标准思路是：

1. 在 `__init__()` 里定义结构和状态
2. 在 `forward()` 里写数据如何流动
3. 在外部通过 `model(x)` 调用

---

## 14. 常见坑

### 14.1 忘记写 `super().__init__()`

会导致：

- 参数无法正常注册
- 子模块无法正常注册
- 某些操作直接报错

---

### 14.2 把普通 Tensor 当成参数

错误示例：

```python
self.w = torch.randn(10, 10)
```

这样优化器看不到它。

正确写法：

```python
self.w = nn.Parameter(torch.randn(10, 10))
```

---

### 14.3 该用 buffer 却只写成普通属性

例如固定位置编码、mask、running statistics 这类状态，如果只是：

```python
self.pe = pe
```

可能会出现：

- `to("cuda")` 时没跟过去
- `state_dict()` 里没保存好

更好的方式通常是：

```python
self.register_buffer("pe", pe)
```

---

### 14.4 直接调用 `forward()`

不推荐：

```python
model.forward(x)
```

推荐：

```python
model(x)
```

因为前者可能绕过 `__call__()` 内部的一些框架机制。

---

### 14.5 把 `eval()` 当成“关闭梯度”

这不准确。

- `eval()` 只切模式
- `no_grad()` 才是关闭梯度记录

---

## 15. 结合位置编码模块理解

现在的 [Transformer/PostionalEncoding.py](./Transformer/PostionalEncoding.py) 是一个很好的综合例子：

- 它继承了 `nn.Module`
- 在 `__init__()` 中先 `super().__init__()`
- 定义了 `Dropout` 这种子模块
- 使用 `register_buffer("pe", pe)` 保存固定位置编码
- 在 `forward()` 中完成输入与位置编码的相加

这个模块说明：

并不是只有“有很多卷积层/线性层”的网络才能继承 `nn.Module`。

只要一个对象同时满足：

- 有前向计算逻辑
- 可能有参数或状态
- 希望接入 PyTorch 的训练/保存/迁移体系

它就很适合写成 `nn.Module`。

---

## 16. 最后一句话总结

可以把 `nn.Module` 记成：

“一个既会算，又会管理自己状态的神经网络组件基类。”

如果再精炼一点：

“`forward()` 负责怎么算，`nn.Module` 负责把这个计算单元纳入 PyTorch 的完整训练体系。”
