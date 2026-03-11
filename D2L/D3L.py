#%%
import torch

x = torch.tensor([3.0])
y = torch.tensor([2.0])

x + y, x - y, x * y, x / y, x**y
# %%
x = torch.arange(4.0)
x
# %%
x[3]
# %%
len(x), x.shape
# %%
A = torch.arange(20).reshape(5, 4)
A
# %%
A.T, A.T.shape
# %%
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B, B.sum(), B.sum(axis=0), B.sum(axis=1), B == B.T
# %%
X = torch.arange(24).reshape(2, 3, 4)
X, X.sum(), X.sum(axis=0), X.sum(axis=1), X.sum(axis=2)
# %%
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone() # 通过分配新内存，将A的一个副本分配给B
A, B, A == B, A + B, A - B, A * B, A / (B + 1e-5)
# %%
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
# %%
x = torch.arange(4, dtype=torch.float32)
x, x.sum(), x.sum().item() # item()将单元素张量转换为Python数值
# %%
A.mean(), A.sum() / A.numel() # numel()返回张量中的元素数量
# %%
A.mean(axis=0), A.sum(axis=0) / A.shape[0]