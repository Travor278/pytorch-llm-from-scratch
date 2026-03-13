#%% # 线性回归从零开始实现
import torch
from torch.utils import data
# 'nn'是神经网络模块的缩写，包含了构建神经网络的各种工具和函数
from torch import nn

# 替代 d2l.synthetic_data
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
#%%
def load_array(data_arrays, batch_size, is_train=True):
    """construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train) # 这里的shuffle参数是为了在训练时打乱数据顺序，增加模型的泛化能力

batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))
# %%
# 定义模型
net = nn.Sequential(nn.Linear(2, 1))
# %%
# 初始化模型参数
net[0].weight.data.normal_(0, 0.01) # 使用正态分布初始化权重，均值为0，标准差为0.01
net[0].bias.data.fill_(0) # 将偏置初始化为0
# %%
# 损失函数
loss = nn.MSELoss() # 均方误差损失函数，适用于回归问题，计算预测值与真实值之间的平均平方差
# %%
# 优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03) # 随机梯度下降优化算法，net.parameters()返回模型的所有参数，lr是学习率，控制每次更新的步长
# %%
# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        loss_val = loss(net(X), y)
        trainer.zero_grad()
        loss_val.backward()
        trainer.step()
    loss_val = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {loss_val:f}')
# %%
print(f'w的估计误差: {true_w - net[0].weight.data.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - net[0].bias.data}')