import torch
import torchvision
from torch import nn

try:
    from .paths import checkpoint_path
except ImportError:
    from paths import checkpoint_path

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1，模型的结构+参数
torch.save(vgg16, checkpoint_path("vgg16_method1.pth"))

# 保存方式2，模型参数（官方推荐）
torch.save(vgg16.state_dict(), checkpoint_path("vgg16_method2.pth"))

# 陷阱
class Moli(nn.Module):
    def __init__(self):
        super(Moli, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        return self.conv1(x)
    
moli = Moli()
torch.save(moli, checkpoint_path("moli_mothod1.pth"))
