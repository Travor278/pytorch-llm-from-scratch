#%%
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

try:
    from .paths import HYMENOPTERA_TRAIN_ROOT, log_dir
except ImportError:
    from paths import HYMENOPTERA_TRAIN_ROOT, log_dir
# %%
# python的用法 -> tensor数据类型
# 通过 transforms.toTensor去看两个问题
# 2. 为什么我们要Tensor数据类型
img_path = HYMENOPTERA_TRAIN_ROOT / "ants_image" / "0013035.jpg"
img = Image.open(img_path)
print(img)
writer = SummaryWriter(log_dir("logs"))
# %%
# 1. transforms该如何使用，输入是什么，输出是什么(python)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# %%
print(tensor_img)
# %%
writer.add_image("tensor_img", tensor_img)
writer.close()
