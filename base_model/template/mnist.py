from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image
import os

# 下载 MNIST
mnist = datasets.MNIST(root="./data", train=True, download=True)

# 取第 0 张图片和标签
img, label = mnist[5]
print("Label:", label)

# 保存成 png
os.makedirs("mnist_png", exist_ok=True)
img.save(f"mnist_png/sample_{label}.png")
