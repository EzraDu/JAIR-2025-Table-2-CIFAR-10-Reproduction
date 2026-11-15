from torchvision.datasets import ImageFolder, CIFAR10, GTSRB
from torch.utils.data import Dataset
from torchvision import datasets, transforms as T

from PIL import Image, ImageOps
import io

import torch

class AddGaussianNoise(object):
    """向图像添加高斯噪声的转换"""
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'


class ApplyJPEGCompression(object):
    def __init__(self, quality=75):
        self.quality = quality  # JPEG质量参数，范围从1（最差质量，最高压缩）到100（最佳质量，最低压缩）

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError('Input img should be PIL Image. Got {}'.format(type(img)))

        # 将PIL图像转换为JPEG并重新加载，以模拟JPEG压缩的效果
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer)

    def __repr__(self):
        return self.__class__.__name__ + '(quality={})'.format(self.quality)

def get_dataset(name: str, method: str, train=True):
    # 数据集名称和方法均转为小写
    method = method.lower()
    name = name.lower()

    if name not in ['cifar10', 'gtsrb']:
        raise ValueError(f"Unsupported dataset: {name}")

    # 根据数据集名称选择transform
    if name == 'cifar10':
        transform = {
            'train': T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ]),
            'test': T.Compose([
                T.Resize((32, 32)),
                # T.GaussianBlur(kernel_size=(5, 5), sigma=(0.2, 1.4)),
                # ApplyJPEGCompression(quality=25),  # 应用JPEG压缩，质量参数可以根据需要调整
                T.ToTensor(),
                # AddGaussianNoise(0, 0.08),
                T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ])
        }
    elif name == 'gtsrb':
        transform = {
            'train': T.Compose([
                T.Resize((48, 48), antialias=True),
                T.ToTensor(),
                T.Normalize(mean=[0.34169674, 0.31255573, 0.3215503], std=[0.28010625, 0.26819786, 0.2746381])
            ]),
            'test': T.Compose([
                T.Resize((48, 48), antialias=True),
                # 注释的部分为测试鲁棒性时使用
                # T.GaussianBlur(kernel_size=(5, 5), sigma=(0.2, 1.4)),
                # ApplyJPEGCompression(quality=25),  # 应用JPEG压缩，质量参数可以根据需要调整
                T.ToTensor(),
                # AddGaussianNoise(0, 0.08),
                T.Normalize(mean=[0.34169674, 0.31255573, 0.3215503], std=[0.28010625, 0.26819786, 0.2746381])
            ])
        }

    # 设置transform
    current_transform = transform['train'] if train else transform['test']

    # 根据method处理
    if method == 'noinv':
        if name == 'cifar10':
            dataset = CIFAR10(root='./OriginalData/CIFAR10', train=train, download=True, transform=current_transform,
                              target_transform=lambda x: "Ori")
        elif name == 'gtsrb':
            dataset = GTSRB(root='./OriginalData', split='test', transform=current_transform, download=True,
                            target_transform=lambda x: "Ori")
        else:
            raise NotImplementedError('数据集未定义！')

    elif method in ['cmi', 'dafl', 'deepinv', 'dfq', 'ipr', 'zskt']:
        dataset_path = f'./Invdata/{name}/{method}'
        dataset = ImageFolder(root=dataset_path, transform=current_transform, target_transform=lambda x: "Inv")
    else:
        raise NotImplementedError('逆向方法未实现！')

    return dataset

# name：指定数据集的名称。
# train：标识是否为训练模式。
# method：指定数据处理方法，若传递 noinv 则加载原始数据集。
