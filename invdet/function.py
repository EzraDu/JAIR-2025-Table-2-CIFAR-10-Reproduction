import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import ResNet18_Weights, resnet18
from joblib import dump, load
import numpy as np
from tqdm import tqdm  # 导入tqdm
from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


# ======================================== ↓ wrn相关模块 ↓ =============================================
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        out = self.dropout(out)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropout_rate)

    @staticmethod
    def _make_layer(block, in_planes, out_planes, nb_layers, stride, dropout_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropout_rate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropout_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropout_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropout_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, return_features=False, return_intermediate=False):
        intermediate_outputs = []  # 用于存储中间层输出的列表

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        if return_intermediate:
            intermediate_outputs.append(out)

        out = self.relu(self.bn1(out))
        if return_intermediate:
            intermediate_outputs.append(out)

        out = F.adaptive_avg_pool2d(out, (1, 1))
        if return_intermediate:
            intermediate_outputs.append(out)

        features = out.view(-1, self.nChannels)
        out = self.fc(features)

        if return_features:
            return out, features
        elif return_intermediate:
            return out, intermediate_outputs
        else:
            return out


# ========================================  ↑ wrn相关模块 ↑  =============================================


# ========================================= ↓ vgg相关模块 ↓ ==============================================
class VGG(nn.Module):
    def __init__(self, features, num_classes=43, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x, return_intermediate=False):
        intermediate_outputs = []

        x = self.features(x)
        if return_intermediate:
            intermediate_outputs.append(x)
        x = self.avgpool(x)
        if return_intermediate:
            intermediate_outputs.append(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if return_intermediate:
            return x, intermediate_outputs
        else:
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers.extend([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
            else:
                layers.extend([conv2d, nn.ReLU(inplace=True)])
            in_channels = v
    return nn.Sequential(*layers)


cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']


def vgg11_bn(pretrained=False, num_classes=43, device='cpu'):
    model = VGG(make_layers(cfg, batch_norm=True), num_classes=num_classes)
    if pretrained:
        # Load pretrained weights
        pass  # Add your code here to load pretrained weights if necessary
    return model


# ========================================= ↑ vgg相关模块 ↑ ==============================================

# ====================================== ↓ resnet18相关模块 ↓ ============================================
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        original_model = resnet18(weights=weights)
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.flatten(1)


# ====================================== ↑ resnet18相关模块 ↑ ============================================

def extract_features_cifar10(data_loader, train_method, device, reludim, resnetdim, train=True):
    wrn_features_list = []
    res_features_list = []
    labels_list = []

    model_path = './checkpoints/model/cifar10_wrn40_2.pth'

    # 加载教师模型
    wrn402 = WideResNet(depth=40, num_classes=10, widen_factor=2, dropout_rate=0)
    wrn402 = wrn402.to(device)
    wrn402.eval()
    state_dict = torch.load(model_path, map_location=device)['state_dict']
    wrn402.load_state_dict(state_dict)

    # 加载Resnet18提取特征
    resnet18_extractor = ResNetFeatureExtractor().to(device)
    resnet18_extractor.eval()

    if train:
        block3_features_list = []
        relu_features_list = []

        with torch.no_grad():  # 不计算梯度
            for data in tqdm(data_loader, desc="Extracting features"):  # 使用tqdm包裹data_loader
                inputs, targets = data
                inputs = inputs.to(device)
                _, intermediate_outputs = wrn402(inputs, return_intermediate=True)

                # 提取"教师模型"本身特征并添加到对应列表
                relu_features_list.append(intermediate_outputs[1].view(inputs.size(0), -1).cpu().numpy())

                # 提取图形特征并添加到列表
                res_features_list.append(resnet18_extractor(inputs).cpu().numpy())

                # 标记标签并添加到列表
                labels_list.append(targets)  # 直接将整个元组添加到列表中

        # 将列表中的数组合并成一个大的numpy数组
        relu_features = np.concatenate(relu_features_list, axis=0)
        res_features = np.concatenate(res_features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        if reludim < 8192:
            pca_relu = PCA(n_components=reludim)
            relu_features = pca_relu.fit_transform(relu_features)
            dump(pca_relu, f"./checkpoints/cifar10/pca/[{train_method}]relu_{reludim}.joblib")
        elif reludim == 8192:
            print("Modelfeature与设定值一致，不需要进行降维")
        else:
            raise ValueError("设定Modelfeature维度错误！")


        if resnetdim < 512:
            pca_resnet18 = PCA(n_components=resnetdim)
            res_features = pca_resnet18.fit_transform(res_features)
            dump(pca_resnet18, f"./checkpoints/cifar10/pca/[{train_method}]resnet_{resnetdim}.joblib")
        elif resnetdim == 512:
            print("Resnetfeature与设定值一致，不需要进行降维")
        else:
            raise ValueError("设定Resnetfeature维度错误！")


        wrn_features = relu_features

    else:
        # 加载预训练的PCA模型
        if reludim < 8192:
            pca_relu = load(f"./checkpoints/cifar10/pca/[{train_method}]relu_{reludim}.joblib")
        elif reludim == 8192:
            print("Modelfeature与设定值一致，不需要进行降维")
        else:
            raise ValueError("设定Modelfeature维度错误！")

        if resnetdim < 512:
            pca_resnet18 = load(f"./checkpoints/cifar10/pca/[{train_method}]resnet_{resnetdim}.joblib")
        elif resnetdim == 512:
            print("Resnetfeature与设定值一致，不需要进行降维")
        else:
            raise ValueError("设定Resnetfeature维度错误！")

        with torch.no_grad():  # 不计算梯度
            for data in tqdm(data_loader, desc="Extracting features"):  # 使用tqdm包裹data_loader
                inputs, targets = data
                inputs = inputs.to(device)
                _, intermediate_outputs = wrn402(inputs, return_intermediate=True)

                # 提取"教师模型"本身特征并通过PCA降维
                relu_features = intermediate_outputs[1].view(inputs.size(0), -1).cpu().numpy()
                if reludim < 8192:
                    relu_features = pca_relu.transform(relu_features)

                # 提取图形特征并通过PCA降维
                res_feature = resnet18_extractor(inputs).cpu().numpy()
                if resnetdim < 512:
                    res_feature = pca_resnet18.transform(res_feature)

                # 将特征添加到列表
                wrn_features_list.append(np.concatenate([relu_features], axis=1))
                res_features_list.append(res_feature)

                # 标记标签并添加到列表
                labels_list.append(targets)  # 直接将整个元组添加到列表中

        # 将列表中的数组合并成一个大的numpy数组
        wrn_features = np.concatenate(wrn_features_list, axis=0)
        res_features = np.concatenate(res_features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

    return wrn_features, res_features, labels

def extract_features_gtsrb(data_loader, train_method, device, featuredim, resnetdim, train=True):
    vgg_features_list = []
    res_features_list = []
    labels_list = []

    model_path = './checkpoints/model/saved_epoch-15.tar'
    # 加载教师模型
    vgg11 = vgg11_bn(pretrained=False, num_classes=43, device=device)
    vgg11.to(device)
    # 加载训练权重
    state_dict = torch.load(model_path, map_location=device)['model_state']
    vgg11.load_state_dict(state_dict)
    # 设置为评估模式
    vgg11.eval()

    # 加载Resnet18提取特征
    resnet18_extractor = ResNetFeatureExtractor().to(device)
    resnet18_extractor.eval()

    if train:
        conv_features_list = []

        with torch.no_grad():  # 不计算梯度
            for data in tqdm(data_loader, desc="Extracting features"):  # 使用tqdm包裹data_loader
                inputs, targets = data
                inputs = inputs.to(device)
                _, intermediate_outputs = vgg11(inputs, return_intermediate=True)

                # 提取"教师模型"本身特征并添加到对应列表
                conv_features_list.append(intermediate_outputs[0].view(inputs.size(0), -1).cpu().numpy())

                # 提取图形特征并添加到列表
                res_features_list.append(resnet18_extractor(inputs).cpu().numpy())

                # 标记标签并添加到列表
                labels_list.append(targets)  # 直接将整个元组添加到列表中

        # 将列表中的数组合并成一个大的numpy数组
        conv_features = np.concatenate(conv_features_list, axis=0)
        res_features = np.concatenate(res_features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        if featuredim < 512:
            pca_conv = PCA(n_components=featuredim)
            conv_features = pca_conv.fit_transform(conv_features)
            dump(pca_conv, f"./checkpoints/gtsrb/pca/[{train_method}]feature_{featuredim}.joblib")
        elif featuredim == 512:
            print("Modelfeature与设定值一致，不需要进行降维")
        else:
            raise ValueError("设定Modelfeature维度错误！")

        if resnetdim < 512:
            pca_resnet18 = PCA(n_components=resnetdim)
            res_features = pca_resnet18.fit_transform(res_features)
            dump(pca_resnet18, f"./checkpoints/gtsrb/pca/[{train_method}]resnet_{resnetdim}.joblib")
        elif resnetdim == 512:
            print("Resnetfeature与设定值一致，不需要进行降维")
        else:
            raise ValueError("设定Resnetfeature维度错误！")

        vgg_features = conv_features

    else:
        if featuredim < 512:
            pca_conv = load(f"./checkpoints/gtsrb/pca/[{train_method}]feature_{featuredim}.joblib")
        elif featuredim == 512:
            print("Modelfeature与设定值一致，不需要进行降维")
        else:
            raise ValueError("设定Modelfeature维度错误！")

        if resnetdim < 512:
            pca_resnet18 = load(f"./checkpoints/gtsrb/pca/[{train_method}]resnet_{resnetdim}.joblib")
        elif resnetdim == 512:
            print("Resnetfeature与设定值一致，不需要进行降维")
        else:
            raise ValueError("设定Resnetfeature维度错误！")

        with torch.no_grad():  # 不计算梯度
            for data in tqdm(data_loader, desc="Extracting features"):  # 使用tqdm包裹data_loader
                inputs, targets = data
                inputs = inputs.to(device)
                _, intermediate_outputs = vgg11(inputs, return_intermediate=True)

                # 提取"教师模型"本身特征并通过PCA降维
                conv_features = intermediate_outputs[0].view(inputs.size(0), -1).cpu().numpy()
                if featuredim < 512:
                    conv_features = pca_conv.transform(conv_features)

                res_feature = resnet18_extractor(inputs).cpu().numpy()
                if resnetdim < 512:
                    res_feature = pca_resnet18.transform(res_feature)

                # 将特征添加到列表
                vgg_features_list.append(conv_features)
                res_features_list.append(res_feature)

                # 标记标签并添加到列表
                labels_list.append(targets)  # 直接将整个元组添加到列表中

        # 将列表中的数组合并成一个大的numpy数组
        vgg_features = np.concatenate(vgg_features_list, axis=0)
        res_features = np.concatenate(res_features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

    return vgg_features, res_features, labels