'''ResNet in PyTorch - Torchvision compatible for GTSRB.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=43):
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        # CIFAR-style ResNet (3x3 conv1, no maxpool) to match checkpoint
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        # Maxpool is skipped for small images (32x32 or 28x28)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def get_activation_mapping(self):
        """
        Maps weight-bearing modules to their final activation output modules.
        For ResNet, we target the RELU after each block's addition.
        """
        mapping = {
            "conv1": "relu",
            "fc": "fc"
        }
        
        # Map layers in each block
        for i, layer_group in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for j, block in enumerate(layer_group):
                prefix = f"layer{i+1}.{j}"
                mapping[f"{prefix}.conv1"] = f"{prefix}.relu1"
                mapping[f"{prefix}.conv2"] = f"{prefix}.relu2"
                if block.downsample is not None:
                    # Addition point for downsample
                    mapping[f"{prefix}.downsample.0"] = f"{prefix}.relu2"
                    
        return mapping


def ResNet18(num_classes=43):
    """ResNet-18 model compatible with torchvision checkpoints"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
