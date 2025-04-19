#!/usr/bin/env python3
"""
Lightweight model zoo for Secure‑PoL‑Watermarking experiments
-------------------------------------------------------------
• CIFAR‑10 ResNets (resnet20 … resnet1202)
• CIFAR‑100 / ImageNet‑style ResNets (resnet18 … resnet152)
• Two toy CNNs (SimpleCNN / SimpleConv)

All sub‑modules are initialised with Kaiming‑normal weights **after**
they have been registered (important!).
"""

from __future__ import annotations
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)

# --------------------------------------------------------------------- #
#                    generic weight initialiser                         #
# --------------------------------------------------------------------- #
def _weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / fan_in**0.5
            init.uniform_(m.bias, -bound, bound)
        logging.debug("Init %s", m.__class__.__name__)

# --------------------------------------------------------------------- #
#                          simple CNNs                                  #
# --------------------------------------------------------------------- #
class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.apply(_weights_init)          # <-- ensure init

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SimpleConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool  = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1   = nn.Linear(64 * 5 * 5, 128)
        self.fc2   = nn.Linear(128, 10)
        self.apply(_weights_init)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --------------------------------------------------------------------- #
#                    CIFAR‑10 narrow ResNet (He et al.)                 #
# --------------------------------------------------------------------- #
class LambdaLayer(nn.Module):
    def __init__(self, lambd): super().__init__(); self.lambd = lambd
    def forward(self, x): return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, option="A"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != planes:
            self.shortcut = (
                LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2],
                                            (0, 0, 0, 0, planes // 4, planes // 4)))
                if option == "A" else
                nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                    nn.BatchNorm2d(planes),
                )
            )
        else:
            self.shortcut = nn.Identity()
        self.apply(_weights_init)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, layers[0], 1)
        self.layer2 = self._make_layer(block, 32, layers[1], 2)
        self.layer3 = self._make_layer(block, 64, layers[2], 2)
        self.fc     = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, n_blocks, stride):
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes
        layers += [block(self.in_planes, planes) for _ in range(n_blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = F.avg_pool2d(x, x.size(3)).view(x.size(0), -1)
        return self.fc(x)

def resnet20():   return ResNet(BasicBlock, [3, 3, 3])
def resnet32():   return ResNet(BasicBlock, [5, 5, 5])
def resnet44():   return ResNet(BasicBlock, [7, 7, 7])
def resnet56():   return ResNet(BasicBlock, [9, 9, 9])
def resnet110():  return ResNet(BasicBlock, [18, 18, 18])
def resnet1202(): return ResNet(BasicBlock, [200, 200, 200])

# --------------------------------------------------------------------- #
#              Wide/Deep ResNet for CIFAR‑100 / ImageNet                #
# --------------------------------------------------------------------- #
class BasicBlock2(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.short = (
            nn.Identity()
            if stride == 1 and in_ch == out_ch
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))
        )
        self.apply(_weights_init)

    def forward(self, x): return F.relu(self.res(x) + self.short(x))

class BottleNeck2(nn.Module):
    expansion = 4
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),  nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, stride, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch * 4, 1, bias=False), nn.BatchNorm2d(out_ch * 4),
        )
        self.short = (
            nn.Identity()
            if stride == 1 and in_ch == out_ch * 4
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch * 4, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch * 4))
        )
        self.apply(_weights_init)

    def forward(self, x): return F.relu(self.res(x) + self.short(x))

class ResNet2(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super().__init__()
        self.in_ch = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
        )
        self.stage1 = self._make_layer(block,  64, layers[0], 1)
        self.stage2 = self._make_layer(block, 128, layers[1], 2)
        self.stage3 = self._make_layer(block, 256, layers[2], 2)
        self.stage4 = self._make_layer(block, 512, layers[3], 2)
        self.avg    = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(512 * block.expansion, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, out_ch, n_blocks, stride):
        layers = [block(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch * block.expansion
        layers += [block(self.in_ch, out_ch) for _ in range(n_blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.stage2(x)
        x = self.stage3(x); x = self.stage4(x)
        x = self.avg(x).flatten(1)
        return self.fc(x)

# wide family (CIFAR‑100 default num_classes=100)
def resnet18():  return ResNet2(BasicBlock2,  [2, 2, 2, 2])
def resnet34():  return ResNet2(BasicBlock2,  [3, 4, 6, 3])
def resnet50():  return ResNet2(BottleNeck2, [3, 4, 6, 3])
def resnet101(): return ResNet2(BottleNeck2, [3, 4, 23, 3])
def resnet152(): return ResNet2(BottleNeck2, [3, 8, 36, 3])

# --------------------------------------------------------------------- #
__all__ = [n for n in globals() if n.startswith("resnet")] + [
    "SimpleCNN", "SimpleConv", "BasicBlock2", "BottleNeck2"
]
