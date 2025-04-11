import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from networks import MixedFeatureNet
from torch.nn import Module
import os

num_head = 2



class ELA(nn.Module):
    def __init__(self, in_channels, phi):
        super(ELA, self).__init__()
        '''
        ELA-T 和 ELA-B 设计为轻量级，非常适合网络层数较少或轻量级网络的 CNN 架构
        ELA-B 和 ELA-S 在具有更深结构的网络上表现最佳
        ELA-L 特别适合大型网络。
        '''
        Kernel_size = {'T': 5, 'B': 7, 'S': 5, 'L': 7}[phi]
        groups = {'T': in_channels, 'B': in_channels, 'S': in_channels // 8, 'L': in_channels // 8}[phi]
        num_groups = {'T': 32, 'B': 16, 'S': 16, 'L': 16}[phi]
        pad = Kernel_size // 2
        self.con1 = nn.Conv1d(in_channels, in_channels, kernel_size=Kernel_size, padding=pad, groups=groups, bias=False)
        self.GN = nn.GroupNorm(num_groups, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        b, c, h, w = input.size()
        x_h = torch.mean(input, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(input, dim=2, keepdim=True).view(b, c, w)
        x_h = self.con1(x_h)  # [b,c,h]
        x_w = self.con1(x_w)  # [b,c,w]
        x_h = self.sigmoid(self.GN(x_h)).view(b, c, h, 1)  # [b, c, h, 1]
        x_w = self.sigmoid(self.GN(x_w)).view(b, c, 1, w)  # [b, c, 1, w]
        return x_h * x_w * input


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class MHAN(nn.Module):
    def __init__(self, num_class=7, num_head=2, pretrained=True):
        super(MHAN, self).__init__()

        net = MixedFeatureNet.MixedFeatureNet()

        if pretrained:
            net = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"))

        self.features = nn.Sequential(*list(net.children())[:-4])
        self.num_head = num_head
        for i in range(int(num_head)):
            setattr(self, "cat_head%d" % (i), MHAtt())
        self.ela = ELA(in_channels=512, phi='T')
        self.Linear = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.flatten = Flatten()
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)
        self.output = nn.Sequential(nn.Dropout(0.1), Flatten())
        self.classifier = nn.Linear(512, num_class)

    def forward(self, x):
        x = self.features(x)
        x = self.ela(x)
        heads = []

        for i in range(self.num_head):
            heads.append(getattr(self, "cat_head%d" % i)(x))
        head_out = heads

        y = heads[0]

        for i in range(1, self.num_head):
            y = torch.max(y, heads[i])

        y = x * y
        feature_map = F.avg_pool2d(y, y.size()[2:])
        feature = self.output(feature_map)
        feature = F.normalize(feature, dim=1)
        out = self.classifier(feature)
        # 打印输出的形状
        # print("Feature shape:", feature.shape)
        # print("Output shape:", out.shape)

        return out, feature, head_out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class MHAtt(nn.Module):
    def __init__(self):
        super().__init__()
        self.SEDDAHead = SEDDA(512, 512)

    def forward(self, x):
        ca = self.SEDDAHead(x)
        return ca


class SEDDA(nn.Module):
    def __init__(self, inp, oup, groups=32, reduction=16):
        super(SEDDA, self).__init__()

        self.Linear_h = Linear_block(inp, inp, groups=inp, kernel=(1, 7), stride=(1, 1), padding=(0, 0))
        self.Linear_w = Linear_block(inp, inp, groups=inp, kernel=(7, 1), stride=(1, 1), padding=(0, 0))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()
        self.Linear = Linear_block(oup, oup, groups=oup, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.flatten = Flatten()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, oup // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(oup // reduction, oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.Linear_h(x)
        x_w = self.Linear_w(x)
        x_w = x_w.permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        z = x_w * x_h
        b, c, _, _ = z.size()
        y = self.avg_pool(z).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return z * y.expand_as(z)