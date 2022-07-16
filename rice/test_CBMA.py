# -- coding: utf-8 --
# @Author: zrs
# @Time: 2022-02-05 15:06
# @File: test_CBMA.py

import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
import torch.nn.functional as F

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


kernel_size = 3
class CBMA_SimpleCNN(nn.Module):
    def __init__(self):
        super(CBMA_SimpleCNN, self).__init__() #224*224*3
        self.CBMA1 = CBAM(16)
        self.CBMA2 = CBAM(24)
        self.CBMA3 = CBAM(32)
        self.CBMA4 = CBAM(48)
        self.CBMA5 = CBAM(64)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(16, 24, kernel_size=kernel_size)
        self.conv3 = nn.Conv2d(24, 32, kernel_size=kernel_size)
        self.conv4 = nn.Conv2d(32, 48, kernel_size=kernel_size)
        self.conv5 = nn.Conv2d(48, 64, kernel_size=kernel_size)
        self.pooling = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1600, 128) #64*5*5
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 17)
        self.dpt = nn.Dropout(p=0.3)
        self.batchNor1 = nn.BatchNorm2d(16)
        self.batchNor2 = nn.BatchNorm2d(24)
        self.batchNor3 = nn.BatchNorm2d(32)
        self.batchNor4 = nn.BatchNorm2d(48)
        self.batchNor5 = nn.BatchNorm2d(64)

    def forward(self, x):
        batch_size = x.size(0)  #224*224*3
        # print(batch_size)
        y = x
        x = F.relu(self.pooling(self.batchNor1(self.conv1(x)))) #111*111*16
        x = self.CBMA1(x)
        x = x + y

        y = x
        x = F.relu(self.pooling(self.batchNor2(self.conv2(x)))) #54*54*24
        x = self.CBMA2(x)
        x = x + y

        y = x
        x = F.relu(self.pooling(self.batchNor3(self.conv3(x)))) #26*26*32
        x = self.CBMA3(x)
        x = x + y

        y = x
        x = F.relu(self.pooling(self.batchNor4(self.conv4(x)))) #12*12*48
        x = self.CBMA4(x)
        x = x + y

        y = x
        x = F.relu(self.pooling(self.batchNor5(self.conv5(x)))) #5*5*64
        x = self.CBMA5(x)
        x = x + y

        x = x.view(batch_size, -1)                              #1600
        x = F.relu(self.dpt(self.fc1(x)))                       #128
        x = F.relu(self.fc2(x))                                 #128
        x = self.fc3(x)                                         #9
        return F.softmax(x, dim=1)