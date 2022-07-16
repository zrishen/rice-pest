# -- coding: utf-8 --
# @Author: zrs
# @Time: 2022-07-16 16:39
# @File: SimpleCNN.py

import torch
from torch import nn
import torch.nn.functional as F
from CBMA import CBAM

kernel_size = 3
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__() #224*224*3
        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(16, 24, kernel_size=kernel_size)
        self.conv3 = nn.Conv2d(24, 32, kernel_size=kernel_size)
        self.conv4 = nn.Conv2d(32, 48, kernel_size=kernel_size)
        self.conv5 = nn.Conv2d(48, 64, kernel_size=kernel_size)
        self.pooling = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1600, 128) #64*5*5
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 9)
        self.dpt = nn.Dropout(p=0.3)
        self.batchNor1 = nn.BatchNorm2d(16)
        self.batchNor2 = nn.BatchNorm2d(24)
        self.batchNor3 = nn.BatchNorm2d(32)
        self.batchNor4 = nn.BatchNorm2d(48)
        self.batchNor5 = nn.BatchNorm2d(64)

    def forward(self, x):
        batch_size = x.size(0)  #224*224*3
        # print(batch_size)
        x = F.relu(self.pooling(self.batchNor1(self.conv1(x)))) #111*111*16
        x = F.relu(self.pooling(self.batchNor2(self.conv2(x)))) #54*54*24
        x = F.relu(self.pooling(self.batchNor3(self.conv3(x)))) #26*26*32
        x = F.relu(self.pooling(self.batchNor4(self.conv4(x)))) #12*12*48
        x = F.relu(self.pooling(self.batchNor5(self.conv5(x)))) #5*5*64
        x = x.contiguous().view(batch_size, -1)                              #1600
        x = F.relu(self.dpt(self.fc1(x)))                       #128
        x = F.relu(self.fc2(x))                                 #128
        x = self.fc3(x)                                         #9
        # return F.softmax(x, dim=1)
        return x

class SimpleCNN_SE(nn.Module):
    def __init__(self):
        super(SimpleCNN_SE, self).__init__()
        self.gp = nn.AdaptiveAvgPool2d(1)
        self.feature = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3),
                                nn.BatchNorm2d(16), nn.MaxPool2d(2), nn.ReLU(),
                                nn.Conv2d(16, 24, kernel_size=3),
                                nn.BatchNorm2d(24), nn.MaxPool2d(2), nn.ReLU(),
                                nn.Conv2d(24, 32, kernel_size=3),
                                nn.BatchNorm2d(32), nn.MaxPool2d(2), nn.ReLU(),
                                nn.Conv2d(32, 48, kernel_size=3),
                                nn.BatchNorm2d(48), nn.MaxPool2d(2), nn.ReLU(),
                                nn.Conv2d(48, 64, kernel_size=3),
                                nn.BatchNorm2d(64), nn.MaxPool2d(2), nn.ReLU(),
                                )

        self.fc = nn.Sequential(nn.Linear(1600, 128),
                                nn.Dropout(p=0.3),
                                nn.ReLU(),
                                nn.Linear(128, 128),
                                nn.ReLU(),
                                nn.Linear(128, 17),
                                nn.Softmax(dim=1))

        self.se = nn.Sequential(nn.Linear(64, 64 // 16),
                                nn.ReLU(inplace=True),
                                nn.Linear(64 // 16, 64),
                                nn.Sigmoid())

    def forward(self, input):
        x = input
        x = self.feature(x)
        b, c, _, _ = x.size()
        y = self.gp(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        y = y.view(input.size()[0], -1)
        out = self.fc(y)
        return out

class SimpleCNN_CBMA(nn.Module):
    def __init__(self):
        super(SimpleCNN_CBMA, self).__init__() #224*224*3
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