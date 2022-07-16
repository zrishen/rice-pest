import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
import torch.nn.functional as F

class SE_simpleCNN(nn.Module):
    def __init__(self):
        super(SE_simpleCNN, self).__init__()
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