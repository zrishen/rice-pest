from torch import nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__() #224*224*3
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 24, kernel_size=3)
        self.conv3 = nn.Conv2d(24, 32, kernel_size=3)
        self.conv4 = nn.Conv2d(32, 48, kernel_size=3)
        self.conv5 = nn.Conv2d(48, 64, kernel_size=3)
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
        x = F.relu(self.pooling(self.batchNor1(self.conv1(x))))
        x = F.relu(self.pooling(self.batchNor2(self.conv2(x))))
        x = F.relu(self.pooling(self.batchNor3(self.conv3(x))))
        x = F.relu(self.pooling(self.batchNor4(self.conv4(x))))
        x = F.relu(self.pooling(self.batchNor5(self.conv5(x))))
        x = x.contiguous().view(batch_size, -1)
        x = F.relu(self.dpt(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)