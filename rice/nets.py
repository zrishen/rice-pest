# -- coding: utf-8 --
import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
import torch.nn.functional as F


pthfile = r"./pretrain_models"

# 加载VGG16
class VGG16_RICE(nn.Module):
    def __init__(self):
        super(VGG16_RICE, self).__init__()
        net = models.vgg16(pretrained=False)
        # net.load_state_dict(torch.load(pthfile + "/vgg16.pth"))
        num_input = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(num_input, 2)
        self.features = net.features
        self.avgpool = net.avgpool
        self.classifier = net.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return torch.sigmoid(x)


# 加载VGG19
class VGG19_RICE(nn.Module):
    def __init__(self):
        super(VGG19_RICE, self).__init__()
        net = models.vgg19(pretrained=False)
        # net.load_state_dict(torch.load(pthfile + "\\vgg19.pth"))
        num_input = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(num_input, 9)
        self.features = net.features
        self.avgpool = net.avgpool
        self.classifier = net.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return torch.sigmoid(x)


# 加载Resnet18
class Resnet18_RICE(nn.Module):
    def __init__(self):
        super(Resnet18_RICE, self).__init__()
        net = models.resnet18(pretrained=False)
        # net.load_state_dict(torch.load(pthfile + "/resnet18.pth"))
        num_input = net.fc.in_features
        net.fc = nn.Linear(num_input, 2)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', net.conv1),
            ('bn1', net.bn1),
            ('relu', net.relu),
            ('maxpool', net.maxpool),
            ('layer1', net.layer1),
            ('layer2', net.layer2),
            ('layer3', net.layer3),
            ('layer4', net.layer4),
        ]))
        self.avgpool = net.avgpool
        self.fc = net.fc

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return torch.sigmoid(x)


# 加载Resnet34
class Resnet34_RICE(nn.Module):
    def __init__(self):
        super(Resnet34_RICE, self).__init__()
        net = models.resnet34(pretrained=False)
        net.load_state_dict(torch.load(pthfile + "/resnet34.pth"))
        num_input = net.fc.in_features
        net.fc = nn.Linear(num_input, 2)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', net.conv1),
            ('bn1', net.bn1),
            ('relu', net.relu),
            ('maxpool', net.maxpool),
            ('layer1', net.layer1),
            ('layer2', net.layer2),
            ('layer3', net.layer3),
            ('layer4', net.layer4),
        ]))
        self.avgpool = net.avgpool
        self.fc = net.fc

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return torch.sigmoid(x)


# 加载Resnet50
class Resnet50_RICE(nn.Module):
    def __init__(self):
        super(Resnet50_RICE, self).__init__()
        net = models.resnet50(pretrained=False)
        net.load_state_dict(torch.load(pthfile + "/resnet50.pth"))
        num_input = net.fc.in_features
        net.fc = nn.Linear(num_input, 2)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', net.conv1),
            ('bn1', net.bn1),
            ('relu', net.relu),
            ('maxpool', net.maxpool),
            ('layer1', net.layer1),
            ('layer2', net.layer2),
            ('layer3', net.layer3),
            ('layer4', net.layer4),
        ]))
        self.avgpool = net.avgpool
        self.fc = net.fc

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return torch.sigmoid(x)


# 加载Resnet152
class Resnet152_RICE(nn.Module):
    def __init__(self):
        super(Resnet152_RICE, self).__init__()
        net = models.resnet152(pretrained=False)
        # net.load_state_dict(torch.load(pthfile + "/resnet152.pth"))
        num_input = net.fc.in_features
        net.fc = nn.Linear(num_input, 2)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', net.conv1),
            ('bn1', net.bn1),
            ('relu', net.relu),
            ('maxpool', net.maxpool),
            ('layer1', net.layer1),
            ('layer2', net.layer2),
            ('layer3', net.layer3),
            ('layer4', net.layer4),
        ]))
        self.avgpool = net.avgpool
        self.fc = net.fc

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return torch.sigmoid(x)


# 加载AlexNet
class AlexNet_RICE(nn.Module):
    def __init__(self):
        super(AlexNet_RICE, self).__init__()
        net = models.alexnet(pretrained=True)
        # net.load_state_dict(torch.load(pthfile + "/alexnet.pth"))
        num_input = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(num_input, 5)
        self.features = net.features
        self.avgpool = net.avgpool
        self.classifier = net.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return torch.sigmoid(x)


# 加载GoogleNet
class GoogleNet_RICE(nn.Module):
    def __init__(self):
        super(GoogleNet_RICE, self).__init__()
        net = models.googlenet(pretrained=True)
        # net.load_state_dict(torch.load(pthfile + "/googlenet.pth"))
        num_input = net.fc.in_features
        net.fc = nn.Linear(num_input, 9)
        self.googlenet_out = net

    def forward(self, x):
        x = self.googlenet_out(x)
        return x


# 加载densenet121
class densenet121_RICE(nn.Module):
    def __init__(self):
        super(densenet121_RICE, self).__init__()
        net = models.densenet121(pretrained=True)
        # net.load_state_dict(torch.load(pthfile + "\\densenet121.pth"))
        num_input = net.classifier.in_features
        net.classifier = nn.Linear(num_input, 2)
        self.features = net.features
        self.classifier = net.classifier

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return torch.sigmoid(out)


# 加载densenet169
class densenet169_RICE(nn.Module):
    def __init__(self):
        super(densenet169_RICE, self).__init__()
        net = models.densenet169(pretrained=True)
        # net.load_state_dict(torch.load(pthfile + "\\densenet169.pth"))
        num_input = net.classifier.in_features
        net.classifier = nn.Linear(num_input, 9)
        self.features = net.features
        self.classifier = net.classifier

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class inceptionv3_RICE(nn.Module):
    def __init__(self):
        super(inceptionv3_RICE, self).__init__()
        net = models.inception_v3(pretrained=False)
        # net.load_state_dict(torch.load(pthfile + "/inceptionv3.pth"))
        num_input = net.fc.in_features
        net.fc = nn.Linear(num_input, 5)
        self.inception_out = net

    def forward(self, x):
        out = self.inception_out(x)
        return torch.sigmoid(out[0])


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

if __name__ == '__main__':
    # net = Resnet18_COVID()
    net = models.vgg19(pretrained=False)
    print(net)
    # for param in net.fc.parameters():
    #     print(param.requires_grad)