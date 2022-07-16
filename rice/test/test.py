# -- coding: utf-8 --
# @Author: zrs
# @Time: 2022-03-02 19:09
# @File: test.py

import torch
import torchvision
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
import os
import nets
from datetime import datetime
from time import sleep
from tqdm import tqdm



# 数据增强
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomResizedCrop((224, 224), scale=(0.5, 1.0)),
    transforms.RandomAffine(0, shear=0.2, scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


# 模型保存函数
def save_checkpoint(model, state, filename):
    state['state_dict'] = model.state_dict()
    torch.save(state, filename)

# 测试函数
def test(model, model_path, test_image):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    with torch.no_grad():
        output = model(test_image)

    return output

if __name__ == '__main__':
    # 设置路径
    # 数据集路径
    save_path = r'./data'
    avg_acc = 0.0
    LR = 0.0001
    val_test_batch_size = 2

    time1 = datetime.now()
    # test_set_path = os.path.join(save_path)
    # print(save_path)
    # 设置参数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 创建数据集
    test_image = datasets.ImageFolder(save_path, transform=train_transformer)
    print(test_image)
    # test_set = datasets.ImageFolder(test_set_path, transform=val_transformer)
    # print(test_set)
    # test_loader = DataLoader(test_set, batch_size=val_test_batch_size, drop_last=False, shuffle=False)

    # 创建模型
    model = nets.SE_simpleCNN()
    model.eval()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    # optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 测试
    # print('best_acc')
    model_path = './best_acc_7.pth.tar'
    # predicates_all, label_all = test(model, model_path, test_image)

    # acc = accuracy_score(label_all, predicates_all)
    # 输出accuracy准确率

    print('acc:{:.4f}'.format(acc))
    time2 = datetime.now()
    time = (time2 - time1).seconds
    print("训练总耗时：", time)