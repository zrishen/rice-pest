# -- coding: utf-8 --
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
from test import SE_simpleCNN
from test_CBMA import CBMA_SimpleCNN


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

# 训练函数
def train(model, epochs, train_loader, criterion, optimizer):
    train_losses = []
    train_acces = []
    best_acc = 0

    for epoch in tqdm(range(1, epochs+1)):
        # print('%s'%(epoch))
        train_allloss = []
        val_allloss = []
        train_time = datetime.now()
        train_correct = 0
        val_correct = 0

        model.train()
        for batch_samples in train_loader:
            img, label = batch_samples[0].to(device), batch_samples[1].to(device)
            output = model(img)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_allloss.append(loss.item())
            _, pred = output.max(1)
            train_correct += (pred == label).sum()

        train_loss = np.average(train_allloss)
        train_losses.append(train_loss)
        train_correct = train_correct.item()
        train_acc = train_correct / len(train_loader.dataset)
        train_acces.append(float(train_acc))

        time_spent = datetime.now() - train_time
        print('epoch:{}, train_loss:{:.4f}, train_acc:{:.4f},  time_spent:{}'.format(epoch, train_loss, train_acc,
                                                                                     time_spent))
        if train_acc > best_acc:
            best_acc = train_acc
            model_path = save_path + r'/best_acc_' + str(i) + '.pth.tar'
            # print("\n=> found better acc validated model, saving to %s" % model_path)
            save_checkpoint(model, {'epoch': epoch, 'best_acc': best_acc}, model_path)
    return train_losses, train_acces

# 测试函数
def test(model, model_path, test_loader, criterion):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    test_loss = 0
    correct = 0
    predicates_all, label_all = [], []

    with torch.no_grad():
        for batch_samples in tqdm(test_loader):
            img, label = batch_samples[0].to(device), batch_samples[1].to(device)
            output = model(img)
            loss = criterion(output, label)
            test_loss += (loss * img.shape[0]).item()
            _, pred = output.max(1)
            correct += (pred == label).sum()
            predicates_all += pred.cpu().numpy().tolist()
            label_all += label.cpu().numpy().tolist()

    test_loss /= len(test_loader.dataset)
    correct = correct.item()
    test_acc = correct / len(test_loader.dataset)
    print('test--> Avg    loss:{:.4f}    acc:{:.4f}'.format(test_loss, test_acc))
    return predicates_all, label_all

if __name__ == '__main__':
    # 设置路径
    # 数据集路径
    data_path = r'D:\Programs\DeepLearning\rice\dataset\FOLD_9'
    save_path = r'.\result\test'
    avg_acc = 0.0
    result =[]

    time1 = datetime.now()
    for i in range(10):
        train_set_path = os.path.join(data_path,str(i),'train')
        test_set_path = os.path.join(data_path, str(i), 'test')
        # 设置参数
        LR = 0.0001  # 学习率
        epochs = 100  # 训练轮数
        train_batch_size = 2  # 批大小
        val_test_batch_size = 2
        # print(torch.cuda.is_available())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 创建数据集
        train_set = datasets.ImageFolder(train_set_path, transform=train_transformer)
        test_set = datasets.ImageFolder(test_set_path, transform=val_transformer)

        train_loader = DataLoader(train_set, batch_size=train_batch_size, drop_last=False, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=val_test_batch_size, drop_last=False, shuffle=False)

        # 创建模型
        model = nets.Resnet152_RICE()
        # model = nets.SimpleCNN()
        # checkpoint = torch.load('./result/class_9_tow/best_acc_4.pth.tar')
        # model.load_state_dict(checkpoint['state_dict'])
        # model.fc3 = nn.Linear(128, 9)
        model.eval()

        #冻结卷积
        # for para in model.conv1.parameters():
        #     para.requires_grad = False
        # for para in model.conv2.parameters():
        #     para.requires_grad = False
        # for para in model.conv3.parameters():
        #     para.requires_grad = False
        # for para in model.conv4.parameters():
        #     para.requires_grad = False
        # for para in model.conv5.parameters():
        #     para.requires_grad = False
        # for para in model.features.parameters():
        #     para.requires_grad = False;
        model.to(device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        # optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # time1 = datetime.now()  # 获取当前时间

        # 训练
        train_losses, train_acces = train(model, epochs, train_loader, criterion, optimizer)
        train_losses, train_acces = np.array(train_losses), np.array(train_acces)

        # 绘制准确率曲线和损失曲线
        plt.figure()
        plt.plot(train_acces)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='lower right')
        plt.savefig(save_path+'/acc_'+str(i)+'.jpg')
        plt.show()

        plt.figure()
        plt.plot(train_losses)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper right')
        plt.savefig(save_path+'/loss_'+str(i)+'.jpg')
        plt.show()


        # 测试
        # print('best_acc')
        model_path = save_path+'/best_acc_'+str(i)+'.pth.tar'
        predicates_all, label_all = test(model, model_path, test_loader, criterion)
        # print(label_all, predicates_all)

        acc = accuracy_score(label_all, predicates_all)
        # 输出accuracy准确率
        avg_acc += 0.1 * acc
        result.append(acc)
        print('kfold_{},acc:{:.4f}'.format(i,acc))

    print('avg_acc:{:.4f}'.format(avg_acc))
    for i in range(10):
        print('{},acc:{:.4f}'.format(i,result[i]))
    time2 = datetime.now()
    time = (time2 - time1).seconds
    print("训练总耗时：", time)