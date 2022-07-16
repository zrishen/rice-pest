# -- coding: utf-8 --
# @Author: zrs
# @Time: 2022-03-04 12:57
# @File: test_KD.py

import nets
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn, optim
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(0, shear=0.2, scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

test_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# 模型保存函数
def save_checkpoint(model, state, filename):
    state['state_dict'] = model.state_dict()
    torch.save(state, filename)

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

# 训练函数
def train(student_model, teacher_model, epochs, T, soft_ratio, learn_loader, train_loader, criterion, optimizer_train, optimizer_learn):
    train_losses = []
    train_acces = []
    best_acc = 0

    for epoch in range(1, epochs+1):
        train_allloss = []
        train_time = datetime.now()
        train_correct = 0

        student_model.train()
        teacher_model.train()
        for batch_samples, batch_learn in zip(train_loader, learn_loader):
            img_learn, lable_learn = batch_learn[0].to(device), batch_learn[1].to(device)
            img_train, label_train = batch_samples[0].to(device), batch_samples[1].to(device)

            # for para in student_model.parameters():
            #     para.requires_grad = True
            # for para in teacher_model.parameters():
            #     para.requires_grad = True

            output_train = student_model(img_train)
            output_learn = teacher_model(img_learn)

            KL = nn.KLDivLoss(reduction='batchmean')
            loss_soft = KL(F.log_softmax(output_train/7, dim=1), F.softmax(output_learn/7, dim=1))
            loss_hard = criterion(F.softmax(output_train, dim=1), label_train)
            # loss_learn = criterion(F.softmax(output_learn, dim=1), lable_learn)
            loss = soft_ratio * loss_soft + (1 - soft_ratio) * loss_hard


            optimizer_train.zero_grad()
            loss.backward()
            optimizer_train.step()

            # optimizer_learn.zero_grad()
            # loss_learn.backward()
            # optimizer_learn.step()

            train_allloss.append(loss.item())
            _, pred = output_train.max(1)
            train_correct += (pred == label_train).sum()

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
            save_checkpoint(student_model, {'epoch': epoch, 'best_acc': best_acc}, model_path)
    return train_losses, train_acces

if __name__ == '__main__':
    precise_image_path = r'D:\Programs\DeepLearning\rice\dataset\FOLD_9'
    vague_image_path = r'D:\Programs\DeepLearning\模糊图片\data\flod_9_dim5'
    save_path = r'.\result\class_9_DK'

    result = []
    T = 8
    avg_acc = 0
    LR = 0.0001
    epochs = 100
    soft_ratio = 0.7
    train_batch_size = 16
    test_batch_size = 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    start_time = datetime.now()
    for i in range(10):
        # 路径
        precise_train_set_path = os.path.join(precise_image_path, str(i), 'train')
        vague_train_set_path = os.path.join(vague_image_path, str(i), 'train')
        precise_test_set_path = os.path.join(precise_image_path, str(i), 'test')
        vague_test_set_path = os.path.join(vague_image_path, str(i), 'test')

        # 数据集
        precise_train_set = datasets.ImageFolder(precise_train_set_path, transform=train_transformer)
        vague_train_set = datasets.ImageFolder(vague_train_set_path, transform=train_transformer)
        precise_test_set = datasets.ImageFolder(precise_test_set_path, transform=test_transformer)
        vague_test_set = datasets.ImageFolder(vague_test_set_path, transform=test_transformer)

        # 数据迭代器
        precise_train_loader = DataLoader(precise_train_set, batch_size=train_batch_size, drop_last=False, shuffle=True)
        vague_train_loader = DataLoader(vague_train_set, batch_size=train_batch_size, drop_last=False, shuffle=True)
        precise_test_loader = DataLoader(precise_test_set, batch_size=test_batch_size, drop_last=False, shuffle=True)
        vague_test_loader = DataLoader(vague_test_set, batch_size=test_batch_size, drop_last=False, shuffle=True)

        # 模型
        student_model = nets.SimpleCNN()
        teacher_model = nets.GoogleNet_RICE()
        student_model.to(device)
        teacher_model.to(device)

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer_train = optim.Adam(student_model.parameters(), lr=LR)
        optimizer_learn = optim.Adam(teacher_model.parameters(), lr=LR)

        # 训练
        train_losses, train_acces = train(student_model, teacher_model, epochs, T, soft_ratio,
                                          precise_train_loader, vague_train_loader, criterion, optimizer_train, optimizer_learn)
        train_losses, train_acces = np.array(train_losses), np.array(train_acces)

        # 绘制准确率曲线和损失曲线
        plt.figure()
        plt.plot(train_acces)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='lower right')
        plt.savefig(save_path + '/acc_' + str(i) + '.jpg')
        # plt.show()

        plt.figure()
        plt.plot(train_losses)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper right')
        plt.savefig(save_path + '/loss_' + str(i) + '.jpg')
        # plt.show()

     # 测试
        # print('best_acc')
        model_path = save_path+'/best_acc_'+str(i)+'.pth.tar'
        predicates_all, label_all = test(student_model, model_path, vague_test_loader, criterion)
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
    time = (time2 - start_time).seconds
    print("训练总耗时：", time)