# -- coding: utf-8 --
# @Author: zrs
# @Time: 2022-04-28 16:52
# @File: main.py

import cv2
import numpy as np
import os
import torch
import nets
import GradCAM
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import torch.nn.functional as F

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

if __name__ == '__main__':
    rice_disease_class = ['BLB', 'BPH', 'Brown_Spot', 'False_Smut', 'Healthy_Plant', 'Hispa', 'Neck_Blast', 'Sheath_Blight_Rot', 'Stemborer']
    model_path = r'.\pth.tar\best_acc_3.pth.tar'
    test_image_path = r'.\images'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = nets.SimpleCNN()
    checkpint = torch.load(model_path)
    model.load_state_dict(checkpint['state_dict'])
    model.eval()
    model.to(device)

    img_set = datasets.ImageFolder(test_image_path, transform=transformer)
    img_loader = DataLoader(img_set, batch_size=1, drop_last=False, shuffle=False)

    for order, batch_samples in enumerate(img_loader):

        img, label = batch_samples[0].to(device), batch_samples[1].to(device)
        output = model(img)
        print('-'*40)
        print(output)
        print(label)
        # for i in range(0, len(rice_disease_class)):
            # if output[0][i] > 0.01:
                # print(rice_disease_class[i]+" : %.2lf%%"%(output[0][i]*100))
        # print(output)
        # print(F.softmax(output, dim=1))
        _, pred = output.max(1)

        GradCAM.make_CAM(img_set.imgs[order][0], int(pred[0]), model_path)
        print("图%s的病例为 : %s" %(os.path.split(img_set.imgs[order][0])[1], rice_disease_class[pred[0]]))
