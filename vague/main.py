# -- coding: utf-8 --
# @Author: zrs
# @Time: 2022-03-03 20:42
# @File: main.py

from PIL import Image, ImageFilter
import os
import numpy as np


class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)


open_path = r'D:\Programs\DeepLearning\rice\dataset\FOLD_9'
save_path = r'.\data\flod_9_dim5'

if __name__ == '__main__':


    for i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        for j in ['train', 'test']:
            for k in ['BPH', 'put_img', 'Brown_Spot', 'False_Smut', 'Healthy_Plant', 'Hispa', 'Neck_Blast', 'Sheath_Blight_Rot', 'Stemborer']:
                # print(type(open_path), type(i), type(j), type(k))
                # print(open_path)
                open_path_now = os.path.join(open_path, i, j, k)
                # print((open_path_now))
                save_path_now = os.path.join(save_path, i, j, k)
                if os.path.exists(save_path_now) == False : os.makedirs(save_path_now)
                open_path_list = os.listdir(open_path_now)
                for l in open_path_list:
                    open_image_path = os.path.join(open_path_now, l)
                    save_image_path = os.path.join(save_path_now, l)
                    frame_image = Image.open(open_image_path)

                    # print(np.array(frame_image).shape)
                    image = frame_image.filter(MyGaussianBlur(radius=5))   # 通过调节redius的值调整图片的清晰度
                    # image.thumbnail((1920, 1020))  # 调整图片的尺寸
                    # print(np.array(image).shape)
                    # image.save(save_image_path)

    print('success')