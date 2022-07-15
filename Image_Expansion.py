# -*-coding = utf-8 -*-
"""
1. Image_flip:翻转图片
2. Image_traslation:平移图片
3. Image_rotate:旋转图片
4. Image_noise:添加噪声
"""
import os
import cv2
import numpy as np
from random import choice
import random

def Image_flip(img):
    """
    :param img:原始图片矩阵
    :return: 0-垂直； 1-水平； -1-垂直&水平
    """
    if img is None:
        return
    paras = [0, 1, -1]
    img_new = cv2.flip(img, choice(paras))
    return img_new

def Image_traslation(img):
    """
    :param img: 原始图片矩阵
    :return: [1, 0, 100]-宽右移100像素； [0, 1, 100]-高下移100像素
    """
    # paras_wide = [[1, 0, 20], [1, 0, -20]]
    # paras_height = [[0, 1, 20], [0, 1, -20]]
    a = random.uniform(3, 20)
    b = random.uniform(3, 20)
    paras_wide = [[1, 0, a], [1, 0, -a]]
    paras_height = [[0, 1, b], [0, 1, -b]]

    rows, cols = img.shape[:2]
    img_shift = np.float32([choice(paras_wide), choice(paras_height)])
    border_value = tuple(int(x) for x in choice(choice(img)))
    img_new = cv2.warpAffine(img, img_shift, (cols, rows), borderValue=border_value)
    return img_new

def Image_rotate(img):
    """
    :param img:原始图片矩阵
    :return:旋转中心，旋转角度，缩放比例
    """
    rows, cols = img.shape[:2]
    rotate_core = (cols/2, rows/2)
    # rotate_angle = [60, -60, 45, -45, 90, -90, 210, 240, -210, -240]

    rotate_angle = random.uniform(15, 350)

    paras = cv2.getRotationMatrix2D(rotate_core, rotate_angle, 1)
    border_value = tuple(int(x) for x in choice(choice(img)))
    img_new = cv2.warpAffine(img, paras, (cols, rows), borderValue=border_value)
    return img_new

def Image_noise(img):
    """
    :param img:原始图片矩阵
    :return: 0-高斯噪声，1-椒盐噪声
    """
    paras = [0, 1]
    gaussian_class = choice(paras)
    noise_ratio = [0.05, 0.06, 0.08]
    if gaussian_class == 1:
        output = np.zeros(img.shape, np.uint8)
        prob = choice(noise_ratio)
        thres = 1 - prob
        #print('prob', prob)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = img[i][j]
        return output
    else:
        mean = 0
        var=choice([0.001, 0.002, 0.003])
        #print('var', var)
        img = np.array(img/255, dtype=float)
        noise = np.random.normal(mean, var**0.5, img.shape)
        out = img + noise
        if out.min() < 0:
            low_clip = -1
        else:
            low_clip = 0
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out*255)
        return out

if __name__ == "__main__":
    # """
    # path_read: 读取原始数据集图片的位置;
    # path_write：图片扩增后存放的位置；
    # # picture_size：图片之后存储的尺寸;
    # enhance_hum: 需要通过扩增手段增加的图片数量
    # """
    # path_read = "C:\\Users\\TanJ\\Desktop\\data\\4\\"
    # path_write = "E:\\python_workspace\\Image_task\\data\\train\\4\\"

    path = "D:\\work\\224x224\\"

    # image_list = [x for x in os.listdir(path_read)]
    # existed_img = len(image_list)

    for i in range(1, 19):
        p_r = path + str(i) + "\\"
        enhance_num = 200

        while enhance_num > 0:
            image_list = [x for x in os.listdir(p_r)]
            img = choice(image_list)
            image = cv2.imread(p_r+img, cv2.IMREAD_COLOR)

            algorithm = [2, 3, 4]
            random_process = choice(algorithm)
            # if random_process == 1:
            #     image = Image_flip(image)
            if random_process == 2:
                image = Image_traslation(image)
            elif random_process == 3:
                image = Image_rotate(image)
            else:
                image = Image_noise(image)

            image_dir = p_r+str(enhance_num).zfill(5)+'.png'
            cv2.imwrite(image_dir, image)
            enhance_num -= 1