import os, sys
import numpy as np
import keras
import scipy
from scipy import ndimage
from keras import layers, optimizers, models
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import random
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout
from keras.models import Model
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.resnet50 import ResNet50
# from keras.applications.densenet import DenseNet121
# from efficientnet.keras import EfficientNet80
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
# from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2


# def DataSet():
#     # 首先需要定义训练集和测试集的路径，这里创建了 train ， 和 test 文件夹
#     train_path_B301_11 = ".\\data\\train\\B301-11\\"
#     train_path_B301_18 = ".\\data\\train\\B301-18\\"
#     train_path_B301_31 = ".\\data\\train\\B301-31\\"
#     train_path_B301_46 = ".\\data\\train\\B301-46\\"
#     train_path_B303_23 = ".\\data\\train\\B303-23\\"
#     train_path_B303_28 = ".\\data\\train\\B303-28\\"
#     train_path_B303_32 = ".\\data\\train\\B303-32\\"
#     train_path_B303_35 = ".\\data\\train\\B303-35\\"
#     train_path_B303_57 = ".\\data\\train\\B303-57\\"
#     train_path_B303_75 = ".\\data\\train\\B303-75\\"
#
#     test_path_B301_11 = ".\\data\\test\\B301-11\\"
#     test_path_B301_18 = ".\\data\\test\\B301-18\\"
#     test_path_B301_31 = ".\\data\\test\\B301-31\\"
#     test_path_B301_46 = ".\\data\\test\\B301-46\\"
#     test_path_B303_23 = ".\\data\\test\\B303-23\\"
#     test_path_B303_28 = ".\\data\\test\\B303-28\\"
#     test_path_B303_32 = ".\\data\\test\\B303-32\\"
#     test_path_B303_35 = ".\\data\\test\\B303-35\\"
#     test_path_B303_57 = ".\\data\\test\\B303-57\\"
#     test_path_B303_75 = ".\\data\\test\\B303-75\\"
#
#     # os.listdir(path) 是 python 中的函数，它会列出 path 下的所有文件名
#     # 比如说 imglist_train_glue 对象就包括了/train/glue/ 路径下所有的图片文件名
#     imglist_train_B301_11 = os.listdir(train_path_B301_11)
#     imglist_train_B301_18 = os.listdir(train_path_B301_18)
#     imglist_train_B301_31 = os.listdir(train_path_B301_31)
#     imglist_train_B301_46 = os.listdir(train_path_B301_46)
#     imglist_train_B303_23 = os.listdir(train_path_B303_23)
#     imglist_train_B303_28 = os.listdir(train_path_B303_28)
#     imglist_train_B303_32 = os.listdir(train_path_B303_32)
#     imglist_train_B303_35 = os.listdir(train_path_B303_35)
#     imglist_train_B303_57 = os.listdir(train_path_B303_57)
#     imglist_train_B303_75 = os.listdir(train_path_B303_75)
#
#     # 下面两行代码读取了 /test/glue 和 /test/medicine 下的所有图片文件名
#     imglist_test_B301_11 = os.listdir(test_path_B301_11)
#     imglist_test_B301_18 = os.listdir(test_path_B301_18)
#     imglist_test_B301_31 = os.listdir(test_path_B301_31)
#     imglist_test_B301_46 = os.listdir(test_path_B301_46)
#     imglist_test_B303_23 = os.listdir(test_path_B303_23)
#     imglist_test_B303_28 = os.listdir(test_path_B303_28)
#     imglist_test_B303_32 = os.listdir(test_path_B303_32)
#     imglist_test_B303_35 = os.listdir(test_path_B303_35)
#     imglist_test_B303_57 = os.listdir(test_path_B303_57)
#     imglist_test_B303_75 = os.listdir(test_path_B303_75)
#
#
#     # 定义两个 numpy：X_train 和 Y_train
#
#     num_train = len(imglist_train_B301_11) + len(imglist_train_B301_18) + len(imglist_train_B301_31) + \
#                 len(imglist_train_B301_46) + len(imglist_train_B303_23) + len(imglist_train_B303_28) + \
#                 len(imglist_train_B303_32) + len(imglist_train_B303_35) + len(imglist_train_B303_57) + \
#                 len(imglist_train_B303_75)
#     num_test = len(imglist_test_B301_11) + len(imglist_test_B301_18) + len(imglist_test_B301_31) + \
#                 len(imglist_test_B301_46) + len(imglist_test_B303_23) + len(imglist_test_B303_28) + \
#                 len(imglist_test_B303_32) + len(imglist_test_B303_35) + len(imglist_test_B303_57) + \
#                 len(imglist_test_B303_75)
#
#     X_train = np.empty((num_train, 224, 224, 3))
#     Y_train = np.empty((num_train, 10))
#
#     # count 对象用来计数，每添加一张图片便加 1
#     count = 0
#     for img_name in imglist_train_B301_11:
#         img_path = train_path_B301_11 + img_name
#         img = image.load_img(img_path, target_size=(224, 224))
#         img = image.img_to_array(img) / 255.0
#         X_train[count] = img
#         Y_train[count] = np.array((1, 0, 0, 0, 0, 0, 0, 0, 0, 0))
#         count += 1
#     for img_name in imglist_train_B301_18:
#         img_path = train_path_B301_18 + img_name
#         img = image.load_img(img_path, target_size=(224, 224))
#         img = image.img_to_array(img) / 255.0
#         X_train[count] = img
#         Y_train[count] = np.array((0, 1, 0, 0, 0, 0, 0, 0, 0, 0))
#         count += 1
#     for img_name in imglist_train_B301_31:
#         img_path = train_path_B301_31 + img_name
#         img = image.load_img(img_path, target_size=(224, 224))
#         img = image.img_to_array(img) / 255.0
#         X_train[count] = img
#         Y_train[count] = np.array((0, 0, 1, 0, 0, 0, 0, 0, 0, 0))
#         count += 1
#     for img_name in imglist_train_B301_46:
#         img_path = train_path_B301_46 + img_name
#         img = image.load_img(img_path, target_size=(224, 224))
#         img = image.img_to_array(img) / 255.0
#         X_train[count] = img
#         Y_train[count] = np.array((0, 0, 0, 1, 0, 0, 0, 0, 0, 0))
#         count += 1
#     for img_name in imglist_train_B303_23:
#         img_path = train_path_B303_23 + img_name
#         img = image.load_img(img_path, target_size=(224, 224))
#         img = image.img_to_array(img) / 255.0
#         X_train[count] = img
#         Y_train[count] = np.array((0, 0, 0, 0, 1, 0, 0, 0, 0, 0))
#         count += 1
#     for img_name in imglist_train_B303_28:
#         img_path = train_path_B303_28 + img_name
#         img = image.load_img(img_path, target_size=(224, 224))
#         img = image.img_to_array(img) / 255.0
#         X_train[count] = img
#         Y_train[count] = np.array((0, 0, 0, 0, 0, 1, 0, 0, 0, 0))
#         count += 1
#     for img_name in imglist_train_B303_32:
#         img_path = train_path_B303_32 + img_name
#         img = image.load_img(img_path, target_size=(224, 224))
#         img = image.img_to_array(img) / 255.0
#         X_train[count] = img
#         Y_train[count] = np.array((0, 0, 0, 0, 0, 0, 1, 0, 0, 0))
#         count += 1
#     for img_name in imglist_train_B303_35:
#         img_path = train_path_B303_35 + img_name
#         img = image.load_img(img_path, target_size=(224, 224))
#         img = image.img_to_array(img) / 255.0
#         X_train[count] = img
#         Y_train[count] = np.array((0, 0, 0, 0, 0, 0, 0, 1, 0, 0))
#         count += 1
#     for img_name in imglist_train_B303_57:
#         img_path = train_path_B303_57 + img_name
#         img = image.load_img(img_path, target_size=(224, 224))
#         img = image.img_to_array(img) / 255.0
#         X_train[count] = img
#         Y_train[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 1, 0))
#         count += 1
#     for img_name in imglist_train_B303_75:
#         img_path = train_path_B303_75 + img_name
#         img = image.load_img(img_path, target_size=(224, 224))
#         img = image.img_to_array(img) / 255.0
#         X_train[count] = img
#         Y_train[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 1))
#         count += 1
#
#     # 下面的代码是准备测试集的数据，与上面的内容完全相同
#     X_test = np.empty((num_test, 224, 224, 3))
#     Y_test = np.empty((num_test, 10))
#     count = 0
#
#     for img_name in imglist_test_B301_11:
#         img_path = test_path_B301_11 + img_name
#         img = image.load_img(img_path, target_size=(224, 224))
#         img = image.img_to_array(img) / 255.0
#         X_test[count] = img
#         Y_test[count] = np.array((1, 0, 0, 0, 0, 0, 0, 0, 0, 0))
#         count += 1
#     for img_name in imglist_test_B301_18:
#         img_path = test_path_B301_18 + img_name
#         img = image.load_img(img_path, target_size=(224, 224))
#         img = image.img_to_array(img) / 255.0
#         X_test[count] = img
#         Y_test[count] = np.array((0, 1, 0, 0, 0, 0, 0, 0, 0, 0))
#         count += 1
#     for img_name in imglist_test_B301_31:
#         img_path = test_path_B301_31 + img_name
#         img = image.load_img(img_path, target_size=(224, 224))
#         img = image.img_to_array(img) / 255.0
#         X_test[count] = img
#         Y_test[count] = np.array((0, 0, 1, 0, 0, 0, 0, 0, 0, 0))
#         count += 1
#     for img_name in imglist_test_B301_46:
#         img_path = test_path_B301_46 + img_name
#         img = image.load_img(img_path, target_size=(224, 224))
#         img = image.img_to_array(img) / 255.0
#         X_test[count] = img
#         Y_test[count] = np.array((0, 0, 0, 1, 0, 0, 0, 0, 0, 0))
#         count += 1
#     for img_name in imglist_test_B303_23:
#         img_path = test_path_B303_23 + img_name
#         img = image.load_img(img_path, target_size=(224, 224))
#         img = image.img_to_array(img) / 255.0
#         X_test[count] = img
#         Y_test[count] = np.array((0, 0, 0, 0, 1, 0, 0, 0, 0, 0))
#         count += 1
#     for img_name in imglist_test_B303_28:
#         img_path = test_path_B303_28 + img_name
#         img = image.load_img(img_path, target_size=(224, 224))
#         img = image.img_to_array(img) / 255.0
#         X_test[count] = img
#         Y_test[count] = np.array((0, 0, 0, 0, 0, 1, 0, 0, 0, 0))
#         count += 1
#     for img_name in imglist_test_B303_32:
#         img_path = test_path_B303_32 + img_name
#         img = image.load_img(img_path, target_size=(224, 224))
#         img = image.img_to_array(img) / 255.0
#         X_test[count] = img
#         Y_test[count] = np.array((0, 0, 0, 0, 0, 0, 1, 0, 0, 0))
#         count += 1
#     for img_name in imglist_test_B303_35:
#         img_path = test_path_B303_35 + img_name
#         img = image.load_img(img_path, target_size=(224, 224))
#         img = image.img_to_array(img) / 255.0
#         X_test[count] = img
#         Y_test[count] = np.array((0, 0, 0, 0, 0, 0, 0, 1, 0, 0))
#         count += 1
#     for img_name in imglist_test_B303_57:
#         img_path = test_path_B303_57 + img_name
#         img = image.load_img(img_path, target_size=(224, 224))
#         img = image.img_to_array(img) / 255.0
#         X_test[count] = img
#         Y_test[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 1, 0))
#         count += 1
#     for img_name in imglist_test_B303_75:
#         img_path = test_path_B303_75 + img_name
#         img = image.load_img(img_path, target_size=(224, 224))
#         img = image.img_to_array(img) / 255.0
#         X_test[count] = img
#         Y_test[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 1))
#         count += 1
#
#     # 打乱训练集中的数据
#     index = [i for i in range(len(X_train))]
#     random.shuffle(index)
#     X_train = X_train[index]
#     Y_train = Y_train[index]
#
#     # 打乱测试集中的数据
#     index = [i for i in range(len(X_test))]
#     random.shuffle(index)
#     X_test = X_test[index]
#     Y_test = Y_test[index]
#
#     return X_train, Y_train, X_test, Y_test

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def DataSet():
    # 首先需要定义训练集和测试集的路径，这里创建了 train ， 和 test 文件夹
    train_path_1 = ".\\data\\train\\1\\"
    train_path_2 = ".\\data\\train\\2\\"
    train_path_3 = ".\\data\\train\\3\\"
    train_path_4 = ".\\data\\train\\4\\"
    train_path_5 = ".\\data\\train\\5\\"
    train_path_6 = ".\\data\\train\\6\\"
    train_path_7 = ".\\data\\train\\7\\"
    train_path_8 = ".\\data\\train\\8\\"
    train_path_9 = ".\\data\\train\\9\\"
    train_path_10 = ".\\data\\train\\10\\"
    train_path_11 = ".\\data\\train\\11\\"
    train_path_12 = ".\\data\\train\\12\\"
    train_path_13 = ".\\data\\train\\13\\"
    train_path_14 = ".\\data\\train\\14\\"
    train_path_15 = ".\\data\\train\\15\\"
    train_path_16 = ".\\data\\train\\16\\"
    train_path_17 = ".\\data\\train\\17\\"
    train_path_18 = ".\\data\\train\\18\\"

    test_path_1 = ".\\data\\test\\1\\"
    test_path_2 = ".\\data\\test\\2\\"
    test_path_3 = ".\\data\\test\\3\\"
    test_path_4 = ".\\data\\test\\4\\"
    test_path_5 = ".\\data\\test\\5\\"
    test_path_6 = ".\\data\\test\\6\\"
    test_path_7 = ".\\data\\test\\7\\"
    test_path_8 = ".\\data\\test\\8\\"
    test_path_9 = ".\\data\\test\\9\\"
    test_path_10 = ".\\data\\test\\10\\"
    test_path_11 = ".\\data\\test\\11\\"
    test_path_12 = ".\\data\\test\\12\\"
    test_path_13 = ".\\data\\test\\13\\"
    test_path_14 = ".\\data\\test\\14\\"
    test_path_15 = ".\\data\\test\\15\\"
    test_path_16 = ".\\data\\test\\16\\"
    test_path_17 = ".\\data\\test\\17\\"
    test_path_18 = ".\\data\\test\\18\\"

    # os.listdir(path) 是 python 中的函数，它会列出 path 下的所有文件名
    # 比如说 imglist_train_glue 对象就包括了/train/glue/ 路径下所有的图片文件名
    imglist_train_1 = os.listdir(train_path_1)
    imglist_train_2 = os.listdir(train_path_2)
    imglist_train_3 = os.listdir(train_path_3)
    imglist_train_4 = os.listdir(train_path_4)
    imglist_train_5 = os.listdir(train_path_5)
    imglist_train_6 = os.listdir(train_path_6)
    imglist_train_7 = os.listdir(train_path_7)
    imglist_train_8 = os.listdir(train_path_8)
    imglist_train_9 = os.listdir(train_path_9)
    imglist_train_10 = os.listdir(train_path_10)
    imglist_train_11 = os.listdir(train_path_11)
    imglist_train_12 = os.listdir(train_path_12)
    imglist_train_13 = os.listdir(train_path_13)
    imglist_train_14 = os.listdir(train_path_14)
    imglist_train_15 = os.listdir(train_path_15)
    imglist_train_16 = os.listdir(train_path_16)
    imglist_train_17 = os.listdir(train_path_17)
    imglist_train_18 = os.listdir(train_path_18)

    # 下面两行代码读取了 /test/glue 和 /test/medicine 下的所有图片文件名
    imglist_test_1 = os.listdir(test_path_1)
    imglist_test_2 = os.listdir(test_path_2)
    imglist_test_3 = os.listdir(test_path_3)
    imglist_test_4 = os.listdir(test_path_4)
    imglist_test_5 = os.listdir(test_path_5)
    imglist_test_6 = os.listdir(test_path_6)
    imglist_test_7 = os.listdir(test_path_7)
    imglist_test_8 = os.listdir(test_path_8)
    imglist_test_9 = os.listdir(test_path_9)
    imglist_test_10 = os.listdir(test_path_10)
    imglist_test_11 = os.listdir(test_path_11)
    imglist_test_12 = os.listdir(test_path_12)
    imglist_test_13 = os.listdir(test_path_13)
    imglist_test_14 = os.listdir(test_path_14)
    imglist_test_15 = os.listdir(test_path_15)
    imglist_test_16 = os.listdir(test_path_16)
    imglist_test_17 = os.listdir(test_path_17)
    imglist_test_18 = os.listdir(test_path_18)


    # 定义两个 numpy：X_train 和 Y_train

    num_train = len(imglist_train_1) + len(imglist_train_2) + len(imglist_train_3) + \
                len(imglist_train_4) + len(imglist_train_5) + len(imglist_train_6) + \
                len(imglist_train_7) + len(imglist_train_8) + len(imglist_train_9) + \
                len(imglist_train_10) + len(imglist_train_11) + len(imglist_train_12) + \
                len(imglist_train_13) + len(imglist_train_14) + len(imglist_train_15) + \
                len(imglist_train_16) + len(imglist_train_17) + len(imglist_train_18)

    num_test =  len(imglist_test_1) + len(imglist_test_2) + len(imglist_test_3) + \
                len(imglist_test_4) + len(imglist_test_5) + len(imglist_test_6) + \
                len(imglist_test_7) + len(imglist_test_8) + len(imglist_test_9) + \
                len(imglist_test_10) + len(imglist_test_11) + len(imglist_test_12) + \
                len(imglist_test_13) + len(imglist_test_14) + len(imglist_test_15) + \
                len(imglist_test_16) + len(imglist_test_17) + len(imglist_test_18)


    X_train = np.empty((num_train, 224, 224, 3))
    Y_train = np.empty((num_train, 18))

    # count 对象用来计数，每添加一张图片便加 1
    count = 0
    for img_name in imglist_train_1:
        img_path = train_path_1 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_train_2:
        img_path = train_path_2 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_train_3:
        img_path = train_path_3 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_train_4:
        img_path = train_path_4 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_train_5:
        img_path = train_path_5 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_train_6:
        img_path = train_path_6 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_train_7:
        img_path = train_path_7 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_train_8:
        img_path = train_path_8 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_train_9:
        img_path = train_path_9 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_train_10:
        img_path = train_path_10 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_train_11:
        img_path = train_path_11 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_train_12:
        img_path = train_path_12 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_train_13:
        img_path = train_path_13 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_train_14:
        img_path = train_path_14 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_train_15:
        img_path = train_path_15 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0))
        count += 1
    for img_name in imglist_train_16:
        img_path = train_path_16 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0))
        count += 1
    for img_name in imglist_train_17:
        img_path = train_path_17 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0))
        count += 1
    for img_name in imglist_train_18:
        img_path = train_path_18 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_train[count] = img
        Y_train[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1))
        count += 1



    # 下面的代码是准备测试集的数据，与上面的内容完全相同
    X_test = np.empty((num_test, 224, 224, 3))
    Y_test = np.empty((num_test, 18))
    count = 0

    for img_name in imglist_test_1:
        img_path = test_path_1 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_test_2:
        img_path = test_path_2 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_test_3:
        img_path = test_path_3 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_test_4:
        img_path = test_path_4 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_test_5:
        img_path = test_path_5 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_test_6:
        img_path = test_path_6 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_test_7:
        img_path = test_path_7 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_test_8:
        img_path = test_path_8 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_test_9:
        img_path = test_path_9 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_test_10:
        img_path = test_path_10 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_test_11:
        img_path = test_path_11 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_test_12:
        img_path = test_path_12 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_test_13:
        img_path = test_path_13 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_test_14:
        img_path = test_path_14 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0))
        count += 1
    for img_name in imglist_test_15:
        img_path = test_path_15 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0))
        count += 1
    for img_name in imglist_test_16:
        img_path = test_path_16 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0))
        count += 1
    for img_name in imglist_test_17:
        img_path = test_path_17 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0))
        count += 1
    for img_name in imglist_test_18:
        img_path = test_path_18 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        X_test[count] = img
        Y_test[count] = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1))
        count += 1

    # 打乱训练集中的数据
    index = [i for i in range(len(X_train))]
    random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]

    # 打乱测试集中的数据
    index = [i for i in range(len(X_test))]
    random.shuffle(index)
    X_test = X_test[index]
    Y_test = Y_test[index]

    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = DataSet()
print('X_train shape : ',X_train.shape)
print('Y_train shape : ',Y_train.shape)
print('X_test shape : ',X_test.shape)
print('Y_test shape : ',Y_test.shape)

class LossHistory(keras.callbacks.Callback):
    # 函数开始时创建盛放loss与acc的容器
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    # 按照batch来进行追加数据
    def on_batch_end(self, batch, logs={}):
        # 每一个batch完成后向容器里面追加loss，acc
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        # 每五秒按照当前容器里的值来绘图
        if int(time.time()) % 5 == 0:
            self.draw_p(self.losses['batch'], 'loss', 'train_batch')
            self.draw_p(self.accuracy['batch'], 'acc', 'train_batch')
            self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
            self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')

    def on_epoch_end(self, batch, logs={}):
        # 每一个epoch完成后向容器里面追加loss，acc
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        # 每五秒按照当前容器里的值来绘图
        if int(time.time()) % 5 == 0:
            self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
            self.draw_p(self.accuracy['epoch'], 'acc', 'train_epoch')
            self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')
            self.draw_p(self.val_acc['epoch'], 'acc', 'val_epoch')

    # 绘图，这里把每一种曲线都单独绘图，若想把各种曲线绘制在一张图上的话可修改此方法
    def draw_p(self, lists, label, type):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig(type + '_' + label + '.jpg')

    # 由于这里的绘图设置的是5s绘制一次，当训练结束后得到的图可能不是一个完整的训练过程（最后一次绘图结束，有训练了0-5秒的时间）
    # 所以这里的方法会在整个训练结束以后调用
    def end_draw(self):
        self.draw_p(self.losses['batch'], 'loss', 'train_batch')
        self.draw_p(self.accuracy['batch'], 'acc', 'train_batch')
        self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
        self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')
        self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
        self.draw_p(self.accuracy['epoch'], 'acc', 'train_epoch')
        self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')
        self.draw_p(self.val_acc['epoch'], 'acc', 'val_epoch')
# history = LossHistory()


# # # model = Xception(weights= None, classes=10)
# # # base_model = InceptionResNetV2(weights='imagenet', include_top=False)
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# x = base_model.output
# # x = GlobalAveragePooling2D()(x) # GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
# x = Flatten()(x)
# x = Dense(1024, activation='relu')(x)
# x = Dropout(0.5)(x)
# predictions = Dense(18, activation='softmax')(x)
# model = Model(inputs=base_model.input, outputs=predictions)
#
# def setup_to_transfer_learning(model,base_model):#base_model
#     for layer in base_model.layers:
#         layer.trainable = True
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# setup_to_transfer_learning(model,base_model)


#################################################################################################


# conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# model = models.Sequential()
# model.add(conv_base)
# model.add(Flatten())
# # model.add(layers.Dense(1024, activation='relu'))
# model.add(layers.Dense(18, activation='softmax'))
# # conv_base.trainable = False
# for layer in conv_base.layers:
#     layer.trainable = False
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


base_model = ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(18, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False
    # layer.trainable = True

model.compile(loss='categorical_crossentropy',
              # optimizer=optimizers.RMSprop(lr=1e-3),
              optimizer='adam',
              metrics=['acc'])

model.summary()



# # train
model.fit(X_train, Y_train,
          epochs=1,
          batch_size=32,
          validation_data=(X_test, Y_test))
          # callbacks=[history])
# model.save('Xception.h5')
# history.end_draw()
#
#
#
#
class_indict = {"1":'白色大理岩',"2":'白云岩',"3":'残坡积粉砂质粘土',"4":'第四系腐植土及冲积砾石',"5":'粉屑灰岩',"6":'腐植土及残坡积',"7":'红色粉砂岩、粉砂质页岩',"8":'灰绿色、青灰色泥质页岩',"9":'灰色大理岩',"10":'灰质白云岩',"11":'绢英岩化砾岩',"12":'泥质粉砂岩',"13":'弱片麻细中粒含石榴二长花岗岩',"14":'砂砾石层',"15":'中粗粒二长花岗岩',"16":'磁铁矿脉',"17":'黑云角闪斜长片麻岩',"18":'石英云母片岩和绿泥石英云母片岩'}
# 网络模型的微调
preds = model.evaluate(X_test, Y_test, batch_size=32)
print("loss=" + str(preds[0]))
print("accuracy=" + str(preds[1]))
#
pre = model.predict(X_test)
prediction = np.squeeze(pre)
prediction = np.argmax(prediction, axis=1)
print('预测该图片类别是：', class_indict[str(prediction)], ' 预测概率是：', prediction[prediction])
origin = np.argmax(Y_test, axis=1)
print('#####################################')
cm = confusion_matrix(origin, prediction)
print(cm)
print('#####################################')
cr = classification_report(origin, prediction)
print(cr)

# # result = model.predict(img)
#
# prediction = np.squeeze(pre)
#
# predict_class = np.argmax(pre)
#
# print('预测该图片类别是：', class_indict[str(predict_class)], ' 预测概率是：', prediction[predict_class])
#
# plt.show()

