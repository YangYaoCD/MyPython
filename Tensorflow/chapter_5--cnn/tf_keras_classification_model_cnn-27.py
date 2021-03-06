# -*- coding: utf-8 -*-
# @Time    : 2020/1/7 9:51
# @Author  : YangYao
# @Email   : 1344262070@qq.com
# @File    : tf_keras_classification_model_cnn-27.py
# @Software: PyCharm
'''
理论部分：
    卷积神经网络
        卷积操作、池化操作
    深度可分离卷积
    数据增强
    迁移学习

实战部分：
    keras实现卷积神经网络
    keras实现深度可分离卷积
    keras实战kaggle
        10 monkeys和cifar10
        数据增强与迁移学习
'''
'''
结构：
    卷积神经网络
        （卷积层+（可选）池化层）*N+全连接层*M
        分类任务
    全卷积神经网络
        （卷积层+（可选）池化层）*N+反卷积层*K
        物理分割
'''
'''
卷积——解决问题
    1.局部连接
        图像的区域性
    2.参数共享
        图像特征与位置无关
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time

from tensorflow import keras

# 版本信息
print(tf.__version__)
print(sys.version_info)
for module in mpl, np, sklearn, tf, keras:
    print(module.__name__ + ":" + module.__version__)

# 读取代码
fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

# 归一化训练、验证、测试数据 x=(x-u)/std（u：均值，std：方差）
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# x_train[None,28,28]->[None,784]
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)

print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

class_names = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=32, kernel_size=3,
                              padding='same',
                              activation='relu',
                              input_shape=(28, 28, 1)))
model.add(keras.layers.Conv2D(filters=32, kernel_size=3,
                              padding='same',
                              activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3,
                              padding='same',
                              activation='relu'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3,
                              padding='same',
                              activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3,
                              padding='same',
                              activation='relu'))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3,
                              padding='same',
                              activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Dense(10, activation='softmax'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

logdir = '.\cnn-callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir, "fashion_mnist_model.h5")
callbacks = [
    keras.callbacks.TensorBoard(logdir),  # 开启tensorboard命令（tensorboard --logdir=callbacks）
    keras.callbacks.ModelCheckpoint(output_model_file, sava_best_only=True),
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
]
history = model.fit(x_train_scaled, y_train, epochs=10,
                    validation_data=(x_valid_scaled, y_valid),
                    callbacks=callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.title(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    plt.gca().set_ylim(0, 2)
    plt.show()


plot_learning_curves(history)

model_evaluate = model.evaluate(x_test_scaled, y_test)
print("model_evaluate" + str(model_evaluate))
