'''
为什么要超参数搜索？
    神经网络有很多训练过程中不变的参数
        1.网络结构参数：几层，每层宽度，每层激活函数等
        2.训练参数：batch_size，学习率，学习率衰减算法等
    手工去试耗费人力

搜索策略：
    1.网格搜索
    2.随机搜索
    3.遗传算法搜索
    4.启发式搜索
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
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

from sklearn.model_selection import train_test_split

x_train_all, x_test, y_train_all, y_test = train_test_split(
    housing.data, housing.target, random_state=7)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state=11)
print("x_train", x_train.shape, y_train.shape)
print("x_valid", x_valid.shape, y_valid.shape)
print("x_test", x_test.shape, y_test.shape)

# 归一化训练、验证、测试数据 x=(x-u)/std（u：均值，std：方差）
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

# 多输入
input_wide = keras.layers.Input(shape=[5])
input_deep = keras.layers.Input(shape=[6])
hidden1 = keras.layers.Dense(30, activation='relu')(input_deep)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.concatenate([input_wide, hidden2])
# 多输出
output = keras.layers.Dense(1)(concat)

learning_rate = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
histories = []
for lr in learning_rate:
    optimizer = keras.optimizers.SGD(lr)
    model = keras.models.Model(inputs=[input_wide, input_deep], outputs=output)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)
    ]

    x_train_scaled_wide = x_train_scaled[:, :5]
    x_train_scaled_deep = x_train_scaled[:, 2:]
    x_valid_scaled_wide = x_valid_scaled[:, :5]
    x_valid_scaled_deep = x_valid_scaled[:, 2:]
    x_test_scaled_wide = x_test_scaled[:, :5]
    x_test_scaled_deep = x_test_scaled[:, 2:]
    model.compile(loss="mean_squared_error",
                  optimizer=optimizer)
    history = model.fit([x_train_scaled_wide, x_train_scaled_deep], y_train, epochs=20,
                        validation_data=([x_valid_scaled_wide, x_valid_scaled_deep], y_valid),
                        callbacks=callbacks)
    histories.append(history)


def plot_learning_curves(history, lr):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.title(lr)
    plt.gca().set_ylim(0, 2)
    plt.show()


for lr, his in zip(learning_rate, histories):
    plot_learning_curves(his, lr)

