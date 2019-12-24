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
print("housing.DESCR" + housing.DESCR)
print("data" + str(housing.data.shape))
print("housing.target" + str(housing.target.shape))

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
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
print(model.summary())

# reason for sparse y->index y_.one_hot->[]
model.compile(loss="mean_squared_error",
              optimizer="sgd")
# loss=目标函数，或称损失函数，是网络中的性能函数，也是编译一个模型必须的两个参数之一。由于损失函数种类众多，下面以keras官网手册的为例。

callbacks = [
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)
]

x_train_scaled_wide = x_train_scaled[:, :5]
x_train_scaled_deep = x_train_scaled[:, 2:]
x_valid_scaled_wide = x_valid_scaled[:, :5]
x_valid_scaled_deep = x_valid_scaled[:, 2:]
x_test_scaled_wide = x_test_scaled[:, :5]
x_test_scaled_deep = x_test_scaled[:, 2:]

history = model.fit([x_train_scaled_wide, x_train_scaled_deep], y_train, epochs=100,
                    validation_data=([x_valid_scaled_wide, x_valid_scaled_deep], y_valid),
                    callbacks=callbacks)
print(history.history)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.title(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learning_curves(history)

model_evaluate = model.evaluate([x_test_scaled_wide, x_test_scaled_deep], y_test)
print("model_evaluate" + str(model_evaluate))
