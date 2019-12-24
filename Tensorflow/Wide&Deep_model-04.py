'''
Wide&Deep模型：
    关于：
        1.16年发布，用于分类和回归
        2.应用到了Google Play中的应用推荐
        3.原始论文：https://arxiv.org/pdf/1606.07792v1.pdf

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
print("housing.DESCR" + housing.DESCR)
print("data" + str(housing.data.shape))
print("housing.target" + str(housing.target.shape))

from sklearn.model_selection import train_test_split

x_train_all, x_test, y_train_all, y_test = train_test_split(
    housing.data, housing.target, random_state=7, test_size=0.5)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state=11, test_size=0.8)
print("x_train", x_train.shape, y_train.shape)
print("x_valid", x_valid.shape, y_valid.shape)
print("x_test", x_test.shape, y_test.shape)

# 归一化训练、验证、测试数据 x=(x-u)/std（u：均值，std：方差）
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

# 函数式API（功能API）
input = keras.layers.Input(shape=x_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation='relu')(input)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)

concat = keras.layers.concatenate([input, hidden2])
output = keras.layers.Dense(1)(concat)

model = keras.models.Model(inputs=[input], outputs=[output])

print(model.summary())
# reason for sparse y->index y_.one_hot->[]
model.compile(loss="mean_squared_error",
              optimizer="sgd")
# loss=目标函数，或称损失函数，是网络中的性能函数，也是编译一个模型必须的两个参数之一。由于损失函数种类众多，下面以keras官网手册的为例。

# 添加回调函数(常使用Tensorboard,earlystopping,ModelCheckpoint)
logdir = '.\wide&deeep_model-callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir, "fashion_mnist_model.h5")
callbacks = [
    # keras.callbacks.TensorBoard(logdir),  # 开启tensorboard命令（tensorboard --logdir=callbacks）
    # keras.callbacks.ModelCheckpoint(output_model_file, sava_best_only=True),
    keras.callbacks.EarlyStopping(patience=50, min_delta=1e-2)
]
history = model.fit(x_train_scaled, y_train, epochs=100,
                    validation_data=(x_valid_scaled, y_valid),
                    callbacks=callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.title(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learning_curves(history)

model_evaluate = model.evaluate(x_test_scaled, y_test)
print("model_evaluate" + str(model_evaluate))
