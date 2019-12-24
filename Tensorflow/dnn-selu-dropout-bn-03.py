'''
2-9完成的任务
    1.Keras实现深度神经网络
    2.Keras更改激活函数
    3.Keras实现批归一化
    4.Keras实现dropout
'''

'''
笔记：
    1.如果出现梯度消失
        1. 参数众多，训练不充分
        2. 梯度消失 -> 链式法则 -> 复合函数f(g(x))
    2.为什么批规一化可以缓解梯度消失
    3.'selu'激活函数自带归一化
'''

'''
课后作业：
    selu和AlphaDropout实现机制和算法
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
    x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)

print(x_valid.shape, y_valid.shape)  # x_valid(5000,28,28),y_valid(5000,)
print(x_train.shape, y_train.shape)  # x_train(55000,28,28),y_train(55000,)
print(x_test.shape, y_test.shape)  # x_test(10000,28,28)


def show_single_image(img_arr):
    plt.imshow(img_arr, cmap="binary")
    plt.show()


def show_imags(n_rows, n_cols, x_data, y_data, class_name):
    assert len(x_data) == len(y_data)
    assert n_rows * n_cols < len(x_data)
    plt.figure(figsize=(n_cols * 1.4, n_rows * 1.6))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(x_data[index], cmap="binary", interpolation="nearest")
            plt.axis('off')
            plt.title(class_name[y_data[index]])
    plt.show()


# show_single_image(x_train[0])
class_names = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# show_imags(4, 5, x_train, y_train, class_names)

# tf.keras.models.Sequential()
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
for _ in range(20):
    model.add(keras.layers.Dense(100, activation='selu'))
    # 添加批规一化
    # model.add(keras.layers.BatchNormalization())
    '''
    归一化 放在激活函数前与放在激活函数后
    model.add(keras.layers.Dense(100)
    model.add(keras.layers.BatchNormalization())
    model.add(keras.laters.Activation('relu'))
    '''
model.add(keras.layers.AlphaDropout(rate=0.5))
'''
普通：
model.add(keras.layers.Dropout(rate=0.5))
Alpha:1.均值和方差不变   2.归一化性质不变
'''
model.add(keras.layers.Dense(10, activation='softmax'))
'''
softmax：将向量变成概率分布，x={x1,x2,x3},
            y={e^x1/sum,e^x2/sum,e^x3/sum}
'''

# reason for sparse y->index y_.one_hot->[]
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
# loss=目标函数，或称损失函数，是网络中的性能函数，也是编译一个模型必须的两个参数之一。由于损失函数种类众多，下面以keras官网手册的为例。

print(model.summary())
# 添加回调函数(常使用Tensorboard,earlystopping,ModelCheckpoint)
logdir = '.\dnn-callbacks'
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
print(history.history)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.title(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    plt.gca().set_ylim(0, 2)
    plt.show()


plot_learning_curves(history)

model_evaluate = model.evaluate(x_test_scaled, y_test)
print("model_evaluate" + str(model_evaluate))
