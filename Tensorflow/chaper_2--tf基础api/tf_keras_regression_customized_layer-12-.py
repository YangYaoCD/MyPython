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

layer = tf.keras.layers.Dense(100)
layer = tf.keras.layers.Dense(100, input_shape=(None, 5))
layer(tf.zeros([10, 5]))
print(layer.variables)
help(layer)

# 读取代码
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

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

customized_softplus=keras.layers.Lambda(lambda x:tf.nn.softplus(x))
print(customized_softplus(tf.constant([1.,2.,3.,4.])))
class CustomizedDenseLayer(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(CustomizedDenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        '''构建所需要的参数'''
        self.kernal = self.add_weight(name='kernel',
                                      shape=(input_shape[1],self.units),
                                      initializer='uniform',
                                      trainable=True)
        self.bias=self.add_weight(name='bias',shape=(self.units,),
                                  initializer='zeros',
                                  trainable=True)
        super(CustomizedDenseLayer,self).build(input_shape)

    def call(self, x):
        '''完整正向计算'''
        return self.activation(x@self.kernal+self.bias)


# 函数式API（功能API）
model=keras.models.Sequential([
    CustomizedDenseLayer(30,input_shape=(x_train_scaled.shape[1:]),activation='relu'),
    CustomizedDenseLayer(1,activation='relu'),
    customized_softplus#等价于keras.layers.Dense(1,activation='softplus')
])

print(model.summary())

# reason for sparse y->index y_.one_hot->[]
model.compile(loss="mse",
              optimizer="sgd")
# loss=目标函数，或称损失函数，是网络中的性能函数，也是编译一个模型必须的两个参数之一。由于损失函数种类众多，下面以keras官网手册的为例。

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
    plt.gca().set_ylim(0, 10)
    plt.show()


plot_learning_curves(history)

model_evaluate = model.evaluate(x_test_scaled, y_test)
print("model_evaluate" + str(model_evaluate))
