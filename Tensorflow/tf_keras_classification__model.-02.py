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
    print(module.__name__+":"+module.__version__)

#读取代码
fashion_mnist=keras.datasets.fashion_mnist
(x_train_all,y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]
print(np.max(x_train), np.min(x_train))

#归一化训练、验证、测试数据 x=(x-u)/std（u：均值，std：方差）
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
#x_train[None,28,28]->[None,784]
print(type(x_train))
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)

print(x_valid.shape, y_valid.shape) #x_valid(5000,28,28),y_valid(5000,)
print(x_train.shape, y_train.shape) #x_train(55000,28,28),y_train(55000,)
print(x_test.shape, y_test.shape)   #x_test(10000,28,28)


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
show_imags(4, 5, x_train, y_train, class_names)

# tf.keras.models.Sequential()
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[x_train.shape[1], x_train.shape[2]]),
    keras.layers.Dense(300, activation="relu"), #dense layer是全连接层
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
'''
softmax：将向量变成概率分布，x={x1,x2,x3},
            y={e^x1/sum,e^x2/sum,e^x3/sum}
'''

#reason for sparse y->index y_.one_hot->[]
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

print(str(model.layers))
print(model.summary())
# 添加回调函数(常使用Tensorboard,earlystopping,ModelCheckpoint)
logdir = '.\callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir, "fashion_mnist_model.h5")
callbacks = [
    keras.callbacks.TensorBoard(logdir),#开启tensorboard命令（tensorboard --logdir=callbacks）
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
    plt.gca().set_ylim(0, 1)
    plt.show()

plot_learning_curves(history)

model_evaluate = model.evaluate(x_test_scaled, y_test)
print("model_evaluate"+str(model_evaluate))
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
    print(module.__name__+":"+module.__version__)

#读取代码
fashion_mnist=keras.datasets.fashion_mnist
(x_train_all,y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]
print(np.max(x_train), np.min(x_train))

#归一化训练、验证、测试数据 x=(x-u)/std（u：均值，std：方差）
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
#x_train[None,28,28]->[None,784]
print(type(x_train))
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)

print(x_valid.shape, y_valid.shape) #x_valid(5000,28,28),y_valid(5000,)
print(x_train.shape, y_train.shape) #x_train(55000,28,28),y_train(55000,)
print(x_test.shape, y_test.shape)   #x_test(10000,28,28)


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
show_imags(4, 5, x_train, y_train, class_names)

# tf.keras.models.Sequential()
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[x_train.shape[1], x_train.shape[2]]),
    keras.layers.Dense(300, activation="relu"), #dense layer是全连接层
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
'''
softmax：将向量变成概率分布，x={x1,x2,x3},
            y={e^x1/sum,e^x2/sum,e^x3/sum}
'''

#reason for sparse y->index y_.one_hot->[]
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

print(str(model.layers))
print(model.summary())
# 添加回调函数(常使用Tensorboard,earlystopping,ModelCheckpoint)
logdir = '.\callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir, "fashion_mnist_model.h5")
callbacks = [
    keras.callbacks.TensorBoard(logdir),#开启tensorboard命令（tensorboard --logdir=callbacks）
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
    plt.gca().set_ylim(0, 1)
    plt.show()

plot_learning_curves(history)

model_evaluate = model.evaluate(x_test_scaled, y_test)
print("model_evaluate"+str(model_evaluate))
