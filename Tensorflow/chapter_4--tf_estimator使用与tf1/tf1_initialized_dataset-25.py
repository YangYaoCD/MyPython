# -*- coding: utf-8 -*-
# @Time    : 2020/1/6 20:34
# @Author  : YangYao
# @Email   : 1344262070@qq.com
# @File    : tf1_initialized_dataset-25.py
# @Software: PyCharm
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
print(np.max(x_train), np.min(x_train))

# 归一化训练、验证、测试数据 x=(x-u)/std（u：均值，std：方差）
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# x_train[None,28,28]->[None,784]
print(type(x_train))
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28 * 28)
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28 * 28)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28 * 28)

y_train = np.asarray(y_train, dtype=np.int64)
y_valid = np.asarray(y_valid, dtype=np.int64)
y_test = np.asarray(y_test, dtype=np.int64)


def make_dataset(images, labels, epochs, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset


batch_size = 20
epochs = 10

images_placeholder=tf.placeholder(tf.float32,[None,28*28])
labels_placeholder=tf.placeholder(tf.int64,[None,])
dataset = make_dataset(images_placeholder, labels_placeholder, epochs=epochs, batch_size=batch_size)

# make_one_shot_iterator(): 1. auto initialization 2. can't be re-initialized
dataset_iter = dataset.make_initializable_iterator()
x, y = dataset_iter.get_next()
with tf.Session() as sess:
    sess.run(dataset_iter.initializer,
             feed_dict={
                 images_placeholder:x_train_scaled,
                 labels_placeholder:y_train
             })
    x_val, y_val = sess.run([x, y])
    print(x_val.shape)
    print(y_val.shape)
    sess.run(dataset_iter.initializer,feed_dict={
        images_placeholder:x_valid_scaled,
        labels_placeholder:y_valid})
    x_val, y_val = sess.run([x, y])
    print(x_val.shape)
    print(y_val.shape)

hidden_units = [100, 100]
class_num = 10

input_for_next_layer = x
for hidden_unit in hidden_units:
    input_for_next_layer = tf.layers.dense(input_for_next_layer,
                                           hidden_unit, activation=tf.nn.relu)

logits = tf.layers.dense(input_for_next_layer, class_num)

loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(prediction, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))  # 计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# print(x)
# print(logits)

# session

init = tf.global_variables_initializer()
train_steps_per_epoch = x_train.shape[0] // batch_size

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for step in range(train_steps_per_epoch):
            loss_val, accuracy_val, _ = sess.run(
                [loss, accuracy, train_op])
            print('\r[Train] epoch: %d, step: %d,loss:%3.5f,accuracy:%2.2f'
                  % (epoch, step, loss_val, accuracy_val), end="")