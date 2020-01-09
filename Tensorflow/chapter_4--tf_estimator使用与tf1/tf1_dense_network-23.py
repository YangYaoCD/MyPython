# -*- coding: utf-8 -*-
# @Time    : 2020/1/5 21:31
# @Author  : YangYao
# @Email   : 1344262070@qq.com
# @File    : tf1_dense_network-23.py
# @Software: PyCharm

'''
tf1.0:
    1. tf1.0实现全连接网络
        placeholder, sess, feed_dict
    2. Dataset使用
    3. 自定义estimator
    4. 实战
    5. tf1.0与2.0区别
'''

'''
tf1.0实现全连接网络
    placeholder, tf.layers.dense, tf.train.AdamOptimizer
    tf.losses.sparse_softmax_cross_entropy,
    tf.globel_variables_initializer, feed_dict

Dataset
    Dataset.make_one_shot_iterator
    Datasete.make_initializable_iterator

自定义estimator
    tf.feature_column.input_layer
    tf.estimator.EstimatorSpec
    tf.metrics.accuracy
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

hidden_units = [100, 100]
class_num = 10

x = tf.placeholder(tf.float32, [None, 28 * 28])
y = tf.placeholder(tf.int64, [None])

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
batch_size = 20
epochs = 10
train_steps_per_epoch = x_train.shape[0] // batch_size
valid_steps = x_valid.shape[0] // batch_size


def eval_with_sess(sess, x, y, accuracy, images, labels, batch_size):
    eval_steps = images.shape[0] // batch_size
    eval_accuracies = []
    for step in range(eval_steps):
        batch_data = images[step * batch_size:(step + 1) * batch_size]
        batch_label = labels[step * batch_size:(step + 1) * batch_size]
        accuracy_val = sess.run(accuracy, feed_dict={
            x: batch_data,
            y: batch_label
        })
        eval_accuracies.append(accuracy_val)
    return np.mean(eval_accuracies)


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for step in range(train_steps_per_epoch):
            batch_data = x_train_scaled[step * batch_size:(step + 1) * batch_size]
            batch_label = y_train[step * batch_size:(step + 1) * batch_size]
            loss_val, accuracy_val, _ = sess.run(
                [loss, accuracy, train_op], feed_dict={
                    x: batch_data,
                    y: batch_label})
            print('\r[Train] epoch: %d, step: %d,loss:%3.5f,accuracy:%2.2f'
                  % (epoch, step, loss_val, accuracy_val), end="")
        valid_accuracy = eval_with_sess(sess, x, y, accuracy, x_valid_scaled, y_valid, batch_size)
        print("\t[Valid] acc:%2.2f" % valid_accuracy)
