'''
    自定义求导
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


def f(x):
    return 3. * x ** 2 + 2. * x - 1


def approximate_derivative(f, x, eps=1e-3):
    return (f(x + eps) - f(x - eps)) / (2. * eps)


print("近似导数求法：")
print(approximate_derivative(f, 1.))


def g(x1, x2):
    return (x1 + 5) * (x2 ** 2)


def approximate_gradient(g, x1, x2, eps=1e-3):
    dg_x1 = approximate_derivative(lambda x: g(x, x2), x1, eps)
    dg_x2 = approximate_derivative(lambda x: g(x1, x), x2, eps)
    return dg_x1, dg_x2


print("多元近似求导")
print(approximate_gradient(g, 2., 3.))

'''正课开始'''
# 1.一次求导
x1 = tf.Variable(2.)  # 常量要tape.watch(x)才行
x2 = tf.Variable(3.)
with tf.GradientTape(persistent=True) as tape:
    z = g(x1, x2)
dz_x1x2 = tape.gradient(z, [x1, x2])
print(dz_x1x2)
# 第二种方式求偏导
# dz_x1=tape.gradient(z,x1)
# dz_x2=tape.gradient(z,x2)
# print(dz_x1,dz_x2)

del tape

# 二阶偏微分
x1 = tf.Variable(2.)  # 常量要tape.watch(x)才行
x2 = tf.Variable(3.)
with tf.GradientTape(persistent=True) as outer_tape:
    with tf.GradientTape(persistent=True) as inner_tape:
        z = g(x1, x2)
    inner_grads = inner_tape.gradient(z, [x1, x2])
outer_grads = [outer_tape.gradient(inner_grad, [x1, x2]) for inner_grad in inner_grads]

print("二阶偏微分")
print(outer_grads)
del inner_tape
del outer_tape

# 3.梯度下降的例子
learning_rate = 0.1
x = tf.Variable(0.)

for _ in range(100):
    with tf.GradientTape(persistent=True) as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    x.assign_sub(learning_rate * dz_dx)

print("梯度下降：")
print(x)

# 与optimizer结合
learning_rate = 0.1
x = tf.Variable(0.)

optimizer = keras.optimizers.SGD(lr=learning_rate)

for _ in range(100):
    with tf.GradientTape(persistent=True) as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    optimizer.apply_gradients([(dz_dx, x)])

print("optimizer梯度下降：")
print(x)
