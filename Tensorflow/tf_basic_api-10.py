'''
Tensorflow基础API：
    基础API
    基础API与keras的集成
        自定义损失函数
        自定义层次
    @tf.function的使用
        图结构
    自定义求导
'''

'''
API列表：
    1.基础数据类型：
        tf.constant,tf.string
        tf.ragged.constant.,tf.SparseTensor,tf.Variable
    2.自定义损失函数：tf.reduce_mean
    3.自定义层次：keras.layers.Lambda和继承法
    4.tf.function：
        tf.function,tf.autograph.to_code,get_concrete_function
    5.GraphDef：
        get_operations,get_operation_by_name
        get_tensor_by_name,as_graph_def
    6.自动求导
        tf.GradientTape
        Optimizer.apply_gradients
'''

'''
@tf.function：
    将python函数编译成图
    易于将模型导出成GraphDef+checkpoint或者SavedModel
    是的eager execution可以默认打开
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

t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
print(t)
print(t[:, 1:])
print(t[:, 2])
# ops
print(t + 10)
print(tf.square(t))
print(t @ tf.transpose(t))
# numpy&tensor conversion
print(t.numpy())
print(np.square(t))
np_t = np.array([[1., 2., 3.], [4., 5., 6.]])
print(tf.constant(np_t))
print("+++++++++++++++")
print(str(zip(t, np_t)))

# Scalars
t = tf.constant(2.738)
print(t.numpy())
print(t.shape)
# strings
t = tf.constant("cafe")
print(t)
print(tf.strings.length(t))
print(tf.strings.length(t, unit="UTF8_CHAR"))
print(tf.strings.unicode_decode(t, "UTF-8"))
# string array
t = tf.constant(["cafe", "coffee", "咖啡"])
print(tf.strings.length(t, unit="UTF8_CHAR"))
print(tf.strings.unicode_decode(t,
                                "UTF-8"))  # <tf.RaggedTensor [[99, 97, 102, 101], [99, 111, 102, 102, 101, 101], [21654, 21857]]>
# ragged Tensor
r = tf.ragged.constant([[11, 12], [21, 22, 23], [], [41]])
print(r)
print(r[1])
print(r[1:2])
# ops on ranged tensor
r2 = tf.ragged.constant([[51, 52], [], [71]])
tf_concat = tf.concat([r, r2], axis=0)
print(tf_concat)
print(r2.to_tensor())
# sparse tensor
s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]],  # 要排好序，否则调这个函数：tf.sparse.reorder(s)
                    values=[1., 2., 3.],
                    dense_shape=[3, 4])
print("稀疏矩阵：")
print(s)
print(tf.sparse.to_dense(s))
# ops on sparse tensors
s2 = s * 2.0
print("s2:" + str(s2))
try:
    s3 = s + 1
except TypeError as ex:
    print(ex)
s4 = tf.constant([[10., 20.],
                  [30., 40.],
                  [50., 60.],
                  [70., 80.]])
print(tf.sparse.sparse_dense_matmul(s, s4))

# Variables
print("Variables")
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
print(v)
print(v.value())
print(v.numpy())
# assign value(赋值)
v.assign(2 * v)
v[0, 1].assign(42.)
print(v.numpy())
v[1].assign([7., 8., 9.])
print(v.numpy)
