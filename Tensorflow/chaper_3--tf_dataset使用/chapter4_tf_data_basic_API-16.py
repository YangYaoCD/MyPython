'''
tf.data API使用
    1.Dataset基础API使用
    2.Dataset读取csv文件
    3.Dataset读取tfrecord（tensorflow独有格式）文件
'''

'''
API列表：
    1.Dataset基础使用
        tf.data.Dataset.from_tensor_slices
        repeat, batch, interleave, map, shuffle, list_files
    2.csv
        tf.data.TextLineDataset, tf.io.decode_csv
    3.Tfrecord
        tf.train.FloatList, tf.train.Int64List, tf.train.BytesList
        tf.train.Feature, tf.train.Features, tf.train.Example
        example.SerializeToString
        tf.io.ParseSingleExample
        tf.io.ValLenFeature, tf.io.FixedLenFeature
        tf.data.TFRecordDataset, tf.io.TFRecordOptions
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

dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
print(np.arange(10))
print(dataset)
# 遍历
for item in dataset:
    print(item)

# 1.repeat epoch
# 2.get batch
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)

# interleave:
# case:文件dataset->具体数据集
dataset2 = dataset.interleave(
    # map_fn
    lambda v: tf.data.Dataset.from_tensor_slices(v),
    # cycle_length
    cycle_length=5,
    # block_length
    block_length=5,
)
for item in dataset2:
    print(item)

# dataset对元组和矩阵的支持验证
x = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array(['cat', 'dog', 'fox'])
dataset3 = tf.data.Dataset.from_tensor_slices((x, y))
print("dataset3")
print(dataset3)

for item_x, item_y in dataset3:
    print(item_x.numpy(), item_y.numpy())

dataset4=tf.data.Dataset.from_tensor_slices({"feature":x,
                                             "label":y})
print("dataset4")
for item in dataset4:
    print(item["feature"].numpy(),item["label"].numpy())

