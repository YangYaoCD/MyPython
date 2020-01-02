'''
tfrecord文件格式
    -> tf.train.Example
        -> tf.train.Features -> {"key" : tf.train.Feature}
            -> tf.train.Feature -> tf.train.ByteList/FloatList/IntList
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

# 读取代码
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

from sklearn.model_selection import train_test_split

x_train_all, x_test, y_train_all, y_test = train_test_split(
    housing.data, housing.target, random_state=7, test_size=0.5)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state=11, test_size=0.8)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

favorite_books = [name.encode('utf-8') for name in ["machine learning", "ccl"]]
favorite_books_byteLIist = tf.train.BytesList(value=favorite_books)
print(favorite_books_byteLIist)

hours_floatList = tf.train.FloatList(value=[15.5, 9.5, 7.0, 8.0])
print(hours_floatList)

age_int64list = tf.train.Int64List(value=[42])
print(age_int64list)

features = tf.train.Features(
    feature={
        "favorite_books": tf.train.Feature(
            bytes_list=favorite_books_byteLIist),
        "hours": tf.train.Feature(
            float_list=hours_floatList),
        "age": tf.train.Feature(
            int64_list=age_int64list),
    }
)
print(features)

print("-------------------Example--------------------")
example = tf.train.Example(features=features)
print(example)

serialized_example = example.SerializeToString()  # 序列化
print(serialized_example)

'''写文件'''
output_dir = 'tfrecord_basic'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
filename = "test.tfrecords"
filename_fullpath = os.path.join(output_dir, filename)
with tf.io.TFRecordWriter(filename_fullpath) as writer:
    for i in range(3):
        writer.write(serialized_example)

'''读取文件'''
print("解析文件")
dataset_zip = tf.data.TFRecordDataset([filename_fullpath])
for serialized_example_tensor in dataset_zip:
    print(serialized_example_tensor)

expected_features = {
    "favorite_books": tf.io.VarLenFeature(dtype=tf.string),
    "hours": tf.io.VarLenFeature(dtype=tf.float32),
    "age": tf.io.FixedLenFeature([], dtype=tf.int64)
}

dataset_zip = tf.data.TFRecordDataset([filename_fullpath])
for serialized_example_tensor in dataset_zip:
    example = tf.io.parse_single_example(
        serialized_example_tensor,
        expected_features
    )
    books = tf.sparse.to_dense(example["favorite_books"], default_value=b"")
    for book in books:
        print(book.numpy().decode("UTF-8"))
    # print(example)

'''存压缩文件'''
filename_fullpath_zip = filename_fullpath + '.zip'
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter(filename_fullpath_zip, options) as writer:
    for i in range(3):
        writer.write(serialized_example)

'''解压缩文件'''
print("解压缩并解析TFRecord文件")
dataset_zip = tf.data.TFRecordDataset([filename_fullpath_zip], compression_type="GZIP")
for serialized_example_tensor in dataset_zip:
    example = tf.io.parse_single_example(
        serialized_example_tensor,
        expected_features
    )
    books = tf.sparse.to_dense(example["favorite_books"], default_value=b"")
    for book in books:
        print(book.numpy().decode("UTF-8"))
    # print(example)
