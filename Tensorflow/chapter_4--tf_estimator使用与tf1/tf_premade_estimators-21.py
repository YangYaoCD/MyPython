# -*- coding: utf-8 -*-
# @Time    : 2020/1/4 20:41
# @Author  : YangYao
# @Email   : 1344262070@qq.com
# @File    : tf_premade_estimators-21.py
# @Software: PyCharm
'''
    预定义estimator使用
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

train_file = "./titanic/train.csv"
eval_file = "./titanic/eval.csv"

train_df = pd.read_csv(train_file)
eval_df = pd.read_csv(eval_file)

y_train = train_df.pop('survived')
y_eval = eval_df.pop('survived')

'''模型'''
categorical_columns = ['sex', 'n_siblings_spouses', 'parch', 'class',
                       'deck', 'embark_town', 'alone']  # 离散特征
numeric_columns = ['age', 'fare']  # 连续特征

feature_columns = []
for categorical_column in categorical_columns:
    vocab = train_df[categorical_column].unique()
    feature_columns.append(tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            categorical_column, vocab)))

for numeric_column in numeric_columns:
    feature_columns.append(
        tf.feature_column.numeric_column(numeric_column, dtype=tf.float32))


def make_dataset(data_df, label_df, epochs=10, shuffle=True,
                 batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(data_df), label_df))  # dict() 函数用于创建一个字典。https://blog.csdn.net/Strive_For_Future/article/details/92571705
    if shuffle:
        dataset = dataset.shuffle(10000)  # shuffle() 方法将序列的所有元素随机排序。
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset


# ouput_dir = 'baseline_model'
# if not os.path.exists(ouput_dir):
#     os.mkdir(ouput_dir)
#
# baseline_estimator = tf.estimator.BaselineClassifier(
#     model_dir=ouput_dir,
#     n_classes=2)
# baseline_estimator.train(input_fn=lambda: make_dataset(train_df, y_train, epochs=100))
# eval = baseline_estimator.evaluate(
#     input_fn=lambda: make_dataset(eval_df, y_eval, epochs=1, shuffle=False, batch_size=20))
# print(eval)

dnn_output_dir = './dnn_model'
if not os.path.exists(dnn_output_dir):
    os.mkdir(dnn_output_dir)

dnn_estimator = tf.estimator.DNNClassifier(
    model_dir=dnn_output_dir, n_classes=2,
    feature_columns=feature_columns, hidden_units=[128, 128],
    activation_fn=tf.nn.relu, optimizer='Adam')
dnn_estimator.train(input_fn=lambda: make_dataset(train_df, y_train, epochs=100))

eval = dnn_estimator.evaluate(input_fn=lambda: make_dataset(
    eval_df, y_eval, epochs=1, shuffle=False))
print(eval)
