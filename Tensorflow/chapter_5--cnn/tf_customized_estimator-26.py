# -*- coding: utf-8 -*-
# @Time    : 2020/1/6 21:01
# @Author  : YangYao
# @Email   : 1344262070@qq.com
# @File    : tf_customized_estimator-26.py
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

ouput_dir = "tf1_customized_estimator"
if not os.path.exists(ouput_dir):
    os.mkdir(ouput_dir)


def make_dataset(data_df, label_df, epochs=10, shuffle=True,
                 batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(data_df), label_df))  # dict() 函数用于创建一个字典。https://blog.csdn.net/Strive_For_Future/article/details/92571705
    if shuffle:
        dataset = dataset.shuffle(10000)  # shuffle() 方法将序列的所有元素随机排序。
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def model_fn(features, labels, mode, params):
    # model runtime state: Train, Eval, Predict
    input_for_next_layer = tf.feature_column.input_layer(features, params['feature_columns'])
    for n_unit in params['hidden_units']:
        input_for_next_layer = tf.layers.dense(input_for_next_layer,
                                               units=n_unit, activation=tf.nn.relu)
    logits = tf.layers.dense(input_for_next_layer, params['n_classes'], activation=None)
    predicted_classes = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "class_ids": predicted_classes[:, tf.newaxis],
            "probabilities": tf.nn.softmax(logits),
            "logits": logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name="acc_op")
    metrics = {"accuracy": accuracy}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss,
                                          eval_metric_ops=metrics)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=ouput_dir,
    params={
        "feature_columns": feature_columns,
        "hidden_units": [100, 100],
        "n_classes": 2
    }
)
estimator.train(input_fn=lambda: make_dataset(train_df, y_train, epochs=100))
estimator.evaluate(lambda :make_dataset(eval_df,y_eval,epochs=1))