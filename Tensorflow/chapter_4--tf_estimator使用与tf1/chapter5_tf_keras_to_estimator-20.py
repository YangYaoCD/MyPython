# -*- coding: utf-8 -*-
# @Time    : 2020/1/4 11:18
# @Author  : YangYao
# @Email   : 1344262070@qq.com
# @File    : chapter5_tf_keras_to_estimator-20.py
# @Software: PyCharm
'''
tf.estimator使用：
    1.Keras转estimator
    2.使用预定义的estimator
        BaseLineClassifier
        LinearClassifier
        DNNClassifier
    3.tf.feature_column做特征工程
'''

'''
API列表：
    tf.keras.estimator.to_estimator
        train,evaluate
    tf.estimator.BaseLineClassifier
    tf.estimator.LinearClassifier
    tf.estimator.DNNClassifier
    
    tf.feature_column
        categorical_column_with_vocabulary_list
        numeric_column
        indicator_column
        cross_column
    keras.layers.DenseFeatures
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
print(train_df.head())  # 默认前五条数据

y_train = train_df.pop('survived')
y_eval = eval_df.pop('survived')

print("统计量")
print(train_df.describe())

print("数据shape:")
print(train_df.shape, eval_df.shape)

print("年龄的直方图：")
train_df.age.hist(bins=20)
plt.show()
train_df.sex.value_counts().plot(kind='barh')  # barh是水平，barv是垂直
plt.show()

train_df['class'].value_counts().plot(kind='barh')  # 不用上面那种方式是因为列名是class
plt.show()

pd.concat([train_df, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh')
plt.show()

'''模型'''
categorical_columns = ['sex', 'n_siblings_spouses', 'parch', 'class',
                       'deck', 'embark_town', 'alone']  # 离散特征
numeric_columns = ['age', 'fare']  # 连续特征

feature_columns = []
for categorical_column in categorical_columns:
    vocab = train_df[categorical_column].unique()
    print(categorical_column, vocab)
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


train_dataset = make_dataset(train_df, y_train, batch_size=5)
for x, y in train_dataset.take(1):
    print(x, y)

# keras.layers.DenseFeature
print("************字符串转成了浮点型*****************")
for x, y in train_dataset.take(1):
    print(keras.layers.DenseFeatures(feature_columns)(x).numpy())

model = keras.models.Sequential([
    keras.layers.DenseFeatures(feature_columns),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(2, activation='softmax'),
])
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

'''
训练模型的方法：
    1.model.fit
    2.model->estimator->train
'''
# train_dataset = make_dataset(train_df, y_train, epochs=100)
# eval_dataset = make_dataset(eval_df, y_eval, epochs=1, shuffle=False)
# history=model.fit(train_dataset,validation_data=eval_dataset,
#                   steps_per_epoch=train_df.shape[0]//32,validation_steps=eval_df.shape[0]//32,epochs=100)

# estimator = keras.estimator.model_to_estimator(model)
# # input_fn:
# # 1.必须是function
# # 2.返回值类型  a. (features,labels)  b. dataset -> (features,labels)
# estimator.train(input_fn=lambda: make_dataset(train_df, y_train, epochs=100))  # 未完成，有bug
