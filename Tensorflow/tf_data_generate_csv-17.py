'''读写csv文件'''

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

'''写csv文件'''
output_dir = "gennerate_csv"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def save_to_csv(output_dir, data, name_prefix,
                header=None, n_parts=10):
    path_format = os.path.join(output_dir, "{}_{:02d}.csv")
    filenames = []
    for file_idx, row_indices in enumerate(
            np.array_split(np.arange(len(data)), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filenames.append(part_csv)
        with open(part_csv, "wt", encoding='utf-8') as f:
            if header is not None:
                f.write(header + "\n")
            for row_index in row_indices:
                # repr() 函数将对象转化为供解释器读取的形式,返回一个对象的 string 格式。
                f.write(",".join([repr(col) for col in data[row_index]]))
                f.write("\n")
    return filenames


train_data = np.c_[x_train_scaled, y_train]
valid_data = np.c_[x_valid_scaled, y_valid]
test_data = np.c_[x_test_scaled, y_test]
header_cols = housing.feature_names + ["MidianHouseValue"]
header_str = ','.join(header_cols)

train_filenames = save_to_csv(output_dir, train_data,
                              "train", header=header_str, n_parts=20)
valid_filenames = save_to_csv(output_dir, valid_data,
                              "valid", header=header_str, n_parts=10)
test_filenames = save_to_csv(output_dir, test_data,
                             "test", header=header_str, n_parts=10)

import pprint

print("train filenames")
pprint.pprint(train_filenames)

'''读csv文件'''
# 1. filenames -> datasets
# 2. read file -> dataset -> datasets -> merge
# 3. parse csv

print("******读取文件******")
filename_dataset = tf.data.Dataset.list_files(train_filenames)
for filename in filename_dataset:
    print(filename)
n_readers = 5
dataset = filename_dataset.interleave(
    lambda filename: tf.data.TextLineDataset(filename).skip(1),
    cycle_length=n_readers,
)
for line in dataset.take(15):
    print(line.numpy())

# tf.io.decode_csv(str,record_defaults)(3.parse csv)
sample_str = '1,2,3,4,5'
record_defaults = [tf.constant(0, dtype=tf.float32),
                   0,
                   np.nan,
                   "hello",
                   tf.constant([])]
parsed_fields = tf.io.decode_csv(sample_str, record_defaults)
print(parsed_fields)

try:
    parsed_fields = tf.io.decode_csv(',,,', record_defaults)
except tf.errors.InvalidArgumentError as ex:
    print(ex)


def parse_csv_line(line, n_fields=9):
    defs = [tf.constant(np.nan)] * n_fields
    parsed_fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])
    return x, y


'''完整读取转换函数'''


def csv_reader_dataset(filenames, n_readers=5,
                       batch_size=32, n_parse_threads=5,
                       shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    # interleave将多个dataset转为一个dataset
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename).skip(1),
        cycle_length=n_readers
    )
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_csv_line, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset


batch_size = 32
train_set = csv_reader_dataset(train_filenames, batch_size=batch_size)
valid_set = csv_reader_dataset(valid_filenames, batch_size=batch_size)
test_set = csv_reader_dataset(test_filenames, batch_size=batch_size)
# for x_batch, y_batch in train_set.take(2):
#     print("x:")
#     pprint.pprint(x_batch)
#     print("y:")
#     pprint.pprint(y_batch)

model=keras.models.Sequential([
    keras.layers.Dense(30,input_shape=[8],
                       activation='relu'),
    keras.layers.Dense(1),
])
model.compile(loss="mse",
              optimizer="sgd")

callbacks = [
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)
]
history = model.fit(train_set, epochs=100,
                    validation_data=valid_set,
                    validation_steps=3870,
                    steps_per_epoch=111600,
                    callbacks=callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.title(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    plt.gca().set_ylim(0, 10)
    plt.show()


plot_learning_curves(history)

model_evaluate = model.evaluate(x_test_scaled, y_test)
print("model_evaluate" + str(model_evaluate))
