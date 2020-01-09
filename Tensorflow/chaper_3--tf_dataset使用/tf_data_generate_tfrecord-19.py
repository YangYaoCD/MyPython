'''csv文件转tfrecord文件'''

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

source_dir = "./gennerate_csv/"


def get_filenames_by_prefix(source_dir, prefix_name):
    all_files = os.listdir(source_dir)
    result = []
    for filename in all_files:
        if filename.startswith(prefix_name):
            result.append(os.path.join(source_dir, filename))
    return result


train_filenames = get_filenames_by_prefix(source_dir, "train")
valid_filenames = get_filenames_by_prefix(source_dir, "valid")
test_filenames = get_filenames_by_prefix(source_dir, "test")

import pprint

pprint.pprint(train_filenames)
pprint.pprint(valid_filenames)
pprint.pprint(test_filenames)


def parse_csv_line(line, n_fields=9):
    defs = [tf.constant(np.nan)] * n_fields
    parsed_fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])
    return x, y


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


def serialize_example(x, y):
    '''Converts x,y to tf.train.Example and Serialize'''
    input_features = tf.train.FloatList(value=x)
    label = tf.train.FloatList(value=y)
    features = tf.train.Features(
        feature={
            "input_features": tf.train.Feature(float_list=input_features),
            "label": tf.train.Feature(float_list=label)
        }
    )
    example = tf.train.Example(features=features)
    return example.SerializeToString()


def csv_dataset_to_tfrecords(base_filename, dataset,
                             n_shards, steps_per_shard,
                             compression_type=None):
    options = tf.io.TFRecordOptions(compression_type=compression_type)
    all_filenames = []
    for shard_id in range(n_shards):
        filename_fullpath = '{}_{:05d}-of-{:05d}'.format(
            base_filename, shard_id, n_shards)
        with tf.io.TFRecordWriter(filename_fullpath, options) as writer:
            for x_batch, y_batch in dataset.take(steps_per_shard):
                for x_example, y_example in zip(x_batch, y_batch):
                    writer.write(serialize_example(x_example, y_example))
        all_filenames.append(filename_fullpath)
    return all_filenames


n_shards = 20
train_steps_per_shard = 11610 // batch_size // n_shards
valid_steps_per_shard = 3880 // batch_size // n_shards
test_steps_per_shard = 5170 // batch_size // n_shards

output_dir = "generate_tfrecords"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

train_basename = os.path.join(output_dir, "train")
valid_basename = os.path.join(output_dir, "valid")
test_basename = os.path.join(output_dir, "test")

train_tfrecord_fileenames = csv_dataset_to_tfrecords(
    train_basename, train_set, n_shards, train_steps_per_shard, None)
valid_tfrecord_fileenames = csv_dataset_to_tfrecords(
    valid_basename, valid_set, n_shards, valid_steps_per_shard, None)
test_tfrecord_fileenames = csv_dataset_to_tfrecords(
    test_basename, test_set, n_shards, test_steps_per_shard, None)

'''生成压缩的'''
output_dir = "generate_tfrecords_zip"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

train_basename = os.path.join(output_dir, "train")
valid_basename = os.path.join(output_dir, "valid")
test_basename = os.path.join(output_dir, "test")

train_tfrecord_fileenames = csv_dataset_to_tfrecords(
    train_basename, train_set, n_shards, train_steps_per_shard, compression_type="GZIP")
valid_tfrecord_fileenames = csv_dataset_to_tfrecords(
    valid_basename, valid_set, n_shards, valid_steps_per_shard, compression_type="GZIP")
test_tfrecord_fileenames = csv_dataset_to_tfrecords(
    test_basename, test_set, n_shards, test_steps_per_shard, compression_type="GZIP")

print("生成文件文件名")
pprint.pprint(train_tfrecord_fileenames)
pprint.pprint(valid_tfrecord_fileenames)
pprint.pprint(test_tfrecord_fileenames)

"解析example"
expected_features = {
    "input_features": tf.io.FixedLenFeature([8], dtype=tf.float32),
    "label": tf.io.FixedLenFeature([1], dtype=tf.float32)
}


def parse_example(serialized_example):
    example = tf.io.parse_single_example(serialized_example,
                                         expected_features)
    return example["input_features"], example["label"]


def tfrecords_reader_dataset(filenames, n_readers=5,
                             batch_size=32, n_parse_threads=5,
                             shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    # interleave将多个dataset转为一个dataset
    dataset = dataset.interleave(
        lambda filename: tf.data.TFRecordDataset(filename, compression_type="GZIP"),
        cycle_length=n_readers
    )
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_example, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset


batch_size = 32
tfrecords_train_set = tfrecords_reader_dataset(train_tfrecord_fileenames, batch_size=batch_size)
tfrecords_valid_set = tfrecords_reader_dataset(valid_tfrecord_fileenames, batch_size=batch_size)
tfrecords_test_set = tfrecords_reader_dataset(test_tfrecord_fileenames, batch_size=batch_size)

"数据集的使用"
model = keras.models.Sequential([
    keras.layers.Dense(30, input_shape=[8],
                       activation='relu'),
    keras.layers.Dense(1),
])
model.compile(loss="mse",
              optimizer="sgd")

callbacks = [
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)
]
history = model.fit(tfrecords_train_set, epochs=100,
                    validation_data=tfrecords_valid_set,
                    validation_steps=3870 // batch_size,
                    steps_per_epoch=111600 // batch_size,
                    callbacks=callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.title(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learning_curves(history)

model_evaluate = model.evaluate(tfrecords_test_set, steps=5160 // batch_size)
print("model_evaluate" + str(model_evaluate))
