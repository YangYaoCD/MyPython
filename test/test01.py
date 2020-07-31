# import matplotlib as mpl
# import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras

# 版本信息
# print(cv2.__version__)
# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
@tf.function
def square_if_positive(x):
  if x > 0:
    x = x * x
  else:
    x = 0
  return x


print('square_if_positive(2) = {}'.format(square_if_positive(tf.constant(2))))
print('square_if_positive(-2) = {}'.format(square_if_positive(tf.constant(-2))))
print(...)
print(tf.keras.experimental.CosineDecayRestarts)

a=np.array([[1,0]])
b=np.array([[0],[2]])
print(np.dot(b,a))