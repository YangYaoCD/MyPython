# import matplotlib as mpl
# import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys
import time
from tensorflow import keras

# 版本信息
print(cv2.__version__)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
