import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time

data_file = open('C:/Users/yangyao/Desktop/mnist_train.csv', 'r')
data_lists = data_file.readlines()
data_file.close()
print()
