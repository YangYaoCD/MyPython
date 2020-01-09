# import matplotlib as mpl
# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time

from tensorflow import keras

# 版本信息
print(tf.__version__)
print(sys.version_info)
for module in np, sklearn, tf, keras:
    print(module.__name__ + ":" + module.__version__)
