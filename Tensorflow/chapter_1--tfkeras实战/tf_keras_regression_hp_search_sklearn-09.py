'''

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

# 版本信息
print(tf.__version__)
print(sys.version_info)
for module in mpl, np, sklearn, tf, keras:
    print(module.__name__ + ":" + module.__version__)

# 读取代码
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

from sklearn.model_selection import train_test_split

x_train_all, x_test, y_train_all, y_test = train_test_split(
    housing.data, housing.target, random_state=7)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state=11)
print("x_train", x_train.shape, y_train.shape)
print("x_valid", x_valid.shape, y_valid.shape)
print("x_test", x_test.shape, y_test.shape)

# 归一化训练、验证、测试数据 x=(x-u)/std（u：均值，std：方差）
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

'''
RandomizedSearchCV
    1.转化为skelearn的model（KerasRegressor或KerasClassifier）
    2.定义参数集合
    3.搜索参数
'''


def build_model(hidden_layers=1, layer_size=30, learning_rate=3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(layer_size, activation='relu', input_shape=x_train.shape[1:]))
    for _ in range(hidden_layers - 1):
        model.add(keras.layers.Dense(layer_size, activation='relu'))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model


callbacks = [
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)
]
from scipy.stats import reciprocal

# f(x)=1/(x*log(b/a)) a<=x<=b
param_distribution = {
    "hidden_layers": [1, 2, 3, 4],
    "layer_size": np.arange(1, 100),
    "learning_rate": reciprocal(1e-4, 1e-2)
}

'''
为什么查找超参数时只训练7k多？
    cross_validation：训练集分为n份，n-1训练，最后一份验证
'''
from sklearn.model_selection import RandomizedSearchCV

sklearn_model = keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_model)
random_search_cv = RandomizedSearchCV(sklearn_model, param_distribution,
                                      n_iter=10,
                                      n_jobs=1)
random_search_cv.fit(x_train_scaled, y_train, epochs=10, validation_data=(x_valid_scaled, y_valid),
                     callbacks=callbacks)

print("random_search_cv.best_params_=" + str(random_search_cv.best_params_))
print("random_search_cv.best_score_=" + str(random_search_cv.best_score_))
print("random_search_cv.best_estimator_" + str(random_search_cv.best_estimator_))

model = random_search_cv.best_estimator_.model
model_evaluate = model.evaluate(x_test_scaled, y_test)
print(model_evaluate)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.title("sklearn_model")
    plt.gca().set_ylim(0, 1)
    plt.show()

# plot_learning_curves(history)
