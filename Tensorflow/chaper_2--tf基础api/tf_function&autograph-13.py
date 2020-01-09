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


# tf.function and autograph
def scaled_elu(z, scale=1.0, alpha=1.0):
    # z >= 0 ? scale*z : alpha*tf.nn.elu(z)
    is_positive = tf.greater_equal(z, 0.0)
    return scale * tf.where(is_positive, z, alpha * tf.nn.elu(z))


print(scaled_elu(tf.constant(-3.0)))
print(scaled_elu(tf.constant([-3., -2., 3])))

scaled_elu_tf = tf.function(scaled_elu)
print(scaled_elu_tf(tf.constant(-3.)))
print(scaled_elu_tf(tf.constant([-3., -2., 0., 1.])))

print(scaled_elu_tf.python_function is scaled_elu)


# %timeit scaled_elu(tf.random.normal((1000,1000)))
# %timeit scaled_elu_tf(tf.random.normal((1000,1000)))

# 1+1/2+1/2^2+1/2^3+1/2^4+...+1/2^n
@tf.function
def converge_to_2(n_iters):
    total = tf.constant(0.)
    increment = tf.constant(1.)
    for _ in range(n_iters):
        total += increment;
        increment /= 2.0
    return total


print(converge_to_2(20))


def display_tf_code(func):
    code = tf.autograph.to_code(func)
    from IPython.display import display, Markdown
    display(Markdown('******PYTHON\n{}\n**********'.format(code)))


display_tf_code(scaled_elu)

# 转tf.function不能在函数内初始化Variable，必须在外面初始化
var = tf.Variable(0.)


@tf.function
def add_21():
    return var.assign_add(21)


print(add_21())


@tf.function(input_signature=[tf.TensorSpec([None], tf.int32, name='x')])
def cube(z):
    return tf.pow(z, 3)


try:
    print(cube(tf.constant([1., 2., 3.])))
except ValueError as ex:
    print(ex)
print(cube(tf.constant([1, 2, 3])))

'''
    1.@tf.function py func -> tf grapg
    2.get_concrete_function -> add input signature -> SavedModel
'''
cube_func_int32 = cube.get_concrete_function(tf.TensorSpec([None], tf.int32))
print(cube_func_int32)
print(cube_func_int32.graph)
print(str(cube_func_int32.graph.get_operations()[1]))

