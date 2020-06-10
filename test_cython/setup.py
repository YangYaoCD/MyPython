# coding=utf-8

"""
Author : YangYao
Date : 2020/6/1 18:23
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='Hello world app',
    ext_modules=cythonize("test.py"),
)
