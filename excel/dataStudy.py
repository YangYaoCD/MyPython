# coding=utf-8

"""
Author : YangYao
Date : 2020/5/30 20:24
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

df = pd.read_excel('dataStudy.xlsx', 'Sheet1')

# 直方图
fig = plt.figure()
ax = fig.add_subplot(111)
df['Age'].hist(bins=10)
plt.title('Age show')
plt.xlabel('Age')
plt.ylabel('攻击力')
plt.show()
