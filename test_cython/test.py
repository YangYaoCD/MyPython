# coding=utf-8

"""
Author : YangYao
Date : 2020/6/1 18:23
"""


def funcf(i):
    return i > 3

l = list([[1, 2, 3], [4, 5, 6]])
print(l)
l1=list()
for li in l:
    l2 = filter(funcf, li)
    l1.append(list(l2))
print(l1)