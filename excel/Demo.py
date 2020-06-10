# coding=utf-8

"""
Author : YangYao
Date : 2020/5/30 20:36

xls文件读写
"""

# python里面操作excel
import xlrd

data = xlrd.open_workbook('dataStudy.xlsx')
# 获取特定的excel表格页面
table = data.sheets()[0]
# 打印数据内容
# 行数
print(table.nrows)
# 列数
print(table.ncols)

# 循环取所有值
for i in range(table.nrows):
    print(table.row_values(i))

# 数据可视化包
import pygal

# 雷达图 Radar
radar_row = pygal.Radar()

# 每个顶点
title = table.row_values(0)
radar_row.x_labels = title[1:]

# 添加数据
for i in range(table.nrows):
    if i >= 1:
        radar_row.add(table.row_values(i)[0], table.row_values(i)[1:])

radar_row.render_to_file('demo.html')

import xlwt
