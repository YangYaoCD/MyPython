import numpy as np
from math import *
import matplotlib.pyplot as plt

x=np.linspace(0,1,10)
y=7*np.sin(2*np.pi*10*x)
print(y)

ax = plt.subplot(111)
width = 10
hight = 3
ax.arrow(0, 0, 0, hight, width=0.01, head_width=0.1, head_length=0.3, length_includes_head=True, fc='k', ec='k')
ax.arrow(0, 0, width, 0, width=0.01, head_width=0.1, head_length=0.3, length_includes_head=True, fc='k', ec='k')

ax.axes.set_xlim(-0.5, width + 0.2)
ax.axes.set_ylim(-0.5, hight + 0.2)

plotdict = {'dx': x, 'dy': y}
ax.plot('dx', 'dy', 'bD-', data=plotdict)
ax.plot(x,y,'r^-')
plt.title('Original wave')
print("原函数：{}".format(y))

# '''傅里叶变换'''
# def fft(x):
#     x = np.asarray(x, dtype=float)
#     N = x.shape[0]
#     n = np.arange(N)
#     k = n.reshape((N, 1))
#     M = np.exp(-2j * np.pi * k * n / N)
#     f = np.dot(M, x)
#     return f
#
# ##########C++里没有numpy包，需要循环遍历####################
# def fft1(a):
#     N = len(a)
#     f = []
#     for k in range(N):
#         F = 0
#         for m in range(N):
#             F += a[m] * e**(-2j * pi * (m*k) / N)
#         f.append(F)
#     return f
#
# ############三角函数代替指数函数#################
# def fft2(data):
#     length=len(data)
#     fftdata = [0+0j]*length
#     print(fftdata)
#     for k in range(length):
#         for i in range(length):
#             fftdata[k] += data[i]*(cos(-2*pi*k*i/length)+1j*sin(-2*pi*k*i/length))
#     return fftdata
#
# '''傅里叶反变换'''
# def ifft1(a):
#     N = len(a)
#     f = []
#     for k in range(N):
#         F = 0
#         for m in range(N):
#             F += a[m] * e**(2j * pi * (m*k) / N)/N
#         f.append(F)
#     return f
#
# ############三角函数代替指数函数#################
# def ifft2(data):
#     length=len(data)
#     ifftdata = [0+0j]*length
#     print(ifftdata)
#     for k in range(length):
#         for i in range(length):
#             ifftdata[k] += data[i]*(cos(2*pi*k*i/length)+1j*sin(2*pi*k*i/length))/length
#     return ifftdata
#
#
# yf=ifft1(fft(y))
# print("反变换后：{}".format(yf))

# plt.subplot(222)
# plt.plot(x,y)
# plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')  #注意这里的颜色可以查询颜色代码表



# plt.subplot(223)
# plt.plot(x,yf)
# plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r')