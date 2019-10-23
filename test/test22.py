import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft,ifft
# 正反傅里叶变换

'''read file 
fin=open("para.txt")
a=[]
for i in fin:
  a.append(float(i.strip()))
a=np.array(a)
a=a.reshape(9,3)
'''


x=np.linspace(0,1,1400)
y=7*np.sin(2*np.pi*10*x)+2*np.sin(2*np.pi*50*x)

y1=abs(fft(y))
print(y1)

yf=y1/len(x)           #归一化处理
yf1 = yf[range(int(len(x)/2))]  #由于对称性，只取一半区间
y2=(ifft(fft(y)))

ax = plt.subplot(111)
plotdict = {'dx': x, 'dy': y}
ax.plot('dx', 'dy', 'bD-', data=plotdict)


# ax.plot(x,y1,'r^-')
ax.plot(x,y2,'k')

plt.show()