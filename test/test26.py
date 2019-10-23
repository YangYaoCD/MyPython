import numpy as np
import time

a=np.random.rand(1000000)
b=np.random.rand(1000000)

tic=time.time()
c=np.dot(a,b)
toc=time.time()
print("向量版本："+str(1000*(toc-tic))+"ms")
print("答案是："+str(c))

d=np.array([[1,2,3,4]])
print("d的shape:"+str(d.shape))
d.reshape(4,1)
print("d的转置:"+str(d.transpose()))
print("reshapehou的shape:"+str(d.shape))
assert (d.shape==(1,4))
print(d)