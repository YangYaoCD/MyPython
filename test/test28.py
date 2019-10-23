import numpy as np


def softmax(x):
    x_exp = np.exp(x)
    print(x_exp.shape)
    x_sum = np.sum(x_exp,axis=1)
    x_sum=x_sum.reshape(2,1)
    print(x_sum.shape)
    s = x_exp/x_sum
    print(s.shape)
    return s
x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))
sum=np.array([[8260.88614278],[1248.04631753]])
x_exp = np.exp(x)
# print("x.shape:"+str(x.shape))
print("sum.shape:"+str(sum.shape))
print(x_exp/sum)
np.sqrt()