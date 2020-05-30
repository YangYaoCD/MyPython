import numpy as np

X = np.array([[56.0, 0.0, 4.4, 68.0], [1.2, 104.0, 52.0, 8.0], [1.8, 135.0, 99.0, 0.9]])
sum = X.sum(axis=0)
percentage = X / sum.reshape(1, 4) * 100
# 不用reshape也可以
print(percentage)
print(sum.reshape(1, 4))
