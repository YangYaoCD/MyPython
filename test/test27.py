import numpy as np

def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis=0, keepdims=True)
    print("x_norm"+str(x_norm))
    x = x / x_norm
    return x

def mathMethod(x):
    sum=x.sum(axis=0)
    print("sum:" + str(sum))
    ans=x/sum
    return ans

X=np.array([[56.0,0.0,4.4,68.0],[1.2,104.0,52.0,8.0],[1.8,135.0,99.0,0.9]])
print("X:"+str(X))
print("normalizeRows:"+str(normalizeRows(X)))
print("mathMethod:"+str(mathMethod(X)))



