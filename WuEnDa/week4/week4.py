import numpy as np
import h5py
import matplotlib.pyplot as plt
from WuEnDa.week4.testCases_v2 import *
from WuEnDa.week4.dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

#初始化参数parameters(W,b)(W1,W2,b1,b2)
def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters

#正向线性得到Z，cache(A,W,b)
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache

#正向线性激活(一个神经元整体)，传入A，W，b和对应的激活函数，输出A，cache(linear_cache(A_pre,W,b),activation_cache(Z))
def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache

#返回AL(=y^),caches(cache(linear_cache(A_pre,W,b),activation_cache(Z)),......)
def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    for l in range(1, L):#从1到L-1层用relu函数激活
        A_prev = A
        #linear_activation_forward函数返回A，cache(linear_cache(A_pre,W,b),activation_cache(Z))
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
    #第L层用sigmoid激活函数
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)
    assert (AL.shape == (1, X.shape[1]))
    return AL, caches

#计算返回损失函数J
def compute_cost(AL, Y):
    m = Y.shape[1]
    SUM=np.multiply(Y, np.log(AL))
    cost = -1 / m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)),axis=1)
    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost

'''
    dW[l]=(1/m)*dZ[l]*A[l-1].T;
    db[l]=(1/m)*np.sum(dZ[l],axis=1,keepdims=True);
    dA[L-1]=W[l].T*dZ[l];
'''
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW =np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ,axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T,dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

#反向递推单元，返回dA_prev, dW, db
'''
    dZ[l]=dA[l]*g'[l](Z[l])
'''
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache  #linear_cache(A_pre,W,b),activation_cache(Z)

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        print("dZ:"+str(dZ))
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

#反向递推模型
def L_model_backward(AL, Y, caches):
    grads = {}
    #caches(cache(linear_cache(A_pre,W,b),activation_cache(Z)),......)
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    '''!dA[l]=-np.divide(Y,A)+np.divide((1-Y),(1-A))'''
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,current_cache , "sigmoid")

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+2)],current_cache , "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        ### END CODE HERE ###

    return grads

#更新参数W，b
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)]-learning_rate*grads["dW"+str(l+1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)]-learning_rate*grads["db"+str(l+1)]
    ### END CODE HERE ###

    return parameters
parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)

print ("W1 = "+ str(parameters["W1"]))
print ("b1 = "+ str(parameters["b1"]))
print ("W2 = "+ str(parameters["W2"]))
print ("b2 = "+ str(parameters["b2"]))