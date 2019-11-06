import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model

def plot_decision_boundary(model, X, Y):
    # 设坐标轴宽度和高度
    #x[n,:]表示在n个数组（维）中取全部数据，直观来说，x[n,:]就是取第n集合的所有数据,
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # 生成网格点坐标矩阵(np.meshgrid)xx(1008,1030)yy(1008,1030)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    #
    ''' np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等;
        np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
        numpy中的ravel()、flatten()、squeeze()都有将多维数组转换为一维数组的功能，区别：
        ravel()：如果没有必要，不会产生源数据的副本
        flatten()：返回源数据的副本
        squeeze()：只能对维数为1的维度降维
    '''
    ravel_xx = xx.ravel()#revel.xx(1038240,)
    ravel_ = np.c_[ravel_xx, yy.ravel()]#revel_(1038240,2)
    Z = model(ravel_)#Z(1,1038240)
    Z = Z.reshape(xx.shape)
    '''    
    plt.contourf 与 plt.contour 区别：
                f：filled，也即对等高线间的填充区域进行填充（使用不同的颜色）；
                contourf：将不会再绘制等高线（显然不同的颜色分界就表示等高线本身），
    '''
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=Y.reshape(X[0, :].shape), cmap=plt.cm.Spectral)
    

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y

def load_extra_datasets():  
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)
    
    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure