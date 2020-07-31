# coding: utf-8

# In[1]:


from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf

# In[2]:


path_1 = "D:/notebook/Object_Detection/SalinasA/SalinasA.mat"
# path_2 = "D:/notebook/Object_Detection/SalinasA/SalinasA_corrected.mat"
path_2="D:/notebook/Object_Detection/sandie/sandiego.mat"
path_3 = "D:/notebook/Object_Detection/SalinasA/SalinasA_gt.mat"
file_1 = loadmat(path_1)
file_2 = loadmat(path_2)
file_3 = loadmat(path_3)

# In[3]:


salinas = file_1['salinasA']
# salinas_corrected = file_2['salinasA_corrected']
salinas_corrected=file_2['sandiego_reflectance_mat']
salinas_gt = file_3['salinasA_gt']
print(file_1.keys())
print(salinas.shape)
print(file_2.keys())
print(salinas_corrected.shape)
print(file_3.keys())
print(salinas_gt.shape)


# In[4]:


def show_single_image(img_arr):
    plt.imshow(img_arr, cmap="binary")
    plt.show()


# In[5]:


show_single_image(salinas_gt)

# In[6]:


salinas_corrected = np.array(salinas_corrected, dtype=np.float64)
salinas_gt = np.array(salinas_gt, dtype=np.float64)


# In[7]:


def LRX(maxtric, threshold, H1, H0, r1, r2):
    N = r2 * r2 - r1 * r1
    output_arr = np.zeros((maxtric.shape[0], maxtric.shape[1]), dtype=np.float64)
    count = 0
    count_error=0

    # 中心点(m,n)扫描
    all_pic = [(i, j) for i in range(maxtric.shape[0]) for j in range(maxtric.shape[1])]
    # all_pic=[(200,200),[150,150]]
    for (m, n) in all_pic:
        starttime = time.time()
        outer = [(i, j) for i in range(m - r2 // 2, m + r2 // 2 + 1) for j in range(n - r2 // 2, n + r2 // 2 + 1)]
        inner = [(i, j) for i in range(m - r1 // 2, m + r1 // 2 + 1) for j in range(n - r1 // 2, n + r1 // 2 + 1)]
        background = list(set(outer) - set(inner))

        # miu代表背景光谱向量的均值
        avera = np.zeros((maxtric.shape[2], 1))
        for k in range(maxtric.shape[2]):
            sum = 0.0
            for (i, j) in background:
                # 判断是否边界有值
                if i < 0 or i >= maxtric.shape[0]:
                    if j < 0 or j >= maxtric.shape[1]:
                        sum += maxtric[2 * m - i, 2 * n - j, k]
                    else:
                        sum += maxtric[2 * m - i, j, k]

                elif j < 0 or j >= maxtric.shape[1]:
                    sum += maxtric[i, 2 * n - j, k]

                else:
                    sum += maxtric[i, j, k]

            avera[k] = sum / N
        avera = np.array(avera).reshape(-1, 1)

        # K 为背景的协方差矩阵
        K = np.zeros((maxtric.shape[2], maxtric.shape[2]), dtype=np.float64)
        for (i, j) in background:
            # 判断是否边界有值
            if i < 0 or i >= maxtric.shape[0]:
                if j < 0 or j >= maxtric.shape[1]:
                    arr = np.subtract(maxtric[2 * m - i, 2 * n - j, :].reshape(-1, 1), avera)
                else:
                    arr = np.subtract(maxtric[2 * m - i, j, :].reshape(-1, 1), avera)
            elif j < 0 or j >= maxtric.shape[1]:
                arr = np.subtract(maxtric[i, 2 * n - j, :].reshape(-1, 1), avera)
            else:
                arr = np.subtract(maxtric[i, j, :].reshape(-1, 1), avera)

            K += np.dot(arr, arr.T) / N

        # LRX算法判断式
        # print("K的秩="+str(np.linalg.matrix_rank(K)))
        try:
            output_arr[m][n] = LRX_discriminant(
                np.matrix(np.subtract(maxtric[m, n, :].reshape(-1, 1), avera)),
                np.matrix(np.linalg.inv(K)))
        except:
            count_error+=1
            print('不可逆' + str(count_error))
            # if count_error%1000==0:
            #     print('不可逆'+str(count_error))

        count += 1
        endtime = time.time()
        if count % 1000 == 0:
            print("进度：" + '{:.2f}%'.format(100 * count / (maxtric.shape[0] * maxtric.shape[1])), end="\t")
            print("大概还剩" + str(round(endtime - starttime, 2) * ((maxtric.shape[0] * maxtric.shape[1]) - count)) + "秒！")
    return output_arr


'''LRX算法判断'''


def LRX_discriminant(arr, K):
    lrx = arr.T * K * arr
    return lrx


# In[ ]:
ind_1=[i for i in range(106)]
ind_2=[i for i in range(113,152)]
ind_3=[i for i in range(166,223)]
ind=ind_1+ind_2+ind_3
a=tf.gather(salinas_corrected,axis=2,indices=ind)
a=np.array(a)
print("a.shape="+str(a.shape))

lrx_array = LRX(a, 10, 1, 0, 1, 3)
print(lrx_array)
