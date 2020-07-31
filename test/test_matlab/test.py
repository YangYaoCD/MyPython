# coding=utf-8

"""
Author : YangYao
Date : 2020/7/11 17:02
"""
import matlab.engine
import numpy as np
engine = matlab.engine.start_matlab()
X=np.random.randn(50,50).tolist()
L,S,RMSE,error=engine.GoDec(matlab.double(X),40,40,5,nargout=4)
print(np.asarray(L))