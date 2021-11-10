import numpy as np
import pandas as pd

# a = np.arange(10)
# b = a[2:7:2]   # 从索引 2 开始到索引 7 停止，间隔为 2
# print(a)
# print(b)

# a=np.zeros((2,3))
# print(a.shape)
# print(type(a.shape))

# 取对角线元素
# di = np.diag_indices(4)
# di=(np.array([0,2,3]),np.array([3,2,1]))
# a = np.arange(16).reshape(4, 4)
# print(a)
# print(a[di])
# a[di]=100
# print(a)

# a=np.array([1,2,3,4,5,6,7,8,9,10])
# a=a.reshape((5,2))
# print(a)

# print(np.sort(a,axis=0))
# print(np.sort(a,axis=1))
#
# print(np.argsort(a,axis=0))
# print(np.argsort(a,axis=1))
# print(a[-5:,0])
# print(a[np.argsort(a,axis=1)])

# df=pd.DataFrame(np.array([[1, 1], [2, 10], [3, 100], [4, 100]]),columns=['a', 'b'])
# print(df)
# df.quantile(.1)

mi1=[0.697,0.774,0.634,0.608,0.556,0.403,0.481,0.437]
tang1=[0.46,0.376,0.264,0.318,0.215,0.237,0.149,0.211]
print(np.mean(mi1))
print(np.var(mi1))
print(np.mean(tang1))
print(np.var(tang1))

mi2=[0.666,0.243,0.245,0.343,0.639,0.657,0.36,0.593,0.719]
tang2=[0.091,0.267,0.057,0.099,0.161,0.198,0.37,0.042,0.103]
print()
print(np.mean(mi2))
print(np.var(mi2))
print(np.mean(tang2))
print(np.var(tang2))
