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

df=pd.DataFrame(np.array([[1, 1], [2, 10], [3, 100], [4, 100]]),columns=['a', 'b'])
print(df)
# df.quantile(.1)
