import numpy as np
import math
from sklearn.metrics import f1_score


def load_data(fname,delimiter,dtype=None):
    data=np.genfromtxt(fname,dtype,delimiter=delimiter)
    X=data[1:,2:]   # 以逗号为分隔，前面为行，后面为列
    Y=data[1:,1]
    m=len(Y)    # 样本数
    return X,Y,m

def computeDistance(train_setX,train_setY,test_data):
    di=np.diag_indices(len(train_setX))     # 生成数组对角线索引元组

    distance = np.zeros(train_setX.shape)
    for j in range(len(train_setX)):
        distance[j]=train_setX[j]-test_data
    distance2=np.dot(distance,np.transpose(distance))[di]
    distance2=np.sqrt(distance2)
    distance2=np.hstack((distance2.reshape(-1,1),train_setY.reshape(-1,1)))     # 将距离向量和训练集Y向量合并（向量转置用reshape，用transport无效）
    sort_indices=np.argsort(distance2[...,0])       # 以距离列向量的值排序得到排序后的索引(高->低)
    distance2=distance2[sort_indices]

    return distance2

def normal_predict(sorted_distance,k,i,predict_results):
    arraytolist=sorted_distance[:k,1].tolist()
    types=set(arraytolist)       # 罗列出列表中不同的元素
    dict={}     # 利用字典统计出不同元素的个数

    for item in types:
        dict.update({item:arraytolist.count(item)})
    print(dict)

    try:
        if dict[b'B']>dict[b'M']:
            result = 'B'
        else:
            result = 'M'
    except KeyError as e:
        if b'B' in types:
            result = 'B'
        else:
            result = 'M'

    print('测试样本{}的预测结果为：{}'.format(i,result))
    predict_results[i]=(result)

def computePerformance(test_setY,predict_results):
    print("\nF1分数={}".format(f1_score(test_setY, predict_results, average='micro')))

if __name__=="__main__":
    X,Y,m=load_data("Breast Cancer_data.csv",",")
    temp = math.ceil(m * 3 / 4)
    train_setX = X[:temp]
    test_setX = X[temp + 1:]
    train_setY = Y[:temp]
    test_setY = Y[temp + 1:]
    train_setX = train_setX.astype(np.float_)  # 把bytes数据类型转换为float64
    test_setX = test_setX.astype(np.float_)

    predict_results = [0]*len(test_setY)  # 对测试集的预测结果(生成固定长度的列表)

    for i in range(test_setX.shape[0]):     # 对测试集逐个预测
        sorted_distance=computeDistance(train_setX,train_setY,test_setX[i])
        normal_predict(sorted_distance,7,i,predict_results)

    test_setY=test_setY.tolist()
    test_setY=[str(i,encoding = "utf-8") for i in test_setY]    # bytes转str

    # 计算性能
    computePerformance(test_setY,predict_results)
