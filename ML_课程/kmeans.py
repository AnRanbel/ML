import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 载入数据集
def load_dataset(filename):
    df = pd.read_csv(filename)
    x_mat = np.array(df.iloc[:, 1: -1])
    y_mat = np.array(df.iloc[:, -1])
    return x_mat, y_mat


# 计算欧几里得距离
def eclud_dist(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))

# 随机初始化k个聚类中心
def rand_centroids(dataset, k):
    n = dataset.shape[1]
    # 初始化k个聚类中心坐标为0向量
    centroids = np.zeros((k, n))
    for j in range(n):
        min_j = min(dataset[:, j])
        max_j = max(dataset[:, j])
        range_j = float(max_j - min_j)
        centroids[:, j] = min_j + range_j * np.random.rand(k)
    return centroids


# k-means 聚类算法
def k_means(dataset, k, dist_method=eclud_dist, init_centroids=rand_centroids):
    """
    :param dataset:         样本数据集
    :param k:               聚类中心数目
    :param dist_method:     计算距离的方法
    :param init_centroids:  初始化聚类中心的方法
    :return: centroids：    最终确定的 k个聚类中心
             cluster_assment:  样本对应聚类中心下标列表
    """
    m = dataset.shape[0]   # 样本数量
    # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
    cluster_assments = np.zeros((m, 2))
    # 初始化k个聚类中心
    centroids = init_centroids(dataset, k)
    cluster_changed = True   # 用来判断聚类是否已经收敛
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            # 把每一个数据点划分到离它最近的中心点
            min_dist = np.inf
            min_index = -1
            for j in range(k):
                dist_ji = dist_method(centroids[j, :], dataset[i, :])
                if dist_ji < min_dist:
                    min_dist = dist_ji
                    min_index = j
            if cluster_assments[i, 0] != min_index:
                cluster_changed = True
            cluster_assments[i, :] = min_index, min_dist
        # 重新计算中心点坐标
        for center in range(k):
            samples = dataset[np.nonzero(cluster_assments[:, 0] == center)[0]]
            centroids[center, :] = np.mean(samples, axis=0)
    return centroids, cluster_assments


def pca(XMat, k):
    """
    參数：
        - XMat：传入的是一个numpy的矩阵格式，行表示样本数，列表示特征
        - k：表示取前k个特征值相应的特征向量
    返回值：
        - finalData：參数一指的是返回的低维矩阵，相应于输入參数二
        - reconData：參数二相应的是移动坐标轴后的矩阵
    """
    m, n = np.shape(XMat)
    if k > n:
        print("k must lower than feature number")
        return

    means = np.mean(XMat, axis=0)       # 计算均值
    data_adjust = XMat - means          # 减去均值
    cov_x = np.cov(data_adjust.T)       # 计算协方差矩阵
    feat_value, feat_vec = np.linalg.eig(cov_x)  # 求解协方差矩阵的特征值和特征向量
    index = np.argsort(-feat_value)     # 依照featValue进行从大到小排序

    #注意特征向量是列向量。而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
    selectVec = np.mat(feat_vec.T[index[:k]]) #所以这里须要进行转置
    finalData = data_adjust * selectVec.T
    reconData = (finalData * selectVec) + means
    return np.array(finalData), reconData


if __name__ == '__main__':
    np.random.seed(13)

    X, y = load_dataset('iris.csv')
    X, _ = pca(X, 2)
    centroids, cluster_assments = k_means(X, 3)
    label_pred = cluster_assments[:, 0]
    # 绘制k-means结果
    x0 = X[label_pred == 0]
    x1 = X[label_pred == 1]
    x2 = X[label_pred == 2]
    plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
    plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
    plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc=2)
    plt.show()
