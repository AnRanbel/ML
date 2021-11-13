import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 载入数据集
def load_dataset(filename):
    df = pd.read_csv(filename)
    x_mat = np.array(df.iloc[:, 0: -1])
    y_mat = np.array(df.iloc[:, -1])
    return x_mat, y_mat

# 朴素贝叶斯分类器(正态分布型)
class GaussianNB(object):

    def __init__(self):
        self._classes = np.array(0)         # 类别数组(存储每个类别的编号)
        self._mu = np.zeros((0, 0))         # 均值矩阵
        self._var = np.zeros((0, 0))        # 方差矩阵
        self._p_class = np.zeros(0)         # 每个类别在总样本中出现的频率

    def fit(self, X, y):
        """模型训练"""
        pass


    def predict(self, X):
        """类别预测"""
        pass

def show_dataset(X, y):
    column_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

    fig = plt.figure('Iris Data', figsize=(15, 15))
    plt.suptitle("Andreson's Iris Dara Set\n(Blue->Setosa|Red->Versicolor|Green->Virginical)")

    for i in range(4):
        for j in range(4):
            plt.subplot(4, 4, 4 * i + (j + 1))
            if i == j:
                plt.text(0.3, 0.4, column_names[i], fontsize=15)
            else:
                plt.scatter(X[:, j], X[:, i], c=y, cmap='brg')

            if i == 0:
                plt.title(column_names[j])
            if j == 0:
                plt.ylabel(column_names[i])

    plt.show()

if __name__ == '__main__':
    X, y = load_dataset('iris_training.csv')

    nb = GaussianNB()
    nb.fit(X, y)

    input, label = load_dataset('iris_test.csv')
    pred = nb.predict(input)

    accuracy = (pred == label).sum() / label.sum()

    print('测试精度： %.2f%%' % (accuracy*100.0))

    show_dataset(np.vstack((X, input)), np.concatenate((y, label)))
