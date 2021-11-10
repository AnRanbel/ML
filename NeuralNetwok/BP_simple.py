#! /usr/bin/python
# -*- encoding:utf8 -*-

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)      # 矩阵点乘，相同位置的对应元素相乘

class BP:
    def __init__(self, layer, iter, max_error):
        self.input_n = layer[0]  # 输入层的节点个数 d
        self.hidden_n = layer[1]  # 隐藏层的节点个数 q
        self.output_n = layer[2]  # 输出层的节点个数 l
        self.gj = []
        self.input_weights = []   # 输入层与隐藏层的权值矩阵
        self.output_weights = []  # 隐藏层与输出层的权值矩阵
        self.iter = iter          # 最大迭代次数
        self.error=[]       # 误差矩阵
        self.max_error = max_error  # 停止的误差范围

        # 阀值即偏置
        # 初始化一个(d+1) * q的矩阵，多加的1是将隐藏层的阀值加入到矩阵运算中（初始化权重和偏置）
        # np.random.random:Return random floats in the half-open interval [0.0, 1.0)
        self.input_weights = np.random.random((self.input_n + 1, self.hidden_n))        # 行数即上一层的结点数+1，列数本层的结点数
        # 初始话一个(q+1) * l的矩阵，多加的1是将输出层的阀值加入到矩阵中简化计算
        self.output_weights = np.random.random((self.hidden_n + 1, self.output_n))

        self.gj = np.zeros(layer[2])

    #  正向传播与反向传播
    def forword_backword(self, X, Y, learning_rate=0.1):
        bh,yj=[],[]
        X_mean=np.mean(np.array(X),axis=0)
        for xj,y in zip(X,Y):
            xj = np.array(xj)
            y = np.array(y)
            input = np.ones((1, xj.shape[0] + 1))       # 将偏置的系数1加入input
            input[:, :-1] = xj    # 前几列放输入x，最后一列是偏置的系数1
            x = input
            ah = np.dot(x,self.input_weights)
            bh = sigmoid(ah)    # 隐藏层1

            input = np.ones((1, self.hidden_n + 1))
            input[:, :-1] = bh
            bh = input
            bj = np.dot(bh, self.output_weights)
            yj.append(sigmoid(bj))    # 输出层

        yj = np.array(yj)
        Y = np.array(Y)
        error=0.5 * np.mean(np.square(yj-Y))
        self.error.append(error)

        # 反向传播
        self.gj = (yj-Y) * sigmoid_derivative(yj)
        self.gj=np.mean(self.gj)

        #  更新输入层权值w
        for j in range(self.input_weights.shape[1]):
            for i in range(self.input_weights.shape[0] - 1):
                self.input_weights[i][j] -= learning_rate * self.gj * sigmoid_derivative(bh[0][j]) *self.output_weights[j][0]*X_mean[i]

        # 更新输入层阀值b
        for j in range(self.input_weights.shape[1]):
            self.input_weights[-1][j] -= learning_rate * self.gj*sigmoid_derivative(bh[0][j]) *self.output_weights[j][0]

        #  更新输出层权值w，因为权值矩阵的最后一行表示的是阀值多以循环只到倒数第二行
        for j in range(self.output_weights.shape[1]):
            for i in range(self.output_weights.shape[0] - 1):
                self.output_weights[i][j] -= learning_rate * self.gj * bh[0][i]

        #  更新输出层阀值b，权值矩阵的最后一行表示的是阀值
        for j in range(self.output_weights.shape[1]):
            self.output_weights[-1][j] -= learning_rate * self.gj

        return error

    def fit(self, X, y):
        for i in range(self.iter):
            error=self.forword_backword(X, y)
            if abs(error) <= self.max_error:
                break
        plt.figure(figsize=(15, 12))
        plt.plot(self.error)
        plt.show()

    def predict(self, x_test):
        x_test = np.array(x_test)
        tmp = np.ones((x_test.shape[0], self.input_n + 1))
        tmp[:, :-1] = x_test
        x_test = tmp
        an = np.dot(x_test, self.input_weights)
        bh = sigmoid(an)
        #  多加的1用来与阀值相乘
        tmp = np.ones((bh.shape[0], bh.shape[1] + 1))
        tmp[:, : -1] = bh
        bh = tmp
        bj = np.dot(bh, self.output_weights)
        yj = sigmoid(bj)
        return yj

if __name__ == '__main__':
    #  指定神经网络输入层，隐藏层，输出层的元素个数
    layer = [2, 4, 1]
    X = [
            [1, 1],
            [2, 2],
            [1, 2],
            [1, -1],
            [2, 0],
            [2, -1]
        ]
    y = [[0], [0], [0], [1], [1], [1]]

    # x_test = [[2, 3],
    #           [2, 2]]

    #  设置最大的迭代次数，以及最大误差值
    bp = BP(layer, 1000, 0.0001)
    bp.fit(X, y)
    print(bp.predict(X))
