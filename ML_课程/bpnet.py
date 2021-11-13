import numpy as np
import math
import mniset

class Sigmoid:
    @staticmethod
    def forward(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def backward(y):
        return y * (1 - y)


class Linear:
    def __init__(self, input_size, output_size, activator):
        # self.input_size：输入维度
        # self.output_size：输出维度
        # self.activator：激活函数
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权值矩阵self.w和偏置self.b
        self.w = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        self.b = np.zeros((output_size, 1))

        self.w_grad_total = np.zeros((output_size, input_size))
        self.b_grad_total = np.zeros((output_size, 1))

    def forward(self, input_data):
        self.input_data = input_data
        self.output_data = self.activator.forward(
            np.dot(self.w, self.input_data) + self.b)

    def backward(self, input_delta):
        # input_delta_为后一层传入的误差
        # output_delta为传入前一层的误差
        self.sen = input_delta * self.activator.backward(self.output_data)
        output_delta = np.dot(self.w.T, self.sen)
        self.w_grad = np.dot(self.sen, self.input_data.T)
        self.b_grad = self.sen
        self.w_grad_total += self.w_grad
        self.b_grad_total += self.b_grad
        return output_delta

    def update(self, lr, MBGD_mode=0):
        # 梯度下降法进行权值更新,有几种更新权值的算法。
        # MBGD_mod==0指SGD模式，即随机梯度下降
        # MBGD_mod==1指mnni_batch模式，即批量梯度下降, 当选取batch为整个训练集时，为BGD模式，即批量梯度下降
        if MBGD_mode == 0:
            self.w -= lr * self.w_grad
            self.b -= lr * self.b_grad
        elif MBGD_mode == 1:
            self.w -= lr * self.w_grad_total
            self.b -= lr * self.b_grad_total
            self.w_grad_total = np.zeros((self.output_size, self.input_size))
            self.b_grad_total = np.zeros((self.output_size, 1))


class BPNet:
    def __init__(self, params, activator):
        # params_array为层维度信息超参数数组
        # layers为网络的层集合
        self.layers = []
        for i in range(len(params) - 1):
            self.layers.append(Linear(params[i], params[i + 1], activator))

    # 网络前向迭代
    def predict(self, sample):
        # 下面一行的output可以理解为输入层输出
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output_data
        return output

    # 网络反向迭代
    def calc_gradient(self, label):
        delta = (self.layers[-1].output_data - label)
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta

    # 一次训练一个样本 ，然后更新权值
    def train_one_sample(self, sample, label, lr):
        self.predict(sample)
        Loss = self.loss(self.layers[-1].output_data, label)
        self.calc_gradient(label)
        self.update(lr)
        return Loss

    # 一次训练一批样本 ，然后更新权值
    def train_batch_sample(self, sample_set, label_set, lr, batch):
        loss = 0.0
        for i in range(batch):
            self.predict(sample_set[i])
            loss += self.loss(self.layers[-1].output_data, label_set[i])
            self.calc_gradient(label_set[i])
        self.update(lr, 1)
        return loss

    def update(self, lr, MBGD_mode=0):
        for layer in self.layers:
            layer.update(lr, MBGD_mode)

    def loss(self, pred, label):
        return 0.5 * ((pred - label) * (pred - label)).sum()

    def gradient_check(self, sample, label):
        self.predict(sample)
        self.calc_gradient(label)
        incre = 10e-4
        for layer in self.layers:
            for i in range(layer.w.shape[0]):
                for j in range(layer.w.shape[1]):
                    layer.w[i][j] += incre
                    pred = self.predict(sample)
                    err1 = self.loss(pred, label)
                    layer.w[i][j] -= 2 * incre
                    pred = self.predict(sample)
                    err2 = self.loss(pred, label)
                    layer.w[i][j] += incre
                    pred_grad = (err1 - err2) / (2 * incre)
                    calc_grad = layer.w_grad[i][j]
                    print('weights(%d,%d): expected - actural %.4e - %.4e' %
                          (i, j, pred_grad, calc_grad))


if __name__ == '__main__':
    params = [28*28, 500, 10]
    net = BPNet(params, Sigmoid)

    train_images = mniset.load_train_images('MNISET')
    train_images = train_images.reshape((train_images.shape[0], -1))

    train_labels = mniset.load_train_labels('MNISET')
    one_hot = np.eye(10)
    train_labels = one_hot[train_labels.astype(np.int32)]

    lr = 0.002
    max_epoch = 10
    batch = 100
    for i in range(max_epoch):
        print('epoch: %d' % i)

        n = train_images.shape[0]
        epoch_loss = 0.0

        '''
        # 遍历训练集迭代
        for j in range(n):
            # 输入数据要转换成列矩阵形式(A[n, 1])
            data = np.expand_dims(train_images[j], axis=1)
            label = np.expand_dims(train_labels[j], axis=1)
            loss = net.train_one_sample(data, label, lr)
            epoch_loss += loss
            if j % 1000 == 0 and j != 0:
                print('batch: %d, loss: %f' % (j, epoch_loss / (j+1)))
        '''

        # 随机批次迭代
        indices = [idx for idx in range(n)]
        np.random.shuffle(indices)
        nbatch = math.ceil(n / batch)
        for j in range(nbatch):
            begin = j * batch
            end = begin + batch if begin + batch <= n else n
            batch_sample = np.expand_dims(train_images[indices[begin: end]], axis=2)
            batch_label = np.expand_dims(train_labels[indices[begin: end]], axis=2)
            loss = net.train_batch_sample(batch_sample, batch_label, lr, end-begin)
            epoch_loss += loss
            if (j % 10) == 0:
                print('batch: %d, loss: %f' % (j, loss / batch))

        print('epoch average loss: %f' % (epoch_loss / n))

    # 测试网络
    test_images = mniset.load_test_images('MNISET')
    test_images = test_images.reshape((test_images.shape[0], -1))
    test_labels = mniset.load_test_labels('MNISET')

    accuracy = 0
    m = test_images.shape[0]
    for i in range(m):
        data = np.expand_dims(test_images[i], axis=1)
        output = net.predict(data)
        if np.argmax(output) == test_labels[i]:
            accuracy += 1

    print('Test Accuracy: %f%%' % (accuracy*100 / m))


    '''    
    print('input: ')
    print(data)
    print('predict: ')
    print(net.predict(data))
    print('true: ')
    print(label)
    net.gradient_check(data, label)
    '''
