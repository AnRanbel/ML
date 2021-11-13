import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


# 载入数据集
def load_dataset(filename):
    df = pd.read_csv(filename, sep='\s+', header=None)
    x_mat = np.array(df.iloc[:, 0: -1])
    y_mat = np.array(df.iloc[:, -1])
    return x_mat, y_mat


class TreeTrunk(object):
    def __init__(self):
        self.tree = dict()

    def _get_pred(self, data_set, value, inequal_type):
        """
        二分类，'lt':小于阈值为-1，大于阈值为1； 'gt'相反
        :param data_set: 输入值特征值
        :param value: 阈值
        :param inequal_type: 判断形式
        :return:
        """
        pred = np.ones(len(data_set))
        if inequal_type == 'lt':
            pred[data_set <= value] = -1.
        else:
            pred[data_set > value] = -1.
        return pred

    def fit(self, data_set, label, D):
        """
        遍历所有的特征，之后遍历该特征的所有值，选出信息在有权值的情况下分类错误最小的特征和值
        :param data_set: 输入特征数据
        :param label: 标签
        :param D: 样本权值
        :return:
        """
        n = data_set.shape[1]
        min_err = np.inf    # A floating point representation of positive infinity.（正无穷大的浮点数代表）
        best_pred = None
        num_step = 10   # 走10步（走完）
        # 遍历所有的特征(找到最佳阈值，及对应的最小分类误差，比较方式——基分类器)
        for i in range(n):
            # 统计第i列属性的数值范围
            min_value = data_set[:, i].min()
            max_value = data_set[:, i].max()
            # 计算测试步长
            step = (max_value - min_value) / num_step
            for j in range(-1, int(num_step) + 1):
                value = (min_value + j * step)      # 阈值
                # 对大于阈值为-1还是小于阈值为-1进行遍历判断
                for inequal in ['lt', 'gt']:
                    pred = self._get_pred(data_set[:, i], value, inequal)
                    wrong_pred = np.ones(len(data_set))
                    wrong_pred[label == pred] = 0
                    # 求取有权值的时候的错误率
                    weight_err = np.sum(D * wrong_pred)     # em
                    if weight_err < min_err:      # 不同的权值，最终会得到不同的阈值、特征维度、比较方式
                        min_err = weight_err    # 最小em（分类误差率）
                        best_pred = pred    # 最佳预测结果
                        self.tree['dim'] = i
                        self.tree['thresh'] = value
                        self.tree['ineq'] = inequal
        return self.tree, min_err, best_pred


class AdaBoost(object):

    def __init__(self):
        self.trees = []
        self.D = None
        self.eps = np.finfo(dtype=float).eps    # Machine limits for floating point types.

    def train_stump(self, data_set, label):
        """
        获取基分类器以及基分类对训练样本的错误率和预测值
        :param data_set: 输入样本
        :param label: 标签
        :return:
        """
        tree = TreeTrunk()
        return tree.fit(data_set, label, self.D)

    def fit(self, data_set, label, max_iter=40, tol=0.001):
        """
        训练模型的入口
        :param data_set: 输入样本数据
        :param label: 标签
        :param tol: 容许的最大误差，误差小于等于这个值就可以停止
        :return:
        """
        m = data_set.shape[0]   # 样本数

        # 初始化样本权值
        self.D = np.array([1/m] * m)
        # 初始化训练结果
        train_result = np.array([0] * m)

        for i in range(max_iter):
            # 基分类器
            tree, err, pred = self.train_stump(data_set, label)

            # 基分类器的系数：alpha = 1 / 2 * log ((1 - e_m) / e_m)      # 底数默认为e
            alpha = 0.5 * np.log((1 - err) / (err + self.eps))
            tree["alpha"] = alpha
            self.trees.append(tree)     # 每一次迭代添加一个分类器

            # 更新权值
            self.D = self.D * np.exp(-1 * alpha * label * pred)
            self.D = self.D / np.sum(self.D)

            # 下面的逻辑都是为了计算当前训练的误差，是不是达到阈值可以停止训练
            train_result = alpha * pred + train_result

            result = np.sign(train_result)     # 符号函数,x>0,sign(x)=1;x=0,sign(x)=0;x<0,sign(x)=-1
            result_err = np.sum(result != label) / len(label)
            print('iter: ', i, 'err: ', result_err)
            if result_err <= tol:     # tol,容许的最大误差
                break

    def _predict_single(self, df_data):
        """
        预测单个样本的结果
        :param df_data:
        :return:
        """
        pre_result = 0
        for tree in self.trees:     # 拿出每棵树的权重系数、特征维度、阈值、比较方式
            alpha = tree['alpha']
            dim = tree['dim']
            thresh = tree['thresh']
            ineq = tree['ineq']
            feature_value = df_data[dim]
            if ineq == 'lt':
                if feature_value <= thresh:
                    result = -1
                else:
                    result = 1
            else:
                if feature_value <= thresh:
                    result = 1
                else:
                    result = -1
            pre_result += alpha * result
        return np.sign(pre_result)


    def predict(self, data_set):
        """
        预测结果
        :param data_set: 输入值
        :return: 预测结果
        """
        result_list = []
        for i in range(len(data_set)):
            ret_df_data = data_set[i]
            result = self._predict_single(ret_df_data)
            result_list.append(result)

        return result_list

def draw_figure(data_set, label, weakClassArr, num_step=10):  # 画图
    matplotlib.rcParams['axes.unicode_minus'] = False  # 防止坐标轴的‘-’变为方块
    matplotlib.rcParams["font.sans-serif"] = ["simhei"]  # 显示中文的方法
    fig = plt.figure()  # 创建画布
    ax = fig.add_subplot(111)  # 添加子图

    red_points_x = []  # 红点的x坐标
    red_points_y = []  # 红点的y坐标
    blue_points_x = []  # 蓝点的x坐标
    blue_points_y = []  # 蓝点的y坐标
    m, n = np.shape(data_set)  # 训练集的维度是 m×n ，m就是样本个数，n就是每个样本的特征数

    for i in range(m):  # 遍历训练集，把红点，蓝点分开存入
        if label[i] == 1:
            red_points_x.append(data_set[i][0])  # 红点x坐标
            red_points_y.append(data_set[i][1])
        else:
            blue_points_x.append(data_set[i][0])
            blue_points_y.append(data_set[i][1])

    line_thresh = 0.00  # 画线阈值，就是不要把线画在点上，而是把线稍微偏移一下，目的就是为了让图更加美观直接
    annotagte_thresh = 0.00  # 箭头间隔，也是为了美观
    x_min = min(data_set[:, 0])
    y_min = min(data_set[:, 1])
    x_max = max(data_set[:, 0])
    y_max = max(data_set[:, 1])
    step_x = (x_max - x_min) / num_step
    x_min -= step_x
    step_y = (y_max - y_min) / num_step
    y_min -= step_y


    v_line_list = []  # 把竖线阈值的信息存起来，包括阈值大小，分类方式，alpha大小都存起来
    h_line_list = []  # 横线阈值也是如此，因为填充每个区域时，竖阈值和横阈值是填充边界，是不一样的，需各自分开存贮
    for baseClassifier in weakClassArr:  # 画阈值
        if baseClassifier['dim'] == 0:  # 画竖线阈值
            if baseClassifier['ineq'] == 'lt':  # 根据分类方式,lt时
                ax.vlines(baseClassifier['thresh'] + line_thresh, y_min, y_max, colors='green', label='阈值')  # 画直线
                ax.arrow(baseClassifier['thresh'] + line_thresh, y_max, 0.08, 0, head_width=0.05, head_length=0.02)  # 显示箭头
                ax.text(baseClassifier['thresh'] + annotagte_thresh, y_max + line_thresh,
                        str(round(baseClassifier['alpha'], 2)))  # 画alpha值
                v_line_list.append(
                    [baseClassifier['thresh'], 1, baseClassifier['alpha']])  # 把竖线信息存入，注意分类方式，lt就存1,gt就存-1

            else:  # gt时，分类方式不同，箭头指向也不同
                ax.vlines(baseClassifier['thresh'] + line_thresh, y_min, y_max, colors='green', label="阈值")
                ax.arrow(baseClassifier['thresh'] + line_thresh, y_min, -0.08, 0, head_width=0.05, head_length=0.02)
                ax.text(baseClassifier['thresh'] + annotagte_thresh, y_min + line_thresh,
                        str(round(baseClassifier['alpha'], 2)))
                v_line_list.append([baseClassifier['thresh'], -1, baseClassifier['alpha']])
        else:  # 画横线阈值
            if baseClassifier['ineq'] == 'lt':  # 根据分类方式，lt时
                ax.hlines(baseClassifier['thresh'] + line_thresh, x_min, x_max, colors='black', label="阈值")
                ax.arrow(x_max, baseClassifier['thresh'] + line_thresh, 0., 0.08, head_width=0.05,
                         head_length=0.05)
                ax.text(x_max + annotagte_thresh, baseClassifier['thresh'], str(round(baseClassifier['alpha'], 2)))
                h_line_list.append([baseClassifier['thresh'], 1, baseClassifier['alpha']])
            else:  # gt时
                ax.hlines(baseClassifier['thresh'] + line_thresh, x_min, x_max, colors='black', label="阈值")
                ax.arrow(x_min + line_thresh, baseClassifier['thresh'], 0., 0.08, head_width=-0.05, head_length=0.05)
                ax.text(x_min + annotagte_thresh, baseClassifier['thresh'], str(round(baseClassifier['alpha'], 2)))
                h_line_list.append([baseClassifier['thresh'], -1, baseClassifier['alpha']])
    v_line_list.sort(key=lambda x: x[0])  # 我们把存好的竖线信息按照阈值大小从小到大排序，因为我们填充颜色是从左上角开始，所以竖线从小到大排
    h_line_list.sort(key=lambda x: x[0], reverse=True)  # 横线从大到小排序
    v_line_list_size = len(v_line_list)  # 排好之后，得到竖线有多少条
    h_line_list_size = len(h_line_list)  # 得到横线有多少条
    alpha_value = [x[2] for x in v_line_list] + [y[2] for y in h_line_list]  # 把属性横线的所有alpha值取出来，这里也证实了上面的排序不是无用功
    #print('alpha_value', alpha_value)

    for i in range(h_line_list_size + 1):  # 开始填充颜色，(横线的条数+1) × (竖线的条数+1) = 分割的区域数,然后开始往这几个区域填颜色
        for j in range(v_line_list_size + 1):  # 我们是左上角开始填充直到右下角，所以采用这种遍历方式
            list_test = list(np.multiply([1] * j + [-1] * (v_line_list_size - j), [x[1] for x in v_line_list])) + list(
                np.multiply([-1] * i + [1] * (h_line_list_size - i), [x[1] for x in h_line_list]))
            # 上面是一个规律公式，后面会用文字解释它
            # print('list_test',list_test)
            temp_value = np.multiply(alpha_value,
                                  list_test)  # list_test其实就是加减号，我们知道了所有alpha值，可是每个alpha是相加还是相加，这就是list_test作用了
            reslut_test = np.sign(sum(temp_value))  # 计算完后，sign一下，然后根据结果进行分类
            if reslut_test == 1:  # 如果是1,就是正类红点
                color_select = 'orange'  # 填充的颜色是橘红色
                hatch_select = '.'  # 填充图案是。
                # print("是正类，红点")
            else:  # 如果是-1,那么是负类蓝点
                color_select = 'green'  # 填充的颜色是绿色
                hatch_select = '*'  # 填充图案是*
                # print("是负类，蓝点")
            if i == 0:  # 上边界     现在开始填充了，用fill_between函数，我们需要得到填充的x坐标范围，和y的坐标范围，x范围就是多条竖线阈值夹着的区域，y范围是横线阈值夹着的范围
                if j == 0:  # 左上角
                    ax.fill_between(x=[x for x in np.arange(x_min, v_line_list[j][0] + line_thresh, 0.001)], y1=y_max,
                                    y2=h_line_list[i][0] + line_thresh, color=color_select, alpha=0.3,
                                    hatch=hatch_select)
                elif j == v_line_list_size:  # 右上角
                    ax.fill_between(x=[x for x in np.arange(v_line_list[-1][0] + line_thresh, x_max, 0.001)], y1=y_max,
                                    y2=h_line_list[i][0] + line_thresh, color=color_select, alpha=0.3,
                                    hatch=hatch_select)
                else:  # 中间部分
                    ax.fill_between(x=[x for x in
                                       np.arange(v_line_list[j - 1][0] + line_thresh, v_line_list[j][0] + line_thresh,
                                              0.001)], y1=y_max, y2=h_line_list[i][0] + line_thresh, color=color_select,
                                    alpha=0.3, hatch=hatch_select)
            elif i == h_line_list_size:  # 下边界
                if j == 0:  # 左下角
                    ax.fill_between(x=[x for x in np.arange(x_min, v_line_list[j][0] + line_thresh, 0.001)],
                                    y1=h_line_list[-1][0] + line_thresh, y2=y_min, color=color_select, alpha=0.3,
                                    hatch=hatch_select)
                elif j == v_line_list_size:  # 右下角
                    ax.fill_between(x=[x for x in np.arange(v_line_list[-1][0] + line_thresh, x_max, 0.001)],
                                    y1=h_line_list[-1][0] + line_thresh, y2=y_min, color=color_select, alpha=0.3,
                                    hatch=hatch_select)
                else:  # 中间部分
                    ax.fill_between(x=[x for x in
                                       np.arange(v_line_list[j - 1][0] + line_thresh, v_line_list[j][0] + line_thresh,
                                              0.001)], y1=h_line_list[-1][0] + line_thresh, y2=y_min,
                                    color=color_select, alpha=0.3, hatch=hatch_select)
            else:
                if j == 0:  # 中左角
                    ax.fill_between(x=[x for x in np.arange(x_min, v_line_list[j][0] + line_thresh, 0.001)],
                                    y1=h_line_list[i - 1][0] + line_thresh, y2=h_line_list[i][0] + line_thresh,
                                    color=color_select, alpha=0.3, hatch=hatch_select)
                elif j == v_line_list_size:  # 中右角
                    ax.fill_between(x=[x for x in np.arange(v_line_list[-1][0] + line_thresh, x_max, 0.001)],
                                    y1=h_line_list[i - 1][0] + line_thresh, y2=h_line_list[i][0] + line_thresh,
                                    color=color_select, alpha=0.3, hatch=hatch_select)
                else:  # 中间部分
                    ax.fill_between(x=[x for x in
                                       np.arange(v_line_list[j - 1][0] + line_thresh, v_line_list[j][0] + line_thresh,
                                              0.001)], y1=h_line_list[i - 1][0] + line_thresh,
                                    y2=h_line_list[i][0] + line_thresh, color=color_select, alpha=0.3,
                                    hatch=hatch_select)

    ax.scatter(red_points_x, red_points_y, s=30, c='red', marker='s', label="red points")  # 画红点
    ax.scatter(blue_points_x, blue_points_y, s=40, label="blue points")  # 画蓝点
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    #ax.legend()  # 显示图例    如果你想用legend设置中文字体，参数设置为 prop=myfont
    ax.set_title("AdaBoost分类")  # 设置标题
    plt.show()


if __name__ == '__main__':  # 运行函数
    data_set, label = load_dataset('testSetRBF.txt')
    ada_boost = AdaBoost()
    ada_boost.fit(data_set, label, 50)

    pred = ada_boost.predict(data_set)
    accuracy = np.sum(pred == label)
    print('Test Accuracy: %f%%' % (accuracy * 100 / len(label)))

    draw_figure(data_set, label, ada_boost.trees)  # 画图

