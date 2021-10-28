import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.ensemble import BaggingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from warnings import filterwarnings
filterwarnings("ignore")

# 在此导入time库，并在开头设置开始时间
import time
start = time.perf_counter()


# Data Preprocessing——Dealing with outliers（处理离群值）
cancer = pd.read_csv('..\KNN\Breast Cancer_data.csv')    # 类型DataFrame
cancer = cancer.drop('id', axis=1)      # 删除“id”列（DataFrame会默认有行索引和列索引)
def outlier(df):
    df_ = df.copy()     # 复制索引和数据（默认深复制）
    df = df.drop('diagnosis', axis=1)

    q1 = df.quantile(0.25)      # 各样本特征值从小到大排序，处于25%的数
    q3 = df.quantile(0.75)      # 各样本特征值从小到大排序，处于75%的数

    iqr = q3 - q1

    lower_limit = q1 - (1.5 * iqr)
    upper_limit = q3 + (1.5 * iqr)

    # 使用插值法（对离群值赋予一个相对合理的新值），使所有值都处于lower_limit、upper_limit之间
    for col in df.columns:      # df.columns=特征标签
        for i in range(0, len(df[col])):    # len(df[col])=样本数量
            if df[col][i] < lower_limit[col]:
                df[col][i] = lower_limit[col]

            if df[col][i] > upper_limit[col]:
                df[col][i] = upper_limit[col]

    for col in df.columns:
        df_[col] = df[col]

    return (df_)

cancer = outlier(cancer)


# Separating features and target
X = cancer.drop('diagnosis', axis=1)
y = cancer.diagnosis


# 方差膨胀因子(Variance Inflation Factor,VIF)：in our correlation matrix, many of our predictor variables were higly correlated.
# To avoid multicollinearity, we must deal with such columns.
def VIF(df):
    vif = pd.DataFrame()
    vif['Predictor'] = df.columns       # 插入了一列
    vif['VIF'] = [variance_inflation_factor(df.values, col) for col in range(len(df.columns))]      # 每个特征返回一个方差膨胀因子
    return vif

vif_df = VIF(X).sort_values('VIF', ascending = False, ignore_index = True)      # 降序排列
print(vif_df.head(8))       # 返回前八行数据


# Removing features with VIF > 10,000
high_vif_features = list(vif_df.Predictor.iloc[:2])     # 取前两个特征
# print(high_vif_features)
vif_features = X.drop(high_vif_features, axis=1)

# Splitting data for training and testing
X_train, X_test, y_train, y_test = train_test_split(vif_features, y, test_size = 0.2, random_state = 39)    # 生成随机训练和测试子集，测试集占0.2

# KNN with VIF features and hyperparameter tuning（超参数：需要人工选择的参数）

# 先对数据预处理然后计算模型
steps = [('scaler', StandardScaler()),      # 特征缩放，归一化处理
         ('knn', BaggingClassifier(KNeighborsClassifier()))]    # Bagging分类器是一个集合元估计器，它使每个基本分类器拟合原始数据集的随机子集，然后将其单个预测（通过投票或平均）进行汇总以形成最终预测
pipeline = Pipeline(steps)

parameters = dict(knn__base_estimator__metric = ['euclidean', 'manhattan', 'minkowski'],    # 欧氏距离、曼哈顿距离、闵氏距离
                  knn__base_estimator__weights =  ['uniform', 'distance'],    # 预测中使用的权重函数。 可能的值：“uniform”：统一权重。 每个邻域中的所有点均被加权。“distance”：权重点与其距离的倒数。 在这种情况下，查询点的近邻比远处的近邻具有更大的影响力。
                  knn__base_estimator__n_neighbors = range(2,15),      # K的取值范围
                  knn__bootstrap = [True, False],   # 样本是否放回——针对BaggingClassifier
                  knn__bootstrap_features = [True, False],  # 样本特征是否放回——针对BaggingClassifier
                  knn__n_estimators = [5])      # 基模型器个数

# 参数估计，得到最优的K值（网格搜索&交叉验证）
cv = GridSearchCV(pipeline,
                  param_grid = parameters,  # 参数搜索范围
                  cv = 5,       # 5折交叉验证
                  scoring = 'accuracy',     # 评分方法-准确率
                  n_jobs = -1,     # 使用所有CPU处理器
                  )

cv.fit(X_train, y_train)      # 在训练集上训练
y_pred = cv.predict(X_test)     # 使用找到的最优参数在估计器上调用预测
knn_accuracy = accuracy_score(y_test,y_pred) * 100

# \033[3开头的是字体颜色;[1m 比 [0m 更亮更粗;[4开头的是背景色
print('\033[1m' +'Best parameters : '+ '\033[0m', cv.best_params_)
print('\033[1m' +'Accuracy : {:.2f}%'.format(knn_accuracy) + '\033[0m')
print('\033[1m' +'Classification report : '+ '\033[0m\n', classification_report(y_test, y_pred))

cm = confusion_matrix(y_test,y_pred)       # 混淆矩阵
print('\033[1m' +'Confusion Matrix : '+ '\033[0m')
sns.heatmap(cm, cmap = 'OrRd',annot = True, fmt='d')    # 根据混淆矩阵生成可视图  cmap-颜色  annot=true-显示数字
plt.show()

# 在程序运行结束的位置添加结束时间
end = time.perf_counter()
print("运行耗时", end-start)