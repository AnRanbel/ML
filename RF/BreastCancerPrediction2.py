import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from warnings import filterwarnings
filterwarnings("ignore")

# import os
# # os.walk 产生包含3个元素的元组：dirpath, dirnames, filenames
# for dirname, _, filenames in os.walk('../KNN'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
#
# cancer = pd.read_csv('..\KNN\Breast Cancer_data.csv')    # 类型DataFrame
# cancer = cancer.drop('id', axis=1)      # 删除“id”列（DataFrame会默认有行索引和列索引)
# print(cancer.sample(5))   # 随机抽取五行数据

# 在此导入time库，并在开头设置开始时间
import time
start = time.perf_counter()


# Data Preprocessing——Dealing with outliers
cancer = pd.read_csv('..\KNN\Breast Cancer_data.csv')    # 类型DataFrame
cancer = cancer.drop('id', axis=1)      # 删除“id”列（DataFrame会默认有行索引和列索引)
def outlier(df):
    df_ = df.copy()
    df = df.drop('diagnosis', axis=1)

    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - (1.5 * iqr)
    upper_limit = q3 + (1.5 * iqr)

    for col in df.columns:
        for i in range(0, len(df[col])):
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
    vif['Predictor'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values, col) for col in range(len(df.columns))]
    return vif

vif_df = VIF(X).sort_values('VIF', ascending = False, ignore_index = True)
print(vif_df.head(8))


# Removing features with VIF > 10,000
high_vif_features = list(vif_df.Predictor.iloc[:2])
vif_features = X.drop(high_vif_features, axis=1)

# Splitting data for training and testing
X_train, X_test, y_train, y_test = train_test_split(vif_features, y, test_size = 0.2, random_state = 39)


# Random Forest Classifier with VIF features and hyperparameter tuning

steps = [('scaler', StandardScaler()),
         ('rf', RandomForestClassifier(random_state = 0))]
pipeline = Pipeline(steps)

parameters = dict(rf__n_estimators = [10,100],      # 森林中树木的数量
                  rf__max_features = ['sqrt', 'log2'],
)


cv = GridSearchCV(pipeline,
                  param_grid = parameters,
                  cv = 5,
                  scoring = 'accuracy',
                  n_jobs = -1)

cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
rf_accuracy = accuracy_score(y_test,y_pred) * 100

print('\033[1m' +'Best parameters : '+ '\033[0m', cv.best_params_)
print('\033[1m' +'Accuracy : {:.2f}%'.format(rf_accuracy) + '\033[0m')
print('\033[1m' +'Classification report : '+ '\033[0m\n', classification_report(y_test, y_pred))

cm = confusion_matrix(y_test,y_pred)
print('\033[1m' +'Confusion Matrix : '+ '\033[0m')
sns.heatmap(cm, cmap = 'OrRd',annot = True, fmt='d')
plt.show()


# 在程序运行结束的位置添加结束时间
end = time.perf_counter()
print("运行耗时", end-start)