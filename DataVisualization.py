import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
from plotly.subplots import make_subplots


from warnings import filterwarnings
filterwarnings("ignore")

cancer = pd.read_csv('.\KNN\Breast Cancer_data.csv')
cancer = cancer.drop('id', axis=1)

# Dropping Unnamed column
cancer = cancer.loc[:, ~cancer.columns.str.contains('^Unnamed')]

# Encoding target variable
"""
pandas categorical data type;A categorical variable takes on a limited, and usually fixed.
All values of categorical data are either in categories or np.nan. Order is defined by the order of categories, 
not lexical order of the values. Internally, the data structure consists of a categories array and an integer 
array of codes which point to the real value in the categories array.

As a signal to other Python libraries that this column should be treated as a categorical variable 
(e.g. to use suitable statistical methods or plot types).

from pandas.api.types import CategoricalDtype
s = pd.Series(["a", "b", "c", "a"])
cat_type = CategoricalDtype(categories=["b", "c", "d"], ordered=True)
s_cat = s.astype(cat_type)
s_cat
Out[30]: 
0    NaN
1      b
2      c
3    NaN
dtype: category
Categories (3, object): ['b' < 'c' < 'd']
"""
cancer.diagnosis = cancer.diagnosis.astype('category')      # 会多出一些结构，比如cat（把M,B变成了1,0）
cancer.diagnosis = cancer.diagnosis.cat.codes   # Accessor object for categorical properties of the Series values.
print(cancer.diagnosis.value_counts())     # Return a Series containing counts of unique values.

cancer_mean = cancer.loc[:, 'radius_mean':'fractal_dimension_mean']     # 只留下radius_mean列到fractal_dimension_mean列的数据（全是平均数）
cancer_mean['diagnosis'] = cancer['diagnosis']      # 加上diagnosis列

# Plotly's Scatterplot matrix

dimensions = []
for col in cancer_mean:
    dimensions.append(dict(label=col, values=cancer_mean[col]))

fig = go.Figure(data=go.Splom(      # Splom散点图矩阵
    dimensions=dimensions[:-1],     # 剔除diagnosis列
    showupperhalf=False,       # 只绘制splom的上/下半部分？
    diagonal_visible=False,
    marker=dict(
        color='rgba(135, 206, 250, 0.5)',
        size=5,
        line=dict(
            color='MediumPurple',
            width=0.5))
))

fig.update_layout(
    title='Pairplot for mean attributes of the dataset',
    width=1500,
    height=1500,
)
fig.write_html('Scatterplot matrix.html', auto_open=True)     # 在本地生成网页文件
# fig.show()

# Correlation matrix

plt.figure(figsize = (20, 12), dpi = 150)       # dpi=每英寸的点数（图像分辨率）

corr = cancer.corr()
mask = np.triu(np.ones_like(corr, dtype = bool))       # np.triu:上三角。Return a copy of an array with the elements below the k-th diagonal zeroed.

sns.heatmap(corr,
            mask = mask,    # if passed,data will not be shown in cells where mask is True.
            cmap = 'BuPu',      # 蓝色->紫色
            annot = True,
            linewidths = 0.5,
            fmt = ".2f")

plt.title('Correlation Matrix',
          fontsize = 20,
          weight = 'semibold',
          color = '#de573c')
plt.show()


def subplot_titles(cols):
    """
    Creates titles for the subplot's subplot_titles parameter.
    """
    titles = []
    for i in cols:
        titles.append(i + ' : Distribution')
        titles.append(i + ' : Violin plot')
        titles.append(i + ' by Diagnosis')

    return titles


def subplot(cols, row=0, col=3):
    """
    Takes a dataframe as an input and returns distribution plots for each variable.
    """
    row = len(cols)
    fig = make_subplots(rows=row, cols=3, subplot_titles=subplot_titles(cols))

    for i in range(row):
        fig.add_trace(go.Histogram(x=cancer[cols[i]],opacity=0.7),row=i + 1, col=1)     # 直方图   row/col,subplot在“网页”上排列的位置索引

        fig.add_trace(go.Violin(y=cancer[cols[i]],box_visible=True),row=i + 1, col=2)       # 小提琴图

        fig.add_trace(go.Box(y=cancer[cols[i]][cancer.diagnosis == 0],marker_color='#6ce366',name='Benign'), row=i + 1, col=3)      # 箱形图

        fig.add_trace(go.Box(y=cancer[cols[i]][cancer.diagnosis == 1],marker_color='#de5147',name='Malignant'), row=i + 1, col=3)

        for i in range(row):    # 设置x轴标题
            fig.update_xaxes(title_text=cols[i], row=i + 1)

        fig.update_yaxes(title_text="Count")       # 设置y轴标题
        fig.update_layout(height=450 * row, width=1100,
                          title='Summary of mean tumor attributes (For Diagnois : Green=Benign, Red=Malignant)',
                          showlegend=False,
                          plot_bgcolor="#f7f1cb"
        )
        fig.write_html('Summary of mean tumor attributes.html', auto_open=False)  # 在本地生成网页文件（每一个循环，产生一个网页，故不自动打开）
        # fig.show()

x = subplot(cancer.drop('diagnosis', axis=1).columns)