import pandas as pd

from warnings import filterwarnings
filterwarnings("ignore")


cancer = pd.read_csv('.\KNN\Breast Cancer_data.csv')
cancer = cancer.drop('id', axis=1)

# 探索性数据分析
def EDA(df):
    print('\033[1m' + 'Shape of the data :' + '\033[0m')
    print(df.shape,
          '\n------------------------------------------------------------------------------------\n')

    print('\033[1m' + 'All columns from the dataframe :' + '\033[0m')
    print(df.columns,
          '\n------------------------------------------------------------------------------------\n')

    print('\033[1m' + 'Datatpes and Missing values:' + '\033[0m')
    print(df.info(),
          '\n------------------------------------------------------------------------------------\n')

    print('\033[1m' + 'Summary statistics for the data' + '\033[0m')
    print(df.describe(include='all'),
          '\n------------------------------------------------------------------------------------\n')

    print('\033[1m' + 'Outliers in the data :' + '\033[0m')
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
    print(outliers.sum(),
          '\n------------------------------------------------------------------------------------\n')

    print('\033[1m' + 'Memory used by the data :' + '\033[0m')
    print(df.memory_usage(),
          '\n------------------------------------------------------------------------------------\n')

    print('\033[1m' + 'Number of duplicate values :' + '\033[0m')
    print(df.duplicated().sum())


EDA(cancer)