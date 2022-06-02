import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 数据集路径
DATASET_PATH = './dataset/'
# 训练集白样本数据集路径
TRAIN_WHITE_DATASET_PATH = DATASET_PATH+'train_white_dataset.csv'
# 训练集黑样本数据集路径
TRAIN_BLACK_DATASET_PATH = DATASET_PATH+'train_black_dataset.csv'
# 测试集样本数据集路径
TEST_DATASET_PATH = DATASET_PATH+'test_dataset.csv'
    
    
    
def get_dataset(csv_path):
    """获取数据集

    读取csv文件，并做简单的特征处理
    
    Parameters
    ------------
    csv_path : str
        数据集csv文件路径
        
    Returns
    -------
    dataset : pandas.DataFrame
        数据集
    """
    
    dataset = pd.read_csv(csv_path)
    logger.info('dataset[{}] before shape: {}'.format(csv_path, dataset.shape))
    
    # 1.删除异常的样本数据
    exception_dataset = dataset[dataset['ExceptionError'] > 0]
    dataset = dataset[dataset['ExceptionError'] == 0]
    
    # 2.删除部分特征数据
    #drop_columns = ['ExceptionError', 'HasDebug', 'HasTls', 'HasResources', 'HasRelocations',
    #            'ImageBase', 'ImageSize','EpAddress', 'TimeDateStamp', 'NumberOfExFunctions', 'NumberOfImFunctions']
    #drop_columns = ['LinkerVersion', 'ExportRVA', 'ExportSize', 'ResourceSize', 'DebugRVA',
    #            'DebugSize', 'IATRVA', 'ImageVersion', 'OSVersion', 'StackReserveSize', 'Dll', 'NumberOfSections']
    #drop_columns = ['NumberOfSections', 'TimeDateStamp', 'ExceptionError', 'ImageBase', 'ImageSize', 'EpAddress', 'ExportSize', 'HasResources', 'HasDebug', 'HasTls', 'DebugSize', 'StackReserveSize']
    
    #drop_columns = ['ExceptionError', 'ImageBase', 'ImageSize', 'EpAddress', 'ExportSize', 'TimeDateStamp', 'DebugSize', 'ResourceSize', 'NumberOfSections']

    #dataset = dataset.drop(['ImageBase', 'ImageSize', 'EpAddress', 'ExportSize', 'TimeDateStamp', 'DebugSize', 'ResourceSize', 'NumberOfSections', 'ExceptionError'], axis=1)
    #dataset = dataset.drop(drop_columns, axis=1)
    dataset = dataset.drop('ExceptionError', axis=1)
    
    # 3.缺失值处理，用前一个值填充
    #dataset = dataset.fillna(method='ffill')
    
    logger.info('dataset[{}] after shape: {}'.format(csv_path, dataset.shape))
    return exception_dataset, dataset

    
    
    
train_black_dataset = pd.read_csv(TRAIN_BLACK_DATASET_PATH)
train_white_dataset = pd.read_csv(TRAIN_WHITE_DATASET_PATH)
test_dataset = pd.read_csv(TEST_DATASET_PATH)
#train_black_dataset.describe()
# 6000 4000 9857
train_black_dataset.shape, train_white_dataset.shape, test_dataset.shape

dataset = [train_black_dataset, train_white_dataset, test_dataset]
print(train_black_dataset[train_black_dataset['ExceptionError'] > 0].shape[0],
      train_white_dataset[train_white_dataset['ExceptionError'] > 0].shape[0],
      test_dataset[test_dataset['ExceptionError'] > 0].shape[0])

train_black_dataset_useful = train_black_dataset[train_black_dataset['ExceptionError'] == 0]
train_black_dataset_exceptional = train_black_dataset[train_black_dataset['ExceptionError'] > 0]
train_white_dataset_useful = train_white_dataset[train_white_dataset['ExceptionError'] == 0]
train_white_dataset_exceptional = train_white_dataset[train_white_dataset['ExceptionError'] > 0]
print(train_black_dataset_useful.shape, train_black_dataset_exceptional.shape, train_white_dataset_useful.shape, train_white_dataset_exceptional.shape)


train_black_dataset = train_black_dataset_useful
train_white_dataset = train_white_dataset_useful
train_black_dataset.loc[:, 'label'] = 0
train_white_dataset.loc[:, 'label'] = 1

import seaborn
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import warnings                      # 消除警告
warnings.filterwarnings("ignore")
def change_outlier_to_mean(feature, threshold):
    tmp = int(train_black_dataset[train_black_dataset[feature] < threshold][feature].mean())
    train_black_dataset.loc[train_black_dataset[feature] > threshold, feature] = tmp
    
    tmp = int(train_white_dataset[train_white_dataset[feature] < threshold][feature].mean())
    train_white_dataset.loc[train_white_dataset[feature] > threshold, feature] = tmp
    
    tmp = int(test_dataset[test_dataset[feature] < threshold][feature].mean())
    test_dataset.loc[test_dataset[feature] > threshold, feature] = tmp
change_outlier_to_mean('StackReserveSize', 1e8)
change_outlier_to_mean('NumberOfExFunctions', 20000)
change_outlier_to_mean('ExportRVA', 3e7)
change_outlier_to_mean('DebugRVA', 3e7)
change_outlier_to_mean('IATRVA', 3e7)
change_outlier_to_mean('StackReserveSize', 2e7)


drop_columns = ['ImageBase', 'ImageSize', 'EpAddress', 'ExportSize', 'TimeDateStamp',
 'DebugSize', 'ResourceSize', 'NumberOfSections', 'label']
train_black_dataset.drop(drop_columns, axis=1).to_csv('./hj/train_black_dataset.csv', sep=',', encoding='utf-8', index=False)
train_white_dataset.drop(drop_columns, axis=1).to_csv('./hj/train_white_dataset.csv', sep=',', encoding='utf-8', index=False)
test_dataset.drop(drop_columns, axis=1).to_csv('./hj/test_dataset.csv', sep=',', encoding='utf-8', index=False)

hj = train_black_dataset[train_black_dataset['ImageSize'] > 10000]