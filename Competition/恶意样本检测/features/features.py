import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH                    = './data/'                 # 原始数据路径
DATASET_PATH                 = './dataset/'              # 特征工程后数据集路径
TRAIN_WHITE_DATASET_FILENAME = 'train_white_dataset.csv' # 训练集白样本数据集文件名
TRAIN_BLACK_DATASET_FILENAME = 'train_black_dataset.csv' # 训练集黑样本数据集路径
TRAIN_DATASET_FILENAME       = 'train_dataset.csv'       # 训练集样本数据集文件名
TEST_DATASET_FILENAME        = 'test_dataset.csv'        # 测试集样本数据集文件名
TRAIN_DIRTY_DATASET_PATH     = DATASET_PATH+'train_dirty_dataset.csv' # 训练集脏数据集路径
TEST_DIRTY_DATASET_PATH      = DATASET_PATH+'dirty_test_dataset.csv'  # 测试集脏数据集路径

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
    
    
    # 1.删除异常的样本数据
    exception_dataset = dataset[dataset['ExceptionError'] > 0]
    dataset = dataset[dataset['ExceptionError'] == 0]
    
    # 2.删除部分特征数据
    dataset = dataset.drop('ExceptionError', axis=1)
    
    # 3.缺失值处理，用前一个值填充
    #dataset = dataset.fillna(method='ffill')
    
    logger.info('dataset[{}] after shape: {}'.format(csv_path, dataset.shape))
    return exception_dataset, dataset

    
    
    


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

def 

def features_processing(dataset):

def read_csv(filename):
    """从csv文件读取数据集
    """
    
    path = DATA_PATH + filename
    dataset = pd.read_csv(path)
    return dataset

def save_csv(dataset, filename):
    """数据集保存到csv文件
    """
    
    path = DATASET_PATH + filename
    dataset.to_csv(path, sep=',', encoding='utf-8', index=False)

def main():
    # 获取数据集
    train_black_dataset = read_csv(TRAIN_BLACK_DATASET_FILENAME)
    train_white_dataset = read_csv(TRAIN_WHITE_DATASET_FILENAME)
    test_dataset        = read_csv(TEST_DATASET_FILENAME)
    logger.info([train_black_dataset.shape, train_white_dataset.shape, test_dataset.shape])
    
    # 添加标签
    train_black_dataset['label'] = 0
    train_white_dataset['label'] = 1
    
    # 黑白样本合并
    train_dataset = pd.concat([train_black_dataset, train_white_dataset], ignore_index=True)
    
    #train_black_dataset.describe()
    # 6000 4000 9857
    train_black_dataset.shape, train_white_dataset.shape, test_dataset.shape
    features_processing(dataset)
    
if __name__ == '__main__':
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))