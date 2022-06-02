# -*- coding: utf-8 -*-
"""
文 件 名: training.py
文件描述: 训练模型
作    者: HeJian
创建日期: 2022.05.29
修改日期：2022.05.30
Copyright (c) 2022 HeJian. All rights reserved.
"""

import pandas as pd
import numpy as np
import time
from log import logger
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split # 随机划分为训练子集和测试子集
from sklearn.model_selection import cross_val_score  # 模型评价：训练误差和测试误差
from sklearn.feature_selection import SelectFromModel# 特征选择(三种方法)
from sklearn.metrics import roc_auc_score            # 评价指标
from sklearn.metrics import f1_score                 # F1
#from sklearn.cross_validation import StratifiedKFold # K折交叉验证
from sklearn.model_selection import KFold            # K折交叉验证
from sklearn.ensemble import RandomForestClassifier  # RFC随机森林分类
from sklearn.ensemble import ExtraTreesClassifier    # ETC极端随机树分类
import xgboost as xgb                                # XGB
import lightgbm as lgb                               # LGB
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

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
    print('dataset[{}] before shape: {}'.format(csv_path, dataset.shape))
    
    # 1.删除异常的样本数据
    exception_dataset = dataset[dataset['ExceptionError'] > 0]
    dataset = dataset[dataset['ExceptionError'] == 0]
    
    # 2.删除部分特征数据
    #drop_columns = ['ExceptionError', 'HasDebug', 'HasTls', 'HasResources', 'HasRelocations',
    #            'ImageBase', 'ImageSize','EpAddress', 'TimeDateStamp', 'NumberOfExFunctions', 'NumberOfImFunctions']
    drop_columns = ['LinkerVersion', 'ExportRVA', 'ExportSize', 'ResourceSize', 'DebugRVA',
                'DebugSize', 'IATRVA', 'ImageVersion', 'OSVersion', 'StackReserveSize', 'Dll', 'NumberOfSections']
    #dataset = dataset.drop('ExceptionError', axis=1)
    dataset = dataset.drop(drop_columns, axis=1)
    
    # 3.缺失值处理，用前一个值填充
    dataset = dataset.fillna(method='ffill')
    
    print('dataset[{}] after shape: {}'.format(csv_path, dataset.shape))
    return exception_dataset, dataset

def final_score(black_is_black, black_is_white, white_is_black, white_is_white):
    """最终得分

    根据比赛规则计算
    
    Parameters
    ------------
    black_is_black : str
        表示黑样本被预测为黑样本的数目
    black_is_white : str
        表示黑样本被预测为白样本的数目（漏报）
    white_is_black : str
        表示白样本被预测为黑样本的数目（误报）
    white_is_white : str
        表示白样本被预测为白样本的数目
        
    Returns
    -------
    score : float
        分数
    """
    
    # 召回率
    recall = black_is_black / (black_is_black + black_is_white)
    # 误报率
    error_ratio = white_is_black / (white_is_black + white_is_white)
    # 惩罚系数
    alpha = 1.2
    # 分数
    score = recall - alpha * error_ratio
    
    print('score: {}'.format(round(score*100, 2)))
    return round(score*100, 3)

def model_score(model_name, y_test, y_pred):
    """模型得分
    
    根据比赛规则计算
    
    Parameters
    ------------
    model_name : str
        模型名字
    y_test : pandas.Series
        验证集结果
    y_pred : pandas.Series
        预测结果
        
    Returns
    -------
    """
    
    print('model {}:'.format(model_name))
    black_is_black, black_is_white, white_is_black, white_is_white = confusion_matrix(y_test, y_pred).ravel()
    print('black_is_black = {}'.format(black_is_black))
    print('black_is_white = {}'.format(black_is_white))
    print('white_is_black = {}'.format(white_is_black))
    print('white_is_white = {}'.format(white_is_white))
    final_score(black_is_black, black_is_white, white_is_black, white_is_white)

def random_forest_model(X_train, X_test, y_train, y_test):
    """随机森林模型

    根据比赛规则计算
    
    Parameters
    ------------
    black_is_black : str
        表示黑样本被预测为黑样本的数目
    black_is_white : str
        表示黑样本被预测为白样本的数目（漏报）
    white_is_black : str
        表示白样本被预测为黑样本的数目（误报）
    white_is_white : str
        表示白样本被预测为白样本的数目
        
    Returns
    -------
    score : float
        分数
    """

    RFC = RandomForestClassifier()
    RFC.fit(X_train, y_train)
    y_pred = RFC.predict(X_test)

    model_score("RandomForestClassifier", y_test, y_pred)
    return RFC

def extra_trees_model(X_train, X_test, y_train, y_test):
    """使用极端随机树训练
    """
    
    ETC = ExtraTreesClassifier(random_state=0).fit(X_train, y_train)
    y_pred = ETC.predict(X_test)
    
    model_score("ExtraTreesClassifier", y_test, y_pred)
    return ETC

def XGB_model(X_train, X_test, y_train, y_test):
    """使用XGB模型训练
    """
    
    XGB = xgb.XGBClassifier(n_estimators=110, nthread=-1, max_depth=4, seed=0).fit(X_train, y_train,
                            eval_metric="auc", verbose = False, eval_set=[(X_test, y_test)])
    y_pred = XGB.predict(X_test)
    
    model_score("XGBClassifier", y_test, y_pred)
    return XGB

def lightgbm_model(X_train, X_test, y_train, y_test):
    """lightgbm模型训练
    """
    
    LGB = lgb.LGBMClassifier().fit(X_train, y_train)
    y_pred = LGB.predict(X_test).astype(int)
    
    model_score("LGBMClassifier", y_test, y_pred)
    return LGB

def MLP_model(X_train, X_test, y_train, y_test):
    """多层感知器模型训练
    """
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    MLP = MLPClassifier(hidden_layer_sizes=(12, 12, 12, 12, 12, 12))

    MLP.fit(X_train,y_train)
    y_pred = MLP.predict(X_test)
    
    model_score("MLPClassifier", y_test, y_pred)
    return MLP

def main():
    # 获取数据集
    train_black_exdataset, train_black_dataset = get_dataset(TRAIN_BLACK_DATASET_PATH)
    train_white_exdataset, train_white_dataset = get_dataset(TRAIN_WHITE_DATASET_PATH)
    test_exdataset, test_dataset               = get_dataset(TEST_DATASET_PATH)
    print(train_white_exdataset.shape, train_white_dataset.shape)
    
    # 添加标签
    train_black_dataset['label'] = 0
    train_white_dataset['label'] = 1
    
    # 黑白样本合并
    train_dataset = pd.concat([train_black_dataset, train_white_dataset], ignore_index=True)
    
    # 划分训练集和测试集 70% 30%
    x = train_dataset.drop(['label', 'FileName'], axis=1)
    y = train_dataset['label']
    X = np.asarray(x)
    y = np.asarray(y)
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    # 模型训练
    #model = random_forest_model(X_train, X_test, y_train, y_test)
    #model = XGB_model(X_train, X_test, y_train, y_test)
    #model = lightgbm_model(X_train, X_test, y_train, y_test)
    model = extra_trees_model(X_train, X_test, y_train, y_test)
    #model = MLP_model(X_train, X_test, y_train, y_test)

    # 预测结果
    x = test_dataset.drop(['FileName'], axis=1, inplace=False)
    X = np.asarray(x)
    result = model.predict(X)
    
    # 存储结果
    md5 = test_dataset['FileName']
    exmd5 = test_exdataset['FileName']
    md5 = md5.append(exmd5)
    exresult = np.zeros(len(exmd5), dtype=int)
    result = np.append(result, exresult)
    pred = pd.DataFrame({
                'md5'  : md5,
                'label': result
           })
    print('pred shape: {}'.format(pred.shape))
    pred.to_csv('./result.csv', index=False, header=False)
    
if __name__ == '__main__':
    start_time = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S"))

    main()
    
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    end_time = time.time()
    print('process spend {} s.'.format(round(end_time - start_time, 3)))