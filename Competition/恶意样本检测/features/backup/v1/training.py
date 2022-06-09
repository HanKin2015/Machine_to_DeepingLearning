# -*- coding: utf-8 -*-
"""
文 件 名: training.py
文件描述: 训练模型
作    者: HeJian
创建日期: 2022.05.29
修改日期：2022.05.30
Copyright (c) 2022 HeJian. All rights reserved.
"""

import os
os.environ['NUMEXPR_MAX_THREADS'] = '64'
import pandas as pd
import numpy as np
import time
import datetime
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold

# 数据集路径
DATASET_PATH = './dataset/'
# 训练集样本数据集路径
TRAIN_DATASET_PATH = DATASET_PATH+'train_dataset.csv'
# 模型路径
MODEL_PATH = './model/'
# 恶意样本检测模型路径
MALICIOUS_SAMPLE_DETECTION_MODEL_PATH = MODEL_PATH+'malicious_sample_detection.model'
# 模型分数路径
MODEL_SCORE_PATH = MODEL_PATH+'score'

# 创建模型文件夹
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

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
    
    logger.info('model {}:'.format(model_name))
    black_is_black, black_is_white, white_is_black, white_is_white = confusion_matrix(y_test, y_pred).ravel()
    logger.info('black_is_black = {}'.format(black_is_black))
    logger.info('black_is_white = {}'.format(black_is_white))
    logger.info('white_is_black = {}'.format(white_is_black))
    logger.info('white_is_white = {}'.format(white_is_white))
    
    # 召回率
    recall = black_is_black / (black_is_black + black_is_white)
    # 误报率
    error_ratio = white_is_black / (white_is_black + white_is_white)
    # 惩罚系数
    alpha = 1.2
    # 分数
    score = recall - alpha * error_ratio
    
    logger.info('recall: {}, error_ratio: {}, score: {}'.format(
        round(recall, 4), round(error_ratio, 4), round(score*100, 2)))
    return round(score*100, 2)

def save_test_pred(X_test, y_test, y_pred, score):
    df = pd.DataFrame(X_test)
    df['y_test'] = y_test.reshape(-1,1)
    df['y_pred'] = y_pred.reshape(-1,1)
    logger.info('test_pred result shape: {}'.format(df.shape))
    df.to_csv('he/{}.csv'.format(score), index=False, header=False)

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

    #RFC = RandomForestClassifier(random_state=0)
    RFC = RandomForestClassifier()
    RFC.fit(X_train, y_train)
    y_pred = RFC.predict(X_test)

    score = model_score('RandomForestClassifier', y_test, y_pred)
    
    save_test_pred(X_test, y_test, y_pred, score)
    return RFC, score

def extra_trees_model(X_train, X_test, y_train, y_test):
    """使用极端随机树训练
    """
    
    ETC = ExtraTreesClassifier(random_state=0).fit(X_train, y_train)
    y_pred = ETC.predict(X_test)
    
    score = model_score('ExtraTreesClassifier', y_test, y_pred)
    return ETC, score

def XGB_model(X_train, X_test, y_train, y_test):
    """使用XGB模型训练
    """
    
    XGB = xgb.XGBClassifier(n_estimators=110, nthread=-1, max_depth=4, seed=0).fit(X_train, y_train,
                            eval_metric="auc", verbose = False, eval_set=[(X_test, y_test)])
    y_pred = XGB.predict(X_test)
    
    score = model_score('XGBClassifier', y_test, y_pred)
    return XGB, score

def lightgbm_model(X_train, X_test, y_train, y_test):
    """lightgbm模型训练
    """
    
    LGB = lgb.LGBMClassifier().fit(X_train, y_train)
    y_pred = LGB.predict(X_test).astype(int)
    
    score = model_score('LGBMClassifier', y_test, y_pred)
    return LGB, score

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
    
    score = model_score('MLPClassifier', y_test, y_pred)
    return MLP, score

def gradient_boosting_model(X_train, X_test, y_train, y_test):
    """梯度提升决策树模型训练
    """
    
    original_params = {'n_estimators': 1000, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': 2, 'min_samples_split': 5}
    setting = {'learning_rate': 0.1, 'max_features': 2}
    params = dict(original_params)
    params.update(setting)
 
    GBDT = GradientBoostingClassifier(**params)
    GBDT.fit(X_train, y_train)
    y_pred = GBDT.predict(X_test)
    
    score = model_score('GradientBoostingClassifier', y_test, y_pred)
    return GBDT, score

def save_model(model, score):
    """保存模型
    """
    
    buffer = pickle.dumps(model)
    with open(MALICIOUS_SAMPLE_DETECTION_MODEL_PATH, "wb+") as fd:
        fd.write(buffer)
    with open(MODEL_SCORE_PATH, 'w') as fd:
        fd.write(str(score))

def main():
    # 获取数据集
    train_dataset = pd.read_csv(TRAIN_DATASET_PATH)
    logger.info('train dataset shape: ({}, {}).'.format(train_dataset.shape[0], train_dataset.shape[1]))
    
    # 划分训练集和测试集 70% 30%
    x = train_dataset.drop(['label', 'FileName'], axis=1)
    y = train_dataset['label']
    X = np.asarray(x)
    y = np.asarray(y)
    
    # 进行方差过滤时阈值设置为中位数
    selector = VarianceThreshold()
    X_var = selector.fit_transform(X)
    # X_var是方差过滤掉特征之后的特征矩阵 
    X_chi = SelectKBest(chi2, k=20).fit_transform(X_var, y)
    #cross_val_score(RFC(n_estimators=100, random_state=0), X_chi, y, cv=5).mean()
    X = X_chi
    
    best_model = None
    best_score  = 0
    for k, (train_indices, test_indices) in enumerate(KFold(10, shuffle=True, random_state=0).split(y), start=1):
        #X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        X_train,X_test,y_train,y_test = X[train_indices], X[test_indices], y[train_indices], y[test_indices]
        logger.info([X_train.shape, X_test.shape, y_train.shape, y_test.shape])
        
        # 模型训练
        model, score = random_forest_model(X_train, X_test, y_train, y_test)
        if score > best_score:
            best_score = score
            best_model = model
        #model, score = XGB_model(X_train, X_test, y_train, y_test)
    #model, score = lightgbm_model(X_train, X_test, y_train, y_test)
    #model, score = extra_trees_model(X_train, X_test, y_train, y_test)
    #model, score = MLP_model(X_train, X_test, y_train, y_test)
    #model, score = gradient_boosting_model(X_train, X_test, y_train, y_test)
    save_model(best_model, best_score)

if __name__ == '__main__':
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))