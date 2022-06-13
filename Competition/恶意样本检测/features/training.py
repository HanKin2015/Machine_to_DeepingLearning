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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import StackingClassifier

# 数据集路径
DATASET_PATH = './dataset/'
# 训练集样本数据集路径
TRAIN_DATASET_PATH = DATASET_PATH+'train_dataset.csv'
# 模型路径
MODEL_PATH = './model/'
# 恶意样本检测训练模型路径
MALICIOUS_SAMPLE_DETECTION_MODEL_PATH = MODEL_PATH+'malicious_sample_detection.model'
# 恶意样本检测特征选择器路径
MALICIOUS_SAMPLE_DETECTION_SELECTOR_PATH = MODEL_PATH+'malicious_sample_detection.selector'
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

def feature_selection_model(model):
    RFC = RandomForestClassifier(n_estimators=50).fit(X, y)
    select = SelectFromModel(RFC, prefit=True)
    X = select.transform(X)
    return select

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

    """
    RFC = RandomForestClassifier().fit(X_train, y_train)
    selector = SelectFromModel(RFC, prefit=True)
    save_feature_selector(selector)
    
    logger.info('X_train before selector shape: ({}, {}).'.format(X_train.shape[0], X_train.shape[1]))
    X_train = selector.transform(X_train)
    X_test  = selector.transform(X_test)
    logger.info('X_train after selector shape: ({}, {}).'.format(X_train.shape[0], X_train.shape[1]))
    """
    # K折交叉验证与学习曲线的联合使用来获取最优K值
    scores_cross = []
    Ks = []
    for k in range(86, 87):
        RFC = RandomForestClassifier(n_estimators=k, n_jobs=-1, oob_score=True, class_weight='balanced')  # 实例化模型对象
        score_cross = cross_val_score(RFC, X_train, y_train, cv=10).mean()  # 根据训练数据进行交叉验证，并返回交叉验证的评分
        scores_cross.append(score_cross)
        Ks.append(k)
        
    # 转为数组类型
    scores_arr = np.array(scores_cross)
    Ks_arr = np.array(Ks)
    # 绘制学习曲线
    #plt.plot(Ks, scores_cross)
    #plt.show()
    
    # 获取最高的评分，以及最高评分对应的数组下标，从而获得最优的K值
    score_best = scores_arr.max()  # 在存储评分的array中找到最高评分
    index_best = scores_arr.argmax()  # 找到array中最高评分所对应的下标
    Ks_best = Ks_arr[index_best]  # 根据下标找到最高评分所对应的K值
    logger.info('Ks_best: {}.'.format(Ks_best)) 

    RFC = RandomForestClassifier(n_estimators=Ks_best)
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
    params = {
          # 这些参数需要学习
          'boosting_type': 'gbdt',
          #'boosting_type': 'dart',
          'objective': 'multiclass',
          'metric': 'multi_logloss',  # 评测函数，这个比较重要
          'min_child_weight': 1.5,
          'num_leaves': 2**5,
          'lambda_l2': 10,
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'tree_method': 'exact',
          'seed': 2022,
          'learning_rate': 0.01, # 学习率 重要
          'num_class': 2,  # 重要
          'silent': True,
          }
    #LGB = lgb.LGBMClassifier(params).fit(X_train, y_train)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=0)

    train_matrix = lgb.Dataset(X_train, label=y_train)
    test_matrix = lgb.Dataset(X_test, label=y_test)

    num_round = 200  # 训练的轮数
    early_stopping_rounds = 10
    LGB = lgb.train(params, 
                  train_matrix,
                  num_round,
                  valid_sets=test_matrix,
                  early_stopping_rounds=early_stopping_rounds)
    y_pred = LGB.predict(X_test, num_iteration=LGB.best_iteration).astype(int)
    
    logger.info('score : ', np.mean((y_pred[:,1]>0.5)==y_valid))
    #score = model_score('LGBMClassifier', y_test, y_pred)
    score = np.mean((y_pred[:,1]>0.5)==y_valid)
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

def fusion_model(X, y):
    """将RF、XGBoost、LightGBM融合（单层Stacking）
    """
    
    # 模型列表
    models = [('RFC', RandomForestClassifier()), 
              ('XGB', xgb.XGBClassifier()),
              ('LGB', lgb.LGBMClassifier())
             ]
    # 创建stacking模型
    model = StackingClassifier(models)
    # 设置验证集数据划分方式
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # 验证模型精度
    n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # 打印模型的精度
    logger.info('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

def save_training_model(model, score):
    """保存训练模型
    """
    
    before_score = 0
    with open(MODEL_SCORE_PATH, 'r') as fd:
        before_score = fd.read()
    if score > float(before_score):
        buffer = pickle.dumps(model)
        with open(MALICIOUS_SAMPLE_DETECTION_MODEL_PATH, "wb+") as fd:
            fd.write(buffer)
        with open(MODEL_SCORE_PATH, 'w') as fd:
            fd.write(str(score))

def save_feature_selector(selector):
    """保存特征选择模型
    """
    
    buffer = pickle.dumps(selector)
    with open(MALICIOUS_SAMPLE_DETECTION_SELECTOR_PATH, "wb+") as fd:
        fd.write(buffer)

def main():
    # 获取数据集
    train_dataset = pd.read_csv(TRAIN_DATASET_PATH)
    logger.info('train dataset shape: ({}, {}).'.format(train_dataset.shape[0], train_dataset.shape[1]))
    
    # 划分训练集和测试集 80% 20%
    X = train_dataset.drop(['label', 'FileName'], axis=1).values
    y = train_dataset['label'].values

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logger.info([X_train.shape, X_test.shape, y_train.shape, y_test.shape])
        
    # 模型训练
    #model, score = random_forest_model(X_train, X_test, y_train, y_test)
    #model, score = XGB_model(X_train, X_test, y_train, y_test)
    model, score = lightgbm_model(X_train, X_test, y_train, y_test)
    #model, score = extra_trees_model(X_train, X_test, y_train, y_test)
    #model, score = MLP_model(X_train, X_test, y_train, y_test)
    #model, score = gradient_boosting_model(X_train, X_test, y_train, y_test)
    #fusion_model(X, y)
    #save_training_model(model, score)

if __name__ == '__main__':
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))