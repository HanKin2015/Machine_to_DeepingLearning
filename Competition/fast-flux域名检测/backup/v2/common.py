# -*- coding: utf-8 -*-
"""
文 件 名: common.py
文件描述: 公共文件，包含全局变量和包
作    者: HanKin
创建日期: 2022.10.18
修改日期：2022.10.18

Copyright (c) 2022 HanKin. All rights reserved.
"""

import time
import os
from log import logger
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split # 随机划分为训练子集和测试子集
from sklearn.model_selection import cross_val_score  # 模型评价：训练误差和测试误差
from sklearn.ensemble import RandomForestClassifier  # RFC随机森林分类
from sklearn import svm                              # SVM支持向量机
import xgboost as xgb                                # XGB
import lightgbm as lgb                               # LGB
from sklearn.preprocessing import LabelEncoder

RAW_DATASET_PATH          = './'
DATASET_PATH              = './dataset/'                                               # 数据集路径
MODEL_PATH                = './model/'                                                 # 模型路径
                                                                                       
RAW_TRAIN_DATASET_PATH    = RAW_DATASET_PATH+'fastflux_dataset/train/pdns.csv'         # 训练集原始数据集路径
TRAIN_DATASET_PATH        = DATASET_PATH+'train_dataset.csv'                         # 训练集特征工程数据集路径
FEATURE_TRAIN_DATASET_PATH= DATASET_PATH+'feature_train_dataset.csv' 
TRAIN_LABEL_PATH          = RAW_DATASET_PATH+'fastflux_dataset/train/fastflux_tag.csv' # 训练集标签路径
RAW_TEST_DATASET_PATH     = RAW_DATASET_PATH+'fastflux_dataset/test/pdns.csv'          # 测试集原始数据集路径
TEST_DATASET_PATH         = DATASET_PATH+'test_dataset.csv'                          # 测试集特征工程数据集路径
FEATURE_TEST_DATASET_PATH = DATASET_PATH+'feature_test_dataset.csv' 
RAW_IP_INFO_PATH          = './ip_info/ip_info.csv'
IP_INFO_PATH              = DATASET_PATH+'ip_info.csv'


RESULT_PATH               = './result.csv'                                             # 预测结果存储路径
RFC_MODEL_PATH            = MODEL_PATH+'rfc_model.pkl'
RFC_MODEL_SCORE_PATH      = MODEL_PATH+'rfc_model.score'
BASELINE_MODEL_PATH       = MODEL_PATH+'model.pkl'
BASELINE_MODEL_SCORE_PATH = MODEL_PATH+'model.score'

# 创建数据集路径文件夹
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# 创建模型文件夹
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 线程数量
THREAD_NUM = 64