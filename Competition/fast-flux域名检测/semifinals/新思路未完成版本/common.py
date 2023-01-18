# -*- coding: utf-8 -*-
"""
文 件 名: common.py
文件描述: 公共文件，包含全局变量和包
作    者: 重在参与快乐加倍队
创建日期: 2022.10.18
修改日期：2022.11.22

Copyright (c) 2022 ParticipationDoubled. All rights reserved.
"""

import time
import os
os.environ['NUMEXPR_MAX_THREADS'] = '64'
import psutil
from log import logger
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import pickle
import gc
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split # 随机划分为训练子集和测试子集
from sklearn.model_selection import cross_val_score  # 模型评价：训练误差和测试误差
from sklearn.ensemble import RandomForestClassifier  # RFC随机森林分类
from sklearn import svm                              # SVM支持向量机
from catboost import CatBoostRegressor
from collections import Counter
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD,SparsePCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from gensim.models import Word2Vec
import xgboost as xgb                                # XGB
import lightgbm as lgb                               # LGB
from sklearn.preprocessing import LabelEncoder
import random
import warnings
warnings.filterwarnings('ignore')

RAW_DATASET_PATH          = './'                                                        # 原始数据集路径
DATASET_PATH              = './dataset/'                                                # 特征工程后数据集路径
MODEL_PATH                = './model/'                                                  # 模型路径
OUTPUT_PATH               = './output/'                                                 # 结果输出路径
                                                                          
TRAIN_DATASET_PATH        = RAW_DATASET_PATH+'fastflux_dataset/train/pdns.csv'          # 训练集原始数据集路径
FEATURE_TRAIN_DATASET_PATH= DATASET_PATH+'feature_train_dataset.csv'                    # 训练集特征工程后数据集路径
TRAIN_LABEL_PATH          = RAW_DATASET_PATH+'fastflux_dataset/train/fastflux_tag.csv'  # 训练集标签路径
TEST_DATASET_PATH         = RAW_DATASET_PATH+'fastflux_dataset/test/pdns.csv'           # 测试集原始数据集路径
#TEST_DATASET_PATH         = '/home/aifirst/datasets/aichampion/fast_flux/data/files/test.csv'
FEATURE_TEST_DATASET_PATH = DATASET_PATH+'feature_test_dataset.csv'                     # 测试集特征工程后数据集路径
RAW_IP_INFO_PATH          = RAW_DATASET_PATH+'fastflux_dataset/ip_info_recharge.csv'    # ip_info原始数据集路径
IP_INFO_PATH              = DATASET_PATH+'ip_info.csv'                                  # ip_info处理后的数据集路径

RESULT_PATH               = OUTPUT_PATH+'result.csv'                                    # 预测结果文件存储路径
BASELINE_MODEL_PATH       = MODEL_PATH+'model.pkl'                                      # 模型文件存储路径
BASELINE_MODEL_SCORE_PATH = MODEL_PATH+'model.score'                                    # 模型分数存储路径
PRE_DATASET_PATH          = DATASET_PATH+'pre_dataset.csv'

# 创建数据集路径文件夹
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# 创建模型文件夹
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 创建结果输出文件夹
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# 线程数量
THREAD_NUM = 64

# 概率阈值
THRESHOLD = 0.25

def show_memory_info(hint):
    """显示内存使用情况
    """
    
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    logger.info('{} memory used {:5.2f} MB'.format(hint, memory))