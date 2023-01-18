# -*- coding: utf-8 -*-
"""
文 件 名: common.py
文件描述: 公共文件
作    者: HanKin
创建日期: 2022.10.28
修改日期：2022.11.30

Copyright (c) 2022 HanKin. All rights reserved.
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