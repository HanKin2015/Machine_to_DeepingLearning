# -*- coding: utf-8 -*-
"""
文 件 名: common.py
文件描述: 公共文件库
作    者: HeJian
创建日期: 2022.06.15
修改日期：2022.06.16
Copyright (c) 2022 HeJian. All rights reserved.
"""

import os, re, time, datetime
os.environ['NUMEXPR_MAX_THREADS'] = '64'
import subprocess
from log import logger
from PIL import Image
import binascii
import pefile
from capstone import *
import pickle
from collections import *
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split # 随机划分为训练子集和测试子集
from sklearn.model_selection import cross_val_score  # 模型评价：训练误差和测试误差
from sklearn.feature_selection import SelectFromModel# 特征选择(三种方法)
from sklearn.metrics import roc_auc_score            # 评价指标
from sklearn.metrics import f1_score                 # F1
from sklearn.model_selection import KFold            # K折交叉验证
from sklearn.ensemble import RandomForestClassifier  # RFC随机森林分类
from sklearn.ensemble import ExtraTreesClassifier    # ETC极端随机树分类
import xgboost as xgb                                # XGB
import lightgbm as lgb                               # LGB
from sklearn import tree
from sklearn.feature_extraction import FeatureHasher
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

DATA_PATH                                = './data/'                                        # 数据路径
DATASET_PATH                             = './dataset/'                                     # 数据集路径
SAMPLE_PATH                              = './AIFirst_data/'                                # 样本数据集存储路径
TRAIN_WHITE_PATH                         = SAMPLE_PATH+'train/white/'                       # 训练集白样本路径
TRAIN_BLACK_PATH                         = SAMPLE_PATH+'train/black/'                       # 训练集黑样本路径
TEST_PATH                                = SAMPLE_PATH+'test/'                              # 测试集样本路径
TRAIN_WHITE_GRAY_IMAGES_PATH             = './gray_images/train/white/'                     # 训练集白样本灰度图像存储路径
TRAIN_BLACK_GRAY_IMAGES_PATH             = './gray_images/train/black/'                     # 训练集黑样本灰度图像存储路径
TEST_GRAY_IMAGES_PATH                    = './gray_images/test/'                            # 测试集样本灰度图像存储路径
MODEL_PATH                               = './model/'                                       # 模型路径

TRAIN_DATASET_PATH                       = DATASET_PATH+'train_dataset.csv'                 # 训练集样本数据集路径
TEST_DATASET_PATH                        = DATASET_PATH+'test_dataset.csv'                  # 训练集样本数据集路径
TEST_DIRTY_DATASET_PATH                  = DATASET_PATH+'test_dirty_dataset.csv'            # 测试集脏数据集路径

TRAIN_WHITE_IMAGE_MATRIX_PATH            = DATA_PATH+'train_white_image_matrix.csv'         # 训练集白样本图像矩阵数据集存储路径
TRAIN_BLACK_IMAGE_MATRIX_PATH            = DATA_PATH+'train_black_image_matrix.csv'         # 训练集黑样本图像矩阵数据集存储路径
TEST_IMAGE_MATRIX_PATH                   = DATA_PATH+'test_image_matrix.csv'                # 测试集样本图像矩阵数据集存储路径
TRAIN_BLACK_0_3000_IMAGE_MATRIX_PATH     = DATA_PATH+'train_black_0_3000_image_matrix.csv'  # 训练集黑样本图像矩阵数据集存储路径
TRAIN_BLACK_3000_IMAGE_MATRIX_PATH       = DATA_PATH+'train_black_3000_image_matrix.csv'    # 训练集黑样本图像矩阵数据集存储路径
TEST_0_3000_IMAGE_MATRIX_PATH            = DATA_PATH+'test_0_3000_image_matrix.csv'         # 测试集样本操作指令码3-gram特征存储路径
TEST_3000_6000_IMAGE_MATRIX_PATH         = DATA_PATH+'test_3000_6000_image_matrix.csv'      # 测试集样本操作指令码3-gram特征存储路径
TEST_6000_IMAGE_MATRIX_PATH              = DATA_PATH+'test_6000_image_matrix.csv'           # 测试集样本操作指令码3-gram特征存储路径

TRAIN_WHITE_DATASET_FILENAME             = 'train_white_dataset.csv'                        # 训练集白样本数据集文件名
TRAIN_BLACK_DATASET_FILENAME             = 'train_black_dataset.csv'                        # 训练集黑样本数据集路径
TRAIN_DATASET_FILENAME                   = 'train_dataset.csv'                              # 训练集样本数据集文件名
TEST_DATASET_FILENAME                    = 'test_dataset.csv'                               # 测试集样本数据集文件名
TRAIN_DIRTY_DATASET_FILENAME             = 'train_dirty_dataset.csv'                        # 训练集脏数据集文件名
TEST_DIRTY_DATASET_FILENAME              = 'test_dirty_dataset.csv'                         # 测试集脏数据集文件名
TRAIN_WHITE_CUSTOM_STRINGS_PATH          = 'train_white_strings.csv'                        # 训练集白样本自定义字符串数据集文件名
TRAIN_BLACK_CUSTOM_STRINGS_PATH          = 'train_black_strings.csv'                        # 训练集黑样本自定义字符串数据集文件名
TEST_CUSTOM_STRINGS_PATH                 = 'test_strings.csv'                               # 测试集样本自定义字符串数据集文件名
TRAIN_WHITE_STRING_FEATURES_PATH         = DATA_PATH+'train_white_string_features.csv'      # 训练集白样本字符串特征数据集文件名
TRAIN_BLACK_STRING_FEATURES_PATH         = DATA_PATH+'train_black_string_features.csv'      # 训练集黑样本字符串特征数据集文件名
TEST_STRING_FEATURES_PATH                = DATA_PATH+'test_string_features.csv'             # 测试集样本字符串特征数据集文件名

TRAIN_WHITE_OPCODE_3_GRAM_PATH           = DATA_PATH+'train_white_opcode_3_gram.csv'        # 训练集白样本操作指令码3-gram特征存储路径
TRAIN_BLACK_OPCODE_3_GRAM_PATH           = DATA_PATH+'train_black_opcode_3_gram.csv'        # 训练集黑样本操作指令码3-gram特征存储路径
TEST_OPCODE_3_GRAM_PATH                  = DATA_PATH+'test_opcode_3_gram.csv'               # 测试集样本操作指令码3-gram特征存储路径
TEST_0_3000_OPCODE_3_GRAM_PATH           = DATA_PATH+'test_0_3000_opcode_3_gram.csv'        # 测试集0-3000样本操作指令码3-gram特征存储路径
TEST_3000_6000_OPCODE_3_GRAM_PATH        = DATA_PATH+'test_3000_6000_opcode_3_gram.csv'     # 测试集3000-6000样本操作指令码3-gram特征存储路径
TEST_6000_OPCODE_3_GRAM_PATH             = DATA_PATH+'test_6000_opcode_3_gram.csv'          # 测试集6000-样本操作指令码3-gram特征存储路径

MODEL_SCORE_PATH                         = MODEL_PATH+'score'                               # 模型分数路径
IAMGE_MATRIX_RFC_MODEL_PATH              = MODEL_PATH+'image_matrix_rfc.model'              # RF模型路径
IAMGE_MATRIX_XGB_MODEL_PATH              = MODEL_PATH+'image_matrix_xgb.model'              # XGB模型路径
IAMGE_MATRIX_LGB_MODEL_PATH              = MODEL_PATH+'image_matrix_lgb.model'              # LGB模型路径
MALICIOUS_SAMPLE_DETECTION_MODEL_PATH    = MODEL_PATH+'malicious_sample_detection.model'    # 恶意样本检测训练模型路径
MALICIOUS_SAMPLE_DETECTION_SELECTOR_PATH = MODEL_PATH+'malicious_sample_detection.selector' # 恶意样本检测特征选择器路径
OPCODE_N_GRAM_MODEL_PATH                 = MODEL_PATH+'opcode_n_gram.model'
OPCODE_N_GRAM_MODEL_SCORE_PATH           = MODEL_PATH+'opcode_n_gram.score'
DIRTY_DATASET_MODEL_PATH                 = MODEL_PATH+'dirty_dataset_rfc.model'             # 脏样本训练模型路径
DIRTY_DATASET_MODEL_SCORE_PATH           = MODEL_PATH+'dirty_dataset_rfc.score'             # 脏样本模型分数路径
RESULT_PATH                              = './result.csv'                                   # 预测结果存储路径

# 创建数据集路径文件夹
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# 创建模型文件夹
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 线程数量
THREAD_NUM = 64
