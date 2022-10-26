# -*- coding: utf-8 -*-
"""
文 件 名: get_data.py
文件描述: 获取数据
作    者: HanKin
创建日期: 2022.07.20
修改日期：2022.07.20

Copyright (c) 2022 HanKin. All rights reserved.
"""

import time
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split # 随机划分为训练子集和测试子集
from sklearn.model_selection import cross_val_score  # 模型评价：训练误差和测试误差
from sklearn.ensemble import RandomForestClassifier  # RFC随机森林分类

TRAIN_RAW_DATA_PATH = './idle_machine_detect/train/data/'   # 训练集原始数据路径
TRAIN_DATA_PATH     = './train_data.csv'                    # 训练集数据路径
TRAIN_DATASET_PATH  = './train_dataset.csv'                 # 训练集数据集路径
TEST_RAW_DATA_PATH  = './idle_machine_detect/test/data/'    # 测试集原始数据路径
TEST_DATA_PATH      = './test_data.csv'                     # 测试集数据路径
TEST_DATASET_PATH   = './test_dataset.csv'                  # 测试集数据集路径
TRAIN_LABELS_PATH   = './idle_machine_detect/train/labels.csv'
THREAD_NUM          = 64
RESULT_PATH         = './result.csv'