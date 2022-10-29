# -*- coding: utf-8 -*-
"""
文 件 名: common.py
文件描述: 公共文件
作    者: HanKin
创建日期: 2022.10.28
修改日期：2022.10.28

Copyright (c) 2022 HanKin. All rights reserved.
"""

import time
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split # 随机划分为训练子集和测试子集
from sklearn.model_selection import cross_val_score  # 模型评价：训练误差和测试误差
from sklearn.ensemble import RandomForestClassifier  # RFC随机森林分类
from sklearn.preprocessing import LabelEncoder