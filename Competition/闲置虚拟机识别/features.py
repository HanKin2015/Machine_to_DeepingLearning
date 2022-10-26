# -*- coding: utf-8 -*-
"""
文 件 名: brute_ftp_by_dict.py
文件描述: 使用字典暴力破解登陆ftp
作    者: HanKin
创建日期: 2022.07.18
修改日期：2022.07.18

Copyright (c) 2022 HanKin. All rights reserved.
"""

from common import *

def main():
    train_features = pd.read_csv(TRAIN_DATA_PATH)
    train_label    = pd.read_csv(TRAIN_LABELS_PATH)
    train_dataset  = pd.merge(train_features, train_label, on='ids')
    train_dataset.to_csv(TRAIN_DATASET_PATH, sep=',', encoding='utf-8', index=False)

if __name__ == '__main__':
    #os.system('chcp 936 & cls')
    start_time = time.time()

    main()

    end_time = time.time()
    print('process spend {} s.'.format(round(end_time - start_time, 3)))




