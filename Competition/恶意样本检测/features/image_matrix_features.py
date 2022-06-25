# -*- coding: utf-8 -*-
"""
文 件 名: image_matrix_features.py
文件描述: 图像矩阵特征工程
作    者: HeJian
创建日期: 2022.06.24
修改日期：2022.06.24
Copyright (c) 2022 HeJian. All rights reserved.
"""

from common import *

def main():
    # 获取数据集
    train_white_dataset = pd.read_csv(TRAIN_WHITE_IMAGE_MATRIX_PATH)
    train_black_dataset = pd.read_csv(TRAIN_BLACK_IMAGE_MATRIX_PATH)
    test_dataset = pd.read_csv(TEST_IMAGE_MATRIX_PATH)
    logger.info('train white dataset shape: ({}, {}).'.format(train_white_dataset.shape[0], train_white_dataset.shape[1]))
    logger.info('train black dataset shape: ({}, {}).'.format(train_black_dataset.shape[0], train_black_dataset.shape[1]))
    logger.info('test dataset shape: ({}, {}).'.format(test_dataset.shape[0], test_dataset.shape[1]))
    
    # 添加标签
    train_black_dataset['Label'] = 0
    train_white_dataset['Label'] = 1
    
    # 所有样本合并
    train_dataset = pd.concat([train_black_dataset, train_white_dataset], ignore_index=True, sort=False)
    logger.info('dataset shape: ({}, {}).'.format(train_dataset.shape[0], train_dataset.shape[1]))
    
    # 填充缺失值
    train_dataset = train_dataset.fillna(0)
    test_dataset = test_dataset.fillna(0)
    
    # 保存
    train_dataset.to_csv(TRAIN_IMAGE_MATRIX_PATH, sep=',', encoding='utf-8', index=False)
    test_dataset.to_csv(TEST_IMAGE_MATRIX_PATH_, sep=',', encoding='utf-8', index=False)

if __name__ == '__main__':
    logger.info('******** starting ********')
    start_time = time.time()

    main()
    
    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))









