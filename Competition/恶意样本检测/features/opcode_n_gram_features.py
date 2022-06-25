# -*- coding: utf-8 -*-
"""
文 件 名: opcode_n_gram_features.py
文件描述: 操作指令码n-gram特征工程
作    者: HeJian
创建日期: 2022.06.22
修改日期：2022.06.22
Copyright (c) 2022 HeJian. All rights reserved.
"""

from common import *

def main():
    # 获取数据集
    train_white_dataset = pd.read_csv(TRAIN_WHITE_OPCODE_3_GRAM_PATH)
    train_black_dataset = pd.read_csv(TRAIN_BLACK_OPCODE_3_GRAM_PATH)
    test_0_3000_dataset = pd.read_csv(TEST_0_3000_OPCODE_3_GRAM_PATH)
    test_3000_6000_dataset = pd.read_csv(TEST_3000_6000_OPCODE_3_GRAM_PATH)
    test_6000_dataset = pd.read_csv(TEST_6000_OPCODE_3_GRAM_PATH)
    logger.info([test_0_3000_dataset.shape, test_3000_6000_dataset.shape, test_6000_dataset.shape])
    logger.info('train white dataset shape: ({}, {}).'.format(train_white_dataset.shape[0], train_white_dataset.shape[1]))
    logger.info('train black dataset shape: ({}, {}).'.format(train_black_dataset.shape[0], train_black_dataset.shape[1]))
    
    # 添加标签
    train_black_dataset['Label'] = 0
    train_white_dataset['Label'] = 1
    test_0_3000_dataset['Label'] = -1
    test_3000_6000_dataset['Label'] = -1
    test_6000_dataset['Label'] = -1
    
    # 所有样本合并
    dataset = pd.concat([train_black_dataset, train_white_dataset, test_0_3000_dataset, test_3000_6000_dataset, test_6000_dataset], ignore_index=True, sort=False)
    logger.info('dataset shape: ({}, {}).'.format(dataset.shape[0], dataset.shape[1]))
    
    # 填充缺失值
    dataset = dataset.fillna(0)
    
    # 拆分训练集和测试集
    train_dataset = dataset.query('Label != -1')
    test_dataset  = dataset.query('Label == -1')
    test_dataset.drop(['Label'], axis=1, inplace=True)
    
    # 保存
    train_dataset.to_csv(TRAIN_OPCODE_3_GRAM_PATH, sep=',', encoding='utf-8', index=False)
    test_dataset.to_csv(TEST_OPCODE_3_GRAM_PATH_, sep=',', encoding='utf-8', index=False)

def get_test_dirty_dataset():
    """获取测试集脏数据集
    又是前期埋下的坑
    """
    
    test_dataset = pd.read_csv('./databak/'+TEST_DATASET_FILENAME)
    test_dirty_dataset  = test_dataset.query('ExceptionError > 0 & ExceptionError < 7')
    logger.info(test_dirty_dataset.shape)
    test_dirty_dataset.to_csv(DATASET_PATH+TEST_DIRTY_DATASET_FILENAME, sep=',', encoding='utf-8', index=False)

if __name__ == '__main__':
    logger.info('******** starting ********')
    start_time = time.time()

    main()
    #get_test_dirty_dataset()
    
    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))









