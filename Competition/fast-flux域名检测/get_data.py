# -*- coding: utf-8 -*-
"""
文 件 名: get_data.py
文件描述: 获取数据，原始数据一条记录存在多行，需要将多行合并成一行
作    者: HanKin
创建日期: 2022.10.28
修改日期：2022.10.28

Copyright (c) 2022 HanKin. All rights reserved.
"""

from common import *

def rdata_processing(dataset):
    """
    """
    
    result = pd.DataFrame(columns=['domain', 'ip_list', 'ip_count_list'])
    
    for rrname in tqdm(set(dataset['rrname']), ncols=100):
        df = dataset[dataset['rrname'] == rrname]
        
        ip_count_list = []
        ip_list = []
        for index, row in df.iterrows():
            tmp = eval(row['rdata'])
            ip_count_list.append(len(tmp))
            ip_list = list(set(ip_list).union(set(tmp)))
        
        row = {'domain': rrname, 'ip_list': ip_list, 'ip_count_list': ip_count_list}
        result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)

    logger.info('predict result shape: {}'.format(result.shape))
    return result

def delete_uncorrelated_features(dataset):
    """删除相关性低的特征
    """
    
    uncorrelated_features = ['judgement', 'networkTags', 'threatTags', 'firstFoundTime', 'updateTime',
        'expired', 'openSource', 'location', 'asn', 'samples', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    dataset.drop(uncorrelated_features, axis=1, inplace=True)
    return dataset

def features_processing(dataset):
    # location处理
    dataset = rdata_processing(dataset)
    # 删除不相关的特征（相关性低）
    #dataset = delete_uncorrelated_features(dataset)
    
    # 特殊处理
    
    return dataset

def main():
    # 获取数据集
    train_dataset = pd.read_csv(RAW_TRAIN_DATASET_PATH)
    test_dataset  = pd.read_csv(RAW_TEST_DATASET_PATH)
    logger.info([train_dataset.shape, test_dataset.shape])
    
    # 特征工程
    train_dataset = features_processing(train_dataset)
    test_dataset  = features_processing(test_dataset)
    logger.info('train_dataset: ({}, {}), test_dataset: ({}, {}).'.format(
        train_dataset.shape[0], train_dataset.shape[1],
        test_dataset.shape[0], test_dataset.shape[1],))
    
    # 保存数据集
    logger.info([train_dataset.shape, test_dataset.shape])
    logger.info(train_dataset.columns)
    train_dataset.to_csv(TRAIN_DATASET_PATH, sep=',', encoding='utf-8', index=False)
    test_dataset.to_csv(TEST_DATASET_PATH, sep=',', encoding='utf-8', index=False)

if __name__ == '__main__':
    #os.system('chcp 936 & cls')
    logger.info('******** starting ********')
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.\n'.format(round(end_time - start_time, 3)))