# -*- coding: utf-8 -*-
"""
文 件 名: deal_ip_info.py
文件描述: 处理ip信息
作    者: HanKin
创建日期: 2022.10.28
修改日期：2022.10.28

Copyright (c) 2022 HanKin. All rights reserved.
"""

from common import *

def location_processing(ip_info):
    """location处理，获取洲名、国家、时区
    """
    
    ip_info['location'] = ip_info['location'].apply(lambda x: eval(x))
    ip_info['continent'] = ip_info['location'].apply(lambda x: x['continent'])
    ip_info['country'] = ip_info['location'].apply(lambda x: x['country'])
    ip_info['timeZone'] = ip_info['location'].apply(lambda x: x['timeZone'])
    return ip_info

def delete_uncorrelated_features(dataset):
    """删除相关性低的特征
    """
    
    uncorrelated_features = ['judgement', 'networkTags', 'threatTags', 'firstFoundTime', 'updateTime',
        'expired', 'openSource', 'location', 'asn', 'samples', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    dataset.drop(uncorrelated_features, axis=1, inplace=True)
    return dataset

def features_processing(ip_info):
    # location处理
    ip_info = location_processing(ip_info)
    # 删除不相关的特征（相关性低）
    ip_info = delete_uncorrelated_features(ip_info)
    
    # 特殊处理
    
    return ip_info

def main():
    # 获取数据集
    ip_info = pd.read_csv(RAW_IP_INFO_PATH)
    logger.info([ip_info.shape])
    
    # 特征工程
    ip_info = features_processing(ip_info)
    logger.info([ip_info.shape])
    logger.info(ip_info.columns)
    
    # 保存数据集
    ip_info.to_csv(IP_INFO_PATH, sep=',', encoding='utf-8', index=False)

if __name__ == '__main__':
    #os.system('chcp 936 & cls')
    logger.info('******** starting ********')
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.\n'.format(round(end_time - start_time, 3)))