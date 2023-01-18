# -*- coding: utf-8 -*-
"""
文 件 名: data_preprocessing.py
文件描述: 数据预处理
作    者: 重在参与快乐加倍队
创建日期: 2022.11.23
修改日期：2022.11.24

Copyright (c) 2022 ParticipationDoubled. All rights reserved.
"""

from common import *

def label_encode(series):
    """自然数编码
    """
    
    unique = list(series.unique())
    #unique.sort()
    return series.map(dict(zip(unique, range(series.nunique()))))
    
def ip_info_processing():
    """处理ip_info数据集
    """
    
    ip_info = pd.read_csv(RAW_IP_INFO_PATH)
    logger.info('raw ip_info shape <{}, {}>'.format(ip_info.shape[0], ip_info.shape[1]))
    logger.debug(ip_info.columns)
    
    ip_info['location'] = ip_info['location'].apply(lambda x: eval(x))
    for col in ['continent', 'country', 'countryCode', 'province', 'city', 'district', 'lng', 'lat', 'timeZone']:
        ip_info[col] = ip_info['location'].apply(lambda x: x[col])
     
    ip_info['asn'] = ip_info['asn'].apply(lambda x: eval(x))
    for col in ['number', 'organization', 'operator']:
        ip_info[col] = ip_info['asn'].apply(lambda x: x[col])

    ip_info['networkTags'] = ip_info['networkTags'].apply(lambda x:str(x).replace('[',''))
    ip_info['networkTags'] = ip_info['networkTags'].apply(lambda x:str(x).replace(']',''))
    ip_info['threatTags'] = ip_info['threatTags'].apply(lambda x:str(x).replace('[',''))
    ip_info['threatTags'] = ip_info['threatTags'].apply(lambda x:str(x).replace(']',''))

    info_cols = ['judgement','networkTags','threatTags','expired','openSource', 'samples','continent','country',
            'countryCode','province','city','district','timeZone','organization','operator']

    # 填充缺失值
    for col in info_cols:
        ip_info[col] = ip_info[col].fillna('##')
        ip_info[col] = label_encode(ip_info[col].astype(str))
    
    # 删除无用的特征
    ip_info.drop(['location', 'asn', 'firstFoundTime', 'updateTime'], axis=1, inplace=True)

    logger.info('ip_info shape <{}, {}>'.format(ip_info.shape[0], ip_info.shape[1]))
    ip_info.to_csv(IP_INFO_PATH, sep=',', encoding='utf-8', index=False)
    logger.info('IP_INFO_PATH is saved successfully')

def main():
    """主函数
    """

    # 处理ip_info数据集(True强制执行，False判断文件是否存在)
    if False or not os.path.exists(IP_INFO_PATH):
        ip_info_processing()
    logger.info('data preprocessing done')

if __name__ == '__main__':
    """程序入口
    """
    
    #os.system('chcp 936 & cls')
    logger.info('******** starting ********')
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.\n'.format(round(end_time - start_time, 3)))