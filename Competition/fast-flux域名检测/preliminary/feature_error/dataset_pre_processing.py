# -*- coding: utf-8 -*-
"""
文 件 名: dataset_pre_processing.py
文件描述: 数据预处理，保存文件是否DataFrame内存减少消耗
作    者: 重在参与快乐加倍队
创建日期: 2022.11.23
修改日期：2022.11.23

Copyright (c) 2022 ParticipationDoubled. All rights reserved.
"""

from common import *

def ip_info_processing():
    """处理ip_info数据集
    """
    
    ip_info = pd.read_csv(RAW_IP_INFO_PATH)
    logger.info('ip_info shape <{}, {}>'.format(ip_info.shape[0], ip_info.shape[1]))
    logger.info(ip_info.columns)
    
    if False:
        ip_info['networkTags'] = ['["CSDN"]' for i in range(ip_info.shape[0])]
        ip_info['judgement'] = ['unknown' for i in range(ip_info.shape[0])]
        ip_info['threatTags'] = ['[]' for i in range(ip_info.shape[0])]
        ip_info['expired'] = ['False' for i in range(ip_info.shape[0])]
        ip_info['firstFoundTime'] = ['2021-01-21T00:00:00' for i in range(ip_info.shape[0])]
        ip_info['updateTime'] = ['2021-01-21T00:00:00' for i in range(ip_info.shape[0])]
        ip_info['openSource'] = ['[]' for i in range(ip_info.shape[0])]
    
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

    info_cols = ['judgement','networkTags','threatTags','expired','continent','country',
            'countryCode','province','city','district','timeZone','organization','operator']

    # 填充缺失值
    for col in info_cols:
        ip_info[col] = ip_info[col].fillna('##')
        ip_info[col] = label_encode(ip_info[col].astype(str))
        ip_info[col] = ip_info[col].astype(str)

    ip_info['openSource'] = ip_info['openSource'].apply(lambda x: 1 if len(str(x)) > 2 else 0)
    ip_info['firstFoundTime'] = ip_info['firstFoundTime'].fillna('1970-01-01T08:00:00')
    ip_info['firstFoundTime'] = ip_info['firstFoundTime'].apply(lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%dT%H:%M:%S"))))
    ip_info['updateTime'] = ip_info['updateTime'].fillna('1970-01-01T08:00:00')
    ip_info['updateTime'] = ip_info['updateTime'].apply(lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%dT%H:%M:%S"))))
    ip_info['updateTime_diff'] = ip_info['updateTime'] - ip_info['firstFoundTime']

    ip_info.to_csv(IP_INFO_PATH, sep=',', encoding='utf-8', index=False)

def dataset_pre_processing(dataset):
    """数据集预处理
    """
    
    # 去掉列表的左右中括号
    dataset['rdata'] = dataset['rdata'].apply(lambda x:x[1:-1])

    # 去掉空格
    dataset['rdata'] = dataset['rdata'].apply(lambda x:x.replace(' ', ''))

    # 时间转换成天数
    #dataset['time_first'] = dataset['time_first'].apply(lambda x: int(x / 3600 / 24))
    #dataset['time_last'] = dataset['time_last'].apply(lambda x: int(x / 3600 / 24))

    # 时间差
    dataset['time_diff'] = dataset['time_last'] - dataset['time_first']
    
    dataset['ttl'] = dataset['time_diff'] / dataset['count']
    return dataset

def label_encode(series):
    """自然数编码
    """
    
    unique = list(series.unique())
    #unique.sort()
    return series.map(dict(zip(unique, range(series.nunique()))))

def feature_extraction(dataset):
    """特征提取
    """

    # 数据集预处理
    dataset = dataset_pre_processing(dataset)

    # 聚合
    agg_df = dataset.groupby(['rrname']).agg({'rdata':[list],
                                            'count':['sum', 'max', 'min', 'std', 'mean'],
                                            'time_first':['max', 'min'],
                                            'time_last':['max', 'min'],
                                            'rrtype':['unique'],
                                            'bailiwick':[list, 'nunique'],
                                            'ttl':['max', 'min', 'std', 'mean'],
                                            'time_diff':['sum', 'max', 'min', 'std', 'mean']}).reset_index()

    # 重置列名
    agg_df.columns = [''.join(col).strip() for col in agg_df.columns.values]
    agg_df.reset_index(drop=True, inplace=True)
    
    # 去掉两边引号
    agg_df['rdatalist'] = agg_df['rdatalist'].apply(lambda x:','.join([i[1:-1] for i in x]).replace('\'','').split(','))

    agg_df['rdatalist_count']     = agg_df['rdatalist'].apply(lambda x:len(x))
    agg_df['rdatalist_nunique']   = agg_df['rdatalist'].apply(lambda x:len(set(x)))
    agg_df['bailiwicklist_count'] = agg_df['bailiwicklist'].apply(lambda x:len(x))
    
    # 自然数编码
    for col in ['rrtypeunique']:
        agg_df[col] = label_encode(agg_df[col].astype(str))
    
    # 域名长度
    agg_df['rrname_length'] = agg_df['rrname'].apply(lambda x: len(x))
    
    # 域名中数字数量
    agg_df['rrname_number_count'] = agg_df['rrname'].apply(lambda x: len([ch for ch in x if ch.isdigit()]))
    
    # 域名中点数量
    agg_df['rrname_dot_count'] = agg_df['rrname'].apply(lambda x: x.count('.'))
    
    agg_df['rdatalist_count_rdatalist_nunique'] = agg_df['rdatalist_count'] / agg_df['rdatalist_nunique']
    agg_df['rdatalist_count_bailiwicklist_count'] = agg_df['rdatalist_count'] / agg_df['bailiwicklist_count']
    return agg_df
    
def main():
    """主函数
    """

    # 处理ip_info数据集
    if not os.path.exists(IP_INFO_PATH):
        ip_info_processing()

    show_memory_info('initial')
    # 获取数据集
    train_dataset = pd.read_csv(TRAIN_DATASET_PATH)
    test_dataset  = pd.read_csv(TEST_DATASET_PATH)    
    logger.info('train_dataset shape <{}, {}>'.format(train_dataset.shape[0], train_dataset.shape[1]))
    logger.info('test_dataset  shape <{}, {}>'.format(test_dataset.shape[0], test_dataset.shape[1]))

    show_memory_info('feature_extraction before')
    # 特征提取
    agg_train_dataset = feature_extraction(train_dataset)
    del train_dataset
    gc.collect()
    show_memory_info('train_dataset feature_extraction after')
    test_dataset = feature_extraction(test_dataset)
    del test_dataset
    gc.collect()
    show_memory_info('test_dataset feature_extraction after')

    # 合并训练集和测试集
    dataset = pd.concat([agg_train_dataset, agg_test_dataset], axis=0, ignore_index=True)
    show_memory_info('gc train_dataset test_dataset before')
    del agg_train_dataset, agg_test_dataset
    gc.collect()
    show_memory_info('gc train_dataset test_dataset after')

    # 保存数据集
    dataset.to_csv(PRE_DATASET_PATH, sep=',', encoding='utf-8', index=False)

if __name__ == '__main__':
    """程序入口
    """
    
    #os.system('chcp 936 & cls')
    logger.info('******** starting ********')
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.\n'.format(round(end_time - start_time, 3)))