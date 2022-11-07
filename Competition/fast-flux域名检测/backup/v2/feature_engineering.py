# -*- coding: utf-8 -*-
"""
文 件 名: feature_engineering.py
文件描述: 特征工程
作    者: HanKin
创建日期: 2022.10.18
修改日期：2022.10.18

Copyright (c) 2022 HanKin. All rights reserved.
"""

from common import *

def exception_value_processing_by_delete(dataset, feature, lower_threshold, upper_threshold):
    """异常值处理（删除）
    """
    
    dataset = dataset[(lower_threshold <= dataset[feature]) & (dataset[feature] <= upper_threshold)]
    return dataset

def exception_value_processing_by_median(dataset, feature, lower_threshold, upper_threshold):
    """异常值处理（取中位数）
    """
    
    df = dataset[(lower_threshold <= dataset[feature]) & (dataset[feature] <= upper_threshold)]
    logger.debug('{}<{},{}>: {}/{}.'.format(feature, lower_threshold, upper_threshold, df.shape[0], dataset.shape[0]))
    dataset.loc[dataset[feature] < lower_threshold, feature] = df[feature].median()
    dataset.loc[dataset[feature] > upper_threshold, feature] = df[feature].median()
    return dataset
    
def exception_value_processing_by_mean(dataset, feature, lower_threshold, upper_threshold):
    """异常值处理（取平均值）
    """
    
    df = dataset[(lower_threshold <= dataset[feature]) & (dataset[feature] <= upper_threshold)]
    logger.debug('{}<{},{}>: {}/{}.'.format(feature, lower_threshold, upper_threshold, df.shape[0], dataset.shape[0]))
    dataset.loc[dataset[feature] < lower_threshold, feature] = int(df[feature].mean())
    dataset.loc[dataset[feature] > upper_threshold, feature] = int(df[feature].mean())
    return dataset

def missing_value_processing(dataset):
    """缺失值处理
    """
    
    # 用前一个值填充
    dataset = dataset.fillna(method='ffill')
    return dataset

def exception_value_processing(dataset):
    exception_values = [
        ['SizeOfStackReserve', 0, 2e7],
        ['ExportRVA', 0, 2e7],
        ['DebugRVA', 0, 1e7],
        ['IATRVA', 0, 1e7],
        ]

    for exception_value in exception_values:
        feature, lower_threshold, upper_threshold = [elem for elem in exception_value]
        dataset = exception_value_processing_by_mean(dataset, feature, lower_threshold, upper_threshold)
    
    if 'label' in dataset.columns:
        dataset = exception_value_processing_by_delete(dataset, 'SizeOfImage', 1e4, 5e9)
    return dataset

def delete_uncorrelated_features(dataset):
    """删除相关性低的特征
    """
    
    uncorrelated_features = ['time_first', 'time_last', 'rrtype', 'rdata', 'bailiwick', 'rrname_label', 'bailiwick_label']
    dataset.drop(uncorrelated_features, axis=1, inplace=True)
    return dataset

def datetime_processing(dataset):
    """日期时间处理
    """
    
    dataset['TimeDateStamp'] = dataset['TimeDateStamp'].apply(lambda x: time.strftime("%Y-%m-%d %X", time.localtime(x)))
    ts_objs = np.array([pd.Timestamp(item) for item in np.array(dataset['TimeDateStamp'])])
    dataset['TS_obj'] = ts_objs

    # 日期处理（DayName需要转换成数值特征）
    dataset['Year']       = dataset['TS_obj'].apply(lambda x: x.year)
    dataset['Month']      = dataset['TS_obj'].apply(lambda x: x.month)
    dataset['Day']        = dataset['TS_obj'].apply(lambda x: x.day)
    dataset['DayOfWeek']  = dataset['TS_obj'].apply(lambda x: x.dayofweek)
    dataset['DayName']    = dataset['TS_obj'].apply(lambda x: x.day_name())
    dataset['DayOfYear']  = dataset['TS_obj'].apply(lambda x: x.dayofyear)
    dataset['WeekOfYear'] = dataset['TS_obj'].apply(lambda x: x.weekofyear)
    dataset['Quarter']    = dataset['TS_obj'].apply(lambda x: x.quarter)
    day_name_map = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5,
        'Saturday': 6, 'Sunday': 7}
    dataset['DayNameBinMap'] = dataset['DayName'].map(day_name_map)

    # 时间处理
    dataset['Hour']       = dataset['TS_obj'].apply(lambda x: x.hour)
    dataset['Minute']     = dataset['TS_obj'].apply(lambda x: x.minute)
    dataset['Second']     = dataset['TS_obj'].apply(lambda x: x.second)
    #dataset['MUsecond']   = dataset['TS_obj'].apply(lambda x: x.microsecond)
    #dataset['UTC_offset'] = dataset['TS_obj'].apply(lambda x: x.utcoffset())

    ## 按照早晚切分时间
    hour_bins = [-1, 5, 11, 16, 21, 23]
    bin_names = ['LateNight', 'Morning', 'Afternoon', 'Evening', 'Night']
    dataset['HourBin'] = pd.cut(dataset['Hour'], bins=hour_bins, labels=bin_names)
    hour_bin_dummy_features = pd.get_dummies(dataset['HourBin'])
    dataset = pd.concat([dataset, hour_bin_dummy_features], axis=1)
    
    return dataset

def discrete_value_processing(dataset):
    """
    """
    
    gle = LabelEncoder()
    
    rrname_label = gle.fit_transform(dataset['rrname'])
    rrname_mapping = {index: label for index, label in enumerate(gle.classes_)}
    #logger.info(rrname_mapping)
    dataset['rrname_label'] = rrname_label
    
    bailiwick_label = gle.fit_transform(dataset['bailiwick'])
    bailiwick_mapping = {index: label for index, label in enumerate(gle.classes_)}
    #logger.info(bailiwick_mapping)
    dataset['bailiwick_label'] = bailiwick_label
    
    dataset['rdata_count'] = dataset['rdata'].apply(lambda x: len(x.split(',')))
    dataset['rrname_bailiwick'] = (dataset['rrname'] == dataset['bailiwick'])
    dataset['rrname_bailiwick'] = dataset['rrname_bailiwick'].astype(int)
    return dataset

def time_processing(dataset):
    """观察时间处理
    """
    
    dataset['time_interval'] = dataset['time_last'] - dataset['time_first']
    dataset['time_interval'] = dataset['time_interval'].apply(lambda x: int(x / 86400))
    return dataset

def rdata__processing(dataset):
    """
    """

def features_processing(dataset, ip_info):
    # 缺失值处理
    #dataset = missing_value_processing(dataset)
    # 异常值处理
    #dataset = exception_value_processing(dataset)
    # 日期时间处理
    #datetime_processing(dataset)
    #time_processing(dataset)
    # 离散值处理
    #dataset = discrete_value_processing(dataset)
    # 通过ip地址获取洲和国家的数量
    dataset = get_ip_info(dataset, ip_info)
    # 删除不相关的特征（相关性低）
    #dataset = delete_uncorrelated_features(dataset)
    
    # 特殊处理
    
    return dataset

def get_ip_info(dataset, ip_info):
    """获取ip地址的洲和国家的数量
    优化尝试：eval单次转换慢吗，并不是，也不是for循环，而是dataframe查找慢
    试一下字典
    """
    
    continent_label_dict = dict(zip(ip_info['ip'], ip_info['continent_label']))
    country_label_dict   = dict(zip(ip_info['ip'], ip_info['country_label']))
    
    result = pd.DataFrame(columns=['domain', 'continent_count', 'country_count', 'time_slot_count', 'd_value'])
    logger.info(dataset.columns)
    #for index, row in tqdm(dataset.iterrows(), ncols=100): # 这种方式的进度条效果不佳
    for row_id in tqdm(range(dataset.shape[0]), ncols=100):
        continent_list = []
        country_list = []
        for ip in eval(dataset.iloc[row_id]['ip_list']):
            continent_list.append(continent_label_dict.get(ip))
            country_list.append(country_label_dict.get(ip))
            #continent_list.append(ip_info.query('ip == @ip')['continent_label'].values[0])
            #country_list.append(ip_info.query('ip == @ip')['country_label'].values[0])
        continent_list = list(set(continent_list))
        country_list = list(set(country_list))
        
        ip_count_list = eval(dataset.iloc[row_id]['ip_count_list'])
        
        row = {'domain': dataset.iloc[row_id]['domain'], 'continent_count': len(continent_list),
            'country_count': len(country_list), 'time_slot_count': len(ip_count_list),
            'd_value': max(ip_count_list)-min(ip_count_list)}
        result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)
    return result

def ip_info_processing():
    """将ip地址的洲和国家LabelEncoder
    """
    
    ip_info = pd.read_csv(IP_INFO_PATH)
    logger.info('ip_info shape <{}, {}>'.format(ip_info.shape[0], ip_info.shape[1]))
    logger.info(ip_info.columns)
    
    # TypeError: argument must be a string or number
    ip_info = ip_info.fillna('None')
    
    gle = LabelEncoder()
    
    continent_label = gle.fit_transform(ip_info['continent'])
    continent_mapping = {index: label for index, label in enumerate(gle.classes_)}
    ip_info['continent_label'] = continent_label
    
    country_label = gle.fit_transform(ip_info['country'])
    country_mapping = {index: label for index, label in enumerate(gle.classes_)}
    ip_info['country_label'] = country_label

    return ip_info

def main():
    # 获取数据集
    train_dataset = pd.read_csv(TRAIN_DATASET_PATH)
    train_label   = pd.read_csv(TRAIN_LABEL_PATH)
    test_dataset  = pd.read_csv(TEST_DATASET_PATH)
    logger.info([train_dataset.shape, train_label.shape, test_dataset.shape])
    
    # 处理ip_info数据集
    ip_info = ip_info_processing()
    logger.info('ip_info shape <{}, {}>'.format(ip_info.shape[0], ip_info.shape[1]))
    
    # 特征工程
    train_dataset = features_processing(train_dataset, ip_info)
    test_dataset  = features_processing(test_dataset, ip_info)
    logger.info('train_dataset: ({}, {}), test_dataset: ({}, {}).'.format(
        train_dataset.shape[0], train_dataset.shape[1],
        test_dataset.shape[0], test_dataset.shape[1],))
    
    # 添加标签
    train_dataset = train_dataset.merge(train_label, on='domain', how='left')
    logger.info([train_dataset.shape])
    
    # 将标签移动到最后一列
    label = train_dataset['label']
    train_dataset.drop(['label'], axis=1, inplace=True)
    train_dataset = pd.concat([train_dataset, label], axis=1)
    
    # 保存数据集
    logger.info([train_dataset.shape, test_dataset.shape])
    logger.info(train_dataset.columns)
    train_dataset.to_csv(FEATURE_TRAIN_DATASET_PATH, sep=',', encoding='utf-8', index=False)
    test_dataset.to_csv(FEATURE_TEST_DATASET_PATH, sep=',', encoding='utf-8', index=False)

if __name__ == '__main__':
    #os.system('chcp 936 & cls')
    logger.info('******** starting ********')
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.\n'.format(round(end_time - start_time, 3)))

