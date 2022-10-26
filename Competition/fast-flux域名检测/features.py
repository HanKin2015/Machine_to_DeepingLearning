# -*- coding: utf-8 -*-
"""
文 件 名: features.py
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
    
    uncorrelated_features = ['time_first', 'time_last', 'rrtype', 'rdata',
        'rrname', 'bailiwick']
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
    
    
    return dataset

def features_processing(dataset):
    # 缺失值处理
    #dataset = missing_value_processing(dataset)
    # 异常值处理
    #dataset = exception_value_processing(dataset)
    # 日期时间处理
    #datetime_processing(dataset)
    # 离散值处理
    dataset = discrete_value_processing(dataset)
    
    # 删除不相关的特征（相关性低）
    dataset = delete_uncorrelated_features(dataset)
    
    # 特殊处理
    
    return dataset

def extended_custom_features(dataset, extended_features_path):
    """扩展的特征(自定义的字符串特征)
    """
    
    
    return dataset

def extended_features(dataset, sample_path, extended_features_path):
    """扩展的特征
    """
    
    
    return dataset

def main():
    # 获取数据集
    train_dataset = pd.read_csv(TRAIN_RAW_DATASET_PATH)
    train_label   = pd.read_csv(TRAIN_LABELS_PATH)
    test_dataset  = pd.read_csv(TEST_RAW_DATASET_PATH)
    logger.info([train_dataset.shape, train_label.shape, test_dataset.shape])

    # 添加标签
    train_label.rename(columns = {"domain": "rrname"},  inplace=True)
    train_dataset = train_dataset.merge(train_label, on='rrname', how='left')
    logger.info([train_dataset.shape])
    
    # 去除脏数据
    
    # 特征工程
    train_dataset = features_processing(train_dataset)
    test_dataset  = features_processing(test_dataset)
    logger.info('train_dataset: ({}, {}), test_dataset: ({}, {}).'.format(
        train_dataset.shape[0], train_dataset.shape[1],
        test_dataset.shape[0], test_dataset.shape[1],))
    logger.info(train_dataset.columns)
    
    # 将标签移动到最后一列
    label = train_dataset['label']
    train_dataset.drop(['label'], axis=1, inplace=True)
    train_dataset = pd.concat([train_dataset, label], axis=1)
    
    # 保存数据集
    train_dataset.to_csv(TRAIN_DATASET_PATH, sep=',', encoding='utf-8', index=False)
    test_dataset.to_csv(TEST_DATASET_PATH, sep=',', encoding='utf-8', index=False)

if __name__ == '__main__':
    #os.system('chcp 936 & cls')
    start_time = time.time()

    main()

    end_time = time.time()
    print('process spend {} s.'.format(round(end_time - start_time, 3)))




