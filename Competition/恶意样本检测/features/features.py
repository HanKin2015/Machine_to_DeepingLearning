# -*- coding: utf-8 -*-
"""
文 件 名: features.py
文件描述: 训练模型
作    者: HeJian
创建日期: 2022.05.29
修改日期：2022.05.30
Copyright (c) 2022 HeJian. All rights reserved.
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
    
    uncorrelated_features = ['ExceptionError', 'TimeDateStamp', 'TS_obj', 'DayName',
        'HourBin', 'Hour', 'Minute', 'Second', 'e_res', 'e_res2']
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

def features_processing(dataset):
    # 缺失值处理
    #dataset = missing_value_processing(dataset)
    # 异常值处理
    #dataset = exception_value_processing(dataset)
    # 日期时间处理
    datetime_processing(dataset)
    # 离散值处理
    # 删除不相关的特征（相关性低）
    dataset = delete_uncorrelated_features(dataset)
    
    # 特殊处理
    
    return dataset

def read_csv(filename):
    """从csv文件读取数据集
    """
    
    path = DATA_PATH + filename
    dataset = pd.read_csv(path)
    return dataset

def save_csv(dataset, filename):
    """数据集保存到csv文件
    """
    
    path = DATASET_PATH + filename
    dataset.to_csv(path, sep=',', encoding='utf-8', index=False)

def extended_custom_features(dataset, extended_features_path):
    """扩展的特征(自定义的字符串特征)
    """
    
    names = ['FileName', 'System32', 'System64', 'Http', 'Https', 'Download', 'Heky', 'Wget', 'Curl', 'SystemRoot', 'Windir', 'Root',
        'Load', 'Get', 'Temp', 'Mutex', 'Thread', 'Service']
    df = pd.read_csv(DATA_PATH+extended_features_path, names=names)
    dataset = pd.merge(dataset, df, how='inner', on='FileName')
    return dataset

def extended_features(dataset, sample_path, extended_features_path):
    """扩展的特征
    """
    
    file_names = os.listdir(sample_path)
    logger.info('{} file count: {}.'.format(sample_path, len(file_names)))
    
    df = pd.read_csv(DATA_PATH+extended_features_path, names=range(10000))
    df['FileName'] = file_names
    dataset = pd.merge(dataset, df, how='inner', on='FileName')
    return dataset

def test_dirty_string_features():
    """测试集脏数据字符串特征
    """
    
    test_string_features_dataset = pd.read_csv(TEST_STRING_FEATURES_PATH, names=range(10000))
    file_names = os.listdir(TEST_PATH)
    test_string_features_dataset['FileName'] = file_names
    dirty_file_names = os.listdir('./dirty_files/')
    
    test_dirty_string_features_dataset = test_string_features_dataset[test_string_features_dataset.FileName.isin(dirty_file_names)]
    test_dirty_string_features_dataset.to_csv(DATASET_PATH+TEST_DIRTY_DATASET_FILENAME, sep=',', encoding='utf-8', index=False)

def main():
    # 获取数据集
    train_black_dataset = read_csv(TRAIN_BLACK_DATASET_FILENAME)
    train_white_dataset = read_csv(TRAIN_WHITE_DATASET_FILENAME)
    test_dataset        = read_csv(TEST_DATASET_FILENAME)
    logger.info([train_black_dataset.shape, train_white_dataset.shape, test_dataset.shape])
    
    # 扩展的特征
    #train_black_dataset = extended_features(train_black_dataset, TRAIN_BLACK_PATH, TRAIN_BLACK_STRING_FEATURES_PATH)
    #train_white_dataset = extended_features(train_white_dataset, TRAIN_WHITE_PATH, TRAIN_WHITE_STRING_FEATURES_PATH)
    #test_dataset        = extended_features(test_dataset, TEST_PATH, TEST_STRING_FEATURES_PATH)
    train_black_dataset = extended_custom_features(train_black_dataset, TRAIN_BLACK_CUSTOM_STRINGS_PATH)
    train_white_dataset = extended_custom_features(train_white_dataset, TRAIN_WHITE_CUSTOM_STRINGS_PATH)
    test_dataset        = extended_custom_features(test_dataset, TEST_CUSTOM_STRINGS_PATH)
    logger.info([train_black_dataset.shape, train_white_dataset.shape, test_dataset.shape])
    
    # 添加标签
    train_black_dataset['Label'] = 0
    train_white_dataset['Label'] = 1
    
    # 黑白样本合并
    train_dataset = pd.concat([train_black_dataset, train_white_dataset], ignore_index=True)
    
    # 去除脏数据
    train_dirty_dataset = train_dataset[train_dataset['ExceptionError'] > 0]
    test_dirty_dataset  = test_dataset[test_dataset['ExceptionError'] > 0]
    save_csv(train_dirty_dataset, TRAIN_DIRTY_DATASET_FILENAME)
    save_csv(test_dirty_dataset, TEST_DIRTY_DATASET_FILENAME)
    train_dataset = train_dataset[train_dataset['ExceptionError'] == 0]
    test_dataset  = test_dataset[test_dataset['ExceptionError'] == 0]
    logger.info('train_dirty_dataset: ({}, {}), test_dirty_dataset: ({}, {}).'.format(
        train_dirty_dataset.shape[0], train_dirty_dataset.shape[1],
        test_dirty_dataset.shape[0], test_dirty_dataset.shape[1],))
    
    # 特征工程
    train_dataset = features_processing(train_dataset)
    test_dataset  = features_processing(test_dataset)
    logger.info('train_dataset: ({}, {}), test_dataset: ({}, {}).'.format(
        train_dataset.shape[0], train_dataset.shape[1],
        test_dataset.shape[0], test_dataset.shape[1],))
    
    # 将标签移动到最后一列
    label = train_dataset['Label']
    train_dataset.drop(['Label'], axis=1, inplace=True)
    train_dataset = pd.concat([train_dataset, label], axis=1)
    
    # 保存数据集
    save_csv(train_dataset, TRAIN_DATASET_FILENAME)
    save_csv(test_dataset, TEST_DATASET_FILENAME)

if __name__ == '__main__':
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))