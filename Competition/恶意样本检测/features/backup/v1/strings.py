import subprocess
import os
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
import time
from log import logger
import pandas as pd

SAMPLE_PATH      = './test/'    #AIFirst_data
TRAIN_WHITE_PATH = SAMPLE_PATH+'train/white/' # 训练集白样本路径
TRAIN_BLACK_PATH = SAMPLE_PATH+'train/black/' # 训练集黑样本路径
TEST_PATH        = SAMPLE_PATH+'test/'        # 测试集样本路径
DATA_PATH        = './data/'                  # 数据路径
TRAIN_WHITE_STRING_FEATURES_PATH = DATA_PATH+'train_white_string_features.csv' # 训练集白样本数据集路径
TRAIN_BLACK_STRING_FEATURES_PATH = DATA_PATH+'train_black_string_features.csv' # 训练集黑样本数据集路径
TEST_STRING_FEATURES_PATH        = DATA_PATH+'test_string_features.csv'        # 测试集样本数据集路径

TRAIN_WHITE_CUSTOM_STRINGS_PATH = DATA_PATH+'train_white_strings.csv' # 训练集白样本数据集路径
TRAIN_BLACK_CUSTOM_STRINGS_PATH = DATA_PATH+'train_black_strings.csv' # 训练集黑样本数据集路径
TEST_CUSTOM_STRINGS_PATH        = DATA_PATH+'test_strings.csv'        # 测试集样本数据集路径

# 线程数量
THREAD_NUM = 64

# 创建数据文件夹
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

def extract_strings(filepath):
    '''This methods extracts the strings from a file using the strings command in unix os'''
    strings = subprocess.Popen(['strings', filepath], stdout=subprocess.PIPE).communicate()[0].decode('utf-8').split('\n')
    return strings
    
def get_string_features(all_strings):
    data_features = []
    hasher = FeatureHasher(20000) # We initialize the featurehasher using 20,000 features
    for all_string in all_strings:
        # store string features in dictionary form
        string_features = {}
        for string in all_string:
            string_features[string] = 1

        # hash the features using the hashing trick
        hashed_features = hasher.transform([string_features])
        # do some data munging to get the feature array
        hashed_features = hashed_features.todense()
        hashed_features = numpy.asarray(hashed_features)
        hashed_features = hashed_features[0]
        data_features.extend([hashed_features])
    return data_features

def training(train_black_path, train_white_path):
    train_black_dataset = pd.read_csv(train_black_path, header=None)
    train_white_dataset = pd.read_csv(train_white_path, header=None)
    
    train_dataset = pd.concat([train_black_dataset, train_white_dataset], ignore_index=True)
    logger.info('train_dataset shape: ({}, {}).'.format(train_dataset.shape[0], train_dataset.shape[1]))

    X = train_dataset.values
    y = [0 for _ in range(train_black_dataset.shape[0])] + [1 for _ in range(train_white_dataset.shape[0])]
    logger.info('X shape: ({}, {}).'.format(len(X), len(X[0])))
    logger.info('y len: {}.'.format(len(y)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2022)

    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=2022)
    clf.fit(X_train, y_train)
    logger.info('Random Forest Classifier on hold-out (70% Train, 30% Test): {}.'.format(clf.score(X_test, y_test)))
    logger.info([x for x in clf.feature_importances_ if x > 0.1])

def training_(train_black_path, train_white_path):
    train_black_dataset = pd.read_csv(train_black_path, header=None)
    train_white_dataset = pd.read_csv(train_white_path, header=None)
    
    train_dataset = pd.concat([train_black_dataset, train_white_dataset], ignore_index=True)
    logger.info('train_dataset shape: ({}, {}).'.format(train_dataset.shape[0], train_dataset.shape[1]))

    X = train_dataset.loc[:, 1::].values
    y = [0 for _ in range(train_black_dataset.shape[0])] + [1 for _ in range(train_white_dataset.shape[0])]
    logger.info('X shape: ({}, {}).'.format(len(X), len(X[0])))
    logger.info('y len: {}.'.format(len(y)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2022)

    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=2022)
    clf.fit(X_train, y_train)
    logger.info('Random Forest Classifier on hold-out (70% Train, 30% Test): {}.'.format(clf.score(X_test, y_test)))
    logger.info([x for x in clf.feature_importances_ if x > 0.1])

def save2csv(data, csv_path):
    """数据保存到本地csv文件中
    """
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, sep=',', encoding='utf-8', index=False)

def string_features_processing(sample_path, save_path):
    """字符串特征处理（获取）
    """
    
    file_names = os.listdir(sample_path)
    logger.info('{} file count: {}.'.format(sample_path, len(file_names)))
    
    all_strings = [extract_strings(sample_path + file_names[i]) for i in range(len(file_names))]
    logger.info('all_strings count: {}.'.format(len(all_strings)))
    
    string_features = get_string_features(all_strings)
    logger.info('string_features count: {}.'.format(len(string_features)))
    
    save2csv(string_features, save_path)
    return string_features

def main():
    train_white_features = string_features_processing(TRAIN_WHITE_PATH, TRAIN_WHITE_STRING_FEATURES_PATH)
    train_black_features = string_features_processing(TRAIN_BLACK_PATH, TRAIN_BLACK_STRING_FEATURES_PATH)
    test_features = string_features_processing(TEST_PATH, TEST_STRING_FEATURES_PATH)

    training(TRAIN_BLACK_STRING_FEATURES_PATH, TRAIN_WHITE_STRING_FEATURES_PATH)
    training_(TRAIN_BLACK_CUSTOM_STRINGS_PATH, TRAIN_WHITE_CUSTOM_STRINGS_PATH)

if __name__ == '__main__':
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))

