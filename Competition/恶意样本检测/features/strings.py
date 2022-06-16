# -*- coding: utf-8 -*-
"""
文 件 名: training.py
文件描述: 训练模型
作    者: HeJian
创建日期: 2022.05.29
修改日期：2022.06.16
Copyright (c) 2022 HeJian. All rights reserved.
"""

from common import *

def extract_strings(filepath):
    strings = subprocess.Popen(['strings', filepath], stdout=subprocess.PIPE).communicate()[0].decode('utf-8').split('\n')
    return strings
    
def get_string_features(all_strings):
    data_features = []
    hasher = FeatureHasher(10000)
    for all_string in all_strings:
        string_features = {}
        for string in all_string:
            string_features[string] = 1
        hashed_features = hasher.transform([string_features])
        hashed_features = hashed_features.todense()
        hashed_features = np.asarray(hashed_features)
        hashed_features = hashed_features[0]
        data_features.extend([hashed_features])
    return data_features

def save_training_model(model, score):
    """保存训练模型
    """
    
    before_score = 0
    if os.path.exists(DIRTY_DATASET_MODEL_SCORE_PATH):
        with open(DIRTY_DATASET_MODEL_SCORE_PATH, 'r') as fd:
            before_score = fd.read()
    if score > float(before_score):
        buffer = pickle.dumps(model)
        with open(DIRTY_DATASET_MODEL_PATH, "wb+") as fd:
            fd.write(buffer)
        with open(DIRTY_DATASET_MODEL_SCORE_PATH, 'w') as fd:
            fd.write(str(score))

def training(train_black_path, train_white_path):
    """简单训练一下字符串数据集效果
    """
    
    train_black_dataset = pd.read_csv(train_black_path, header=None)
    train_white_dataset = pd.read_csv(train_white_path, header=None)
    
    train_dataset = pd.concat([train_black_dataset, train_white_dataset], ignore_index=True)
    logger.info('train_dataset shape: ({}, {}).'.format(train_dataset.shape[0], train_dataset.shape[1]))

    if type(train_dataset.iloc[0, 0]) is str:
        X = train_dataset.loc[:, 1::].values
    else:
        X = train_dataset.values
        
    y = [0 for _ in range(train_black_dataset.shape[0])] + [1 for _ in range(train_white_dataset.shape[0])]
    logger.info('X shape: ({}, {}).'.format(len(X), len(X[0])))
    logger.info('y len: {}.'.format(len(y)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2022)

    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=2022)
    clf.fit(X_train, y_train)
    
    score = clf.score(X_test, y_test)
    logger.info('Random Forest Classifier on hold-out (70% Train, 30% Test): {}.'.format(score*100))
    logger.info([x for x in clf.feature_importances_ if x > 0.1])
    
    save_training_model(clf, score)

def save2csv(data, sample_path, csv_path):
    """数据保存到本地csv文件中
    """

    file_names = os.listdir(sample_path)
    logger.info('{} file count: {}.'.format(sample_path, len(file_names)))

    df = pd.DataFrame(data)
    df['FileName'] = file_names
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
    
    save2csv(string_features, sample_path, save_path)
    return string_features

def main():
    #train_white_features = string_features_processing(TRAIN_WHITE_PATH, TRAIN_WHITE_STRING_FEATURES_PATH)
    #train_black_features = string_features_processing(TRAIN_BLACK_PATH, TRAIN_BLACK_STRING_FEATURES_PATH)
    #test_features        = string_features_processing(TEST_PATH, TEST_STRING_FEATURES_PATH)
    test_dirty_features   = string_features_processing('./dirty_files/', DATASET_PATH+TEST_DIRTY_DATASET_FILENAME)
    #training(TRAIN_BLACK_STRING_FEATURES_PATH, TRAIN_WHITE_STRING_FEATURES_PATH)

if __name__ == '__main__':
    start_time = time.time()

    #main()
    
    # 测试自定义指定字符串特征数据集
    #training(TRAIN_BLACK_CUSTOM_STRINGS_PATH, TRAIN_WHITE_CUSTOM_STRINGS_PATH)
    training(TRAIN_BLACK_STRING_FEATURES_PATH, TRAIN_WHITE_STRING_FEATURES_PATH)

    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))

