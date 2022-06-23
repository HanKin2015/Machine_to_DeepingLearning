# -*- coding: utf-8 -*-
"""
文 件 名: predict.py
文件描述: 加载模型进行预测
作    者: HeJian
创建日期: 2022.05.29
修改日期：2022.05.30
Copyright (c) 2022 HeJian. All rights reserved.
"""

from common import *

def load_model(model_path):
    """加载模型
    """

    with open(model_path, 'rb') as fd:
        buffer = fd.read()
    model = pickle.loads(buffer)
    return model

def load_model_(model_path):
    """加载模型
    """

    with open(model_path, 'rb') as fd:
        model = pickle.loads(fd)
    return model

def special_treatment(dataset):
    """特殊样本处理全部置为0
    """
    
    return np.ones(dataset.shape[0], dtype=int)

def predict_(test_dataset_path, test_dirty_dataset_path, model_path, dirty_model_path=None):
    """加载模型进行测试集预测
    包含异常样本特殊处理
    """

    # 获取数据集
    test_dataset = pd.read_csv(test_dataset_path)
    logger.info('test dataset shape: ({}, {}).'.format(test_dataset.shape[0], test_dataset.shape[1]))
    
    test_dirty_dataset = pd.read_csv(test_dirty_dataset_path)
    logger.info('test_dataset: ({}, {}), test_dirty_dataset: ({}, {}).'.format(
        test_dataset.shape[0], test_dataset.shape[1],
        test_dirty_dataset.shape[0], test_dirty_dataset.shape[1],))

    # 模型预测结果
    file_name1 = test_dataset['FileName']
    X = test_dataset.drop(['FileName'], axis=1, inplace=False).values
    #selector = load_model(MALICIOUS_SAMPLE_DETECTION_SELECTOR_PATH)
    #X = selector.transform(X)
    model = load_model(model_path)
    result1 = model.predict(X)
    
    # 异常样本特殊处理
    file_name2 = test_dirty_dataset['FileName']
    #X = test_dirty_dataset.drop(['FileName'], axis=1, inplace=False).values
    #selector = load_model(SELECTOR_PATH)
    #X = selector.transform(X)
    #model = load_model(DIRTY_DATASET_MODEL_PATH)
    #result2 = model.predict(X)
    result2 = special_treatment(test_dirty_dataset)
    
    # 存储结果
    file_name = file_name1.append(file_name2)
    result = np.append(result1, result2)
    df = pd.DataFrame({
                'md5'  : file_name,
                'label': result
           })
    logger.info('predict result shape: {}'.format(df.shape))
    df.to_csv(RESULT_PATH, index=False, header=False)

def predict(test_dataset_path, model_path):
    """加载模型进行测试集预测
    """

    # 获取数据集
    test_dataset = pd.read_csv(test_dataset_path)
    logger.info('test dataset shape: ({}, {}).'.format(test_dataset.shape[0], test_dataset.shape[1]))

    # 模型预测结果
    file_name = test_dataset['FileName']
    X = test_dataset.drop(['FileName'], axis=1, inplace=False).values
    #selector = load_model(MALICIOUS_SAMPLE_DETECTION_SELECTOR_PATH)
    #X = selector.transform(X)
    model = load_model(model_path)
    result = model.predict(X)
    
    # 存储结果
    df = pd.DataFrame({
                'md5'  : file_name,
                'label': result
           })
    logger.info('predict result shape: {}'.format(df.shape))
    df.to_csv(RESULT_PATH, index=False, header=False)

def main():

    #predict_()
    predict_(TEST_IMAGE_MATRIX_PATH, IAMGE_MATRIX_RFC_MODEL_SCORE_PATH)
    #predict_(TEST_OPCODE_3_GRAM_PATH, TEST_DIRTY_DATASET_PATH, OPCODE_N_GRAM_MODEL_PATH)
    #predict_(TEST_DATASET_PATH, TEST_DIRTY_DATASET_PATH, CUSTOM_STRING_RFC_MODEL_PATH)
    
if __name__ == '__main__':
    logger.info('******** starting ********')
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))