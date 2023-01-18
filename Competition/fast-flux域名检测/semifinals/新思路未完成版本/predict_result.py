# -*- coding: utf-8 -*-
"""
文 件 名: predict_result.py
文件描述: 预测结果
作    者: 重在参与快乐加倍队
创建日期: 2022.10.26
修改日期：2022.11.21

Copyright (c) 2022 ParticipationDoubled. All rights reserved.
"""

from common import *
from feature_engineering import feature_processing

def load_model(model_path):
    """加载模型
    """

    logger.info('load model[{}]'.format(model_path))
    with open(model_path, 'rb') as fd:
        buffer = fd.read()
    model = pickle.loads(buffer)
    return model

def predict(test_dataset, model_path):
    """加载模型进行测试集预测
    """
    
    # 模型预测结果
    rrname = test_dataset['rrname']
    X = test_dataset.drop(['rrname'], axis=1, inplace=False).values
    logger.info('isnan[{} {} {}], min_max[{} {}]'.format(np.isnan(X).any(),
        np.isfinite(X).all(), np.isinf(X).all(), X.argmin(), X.argmax()))

    model_path = './model/model_5.pkl'
    model = load_model(model_path)
    result = model.predict(X)

    result = [1 if i > THRESHOLD else 0 for i in result] 
    #result = [0 if i > 0 else 1 for i in result] 
    
    # 存储结果
    df = pd.DataFrame({
                'rrname': rrname,
                'label' : result
           })
    logger.info('predict result shape: {}'.format(df.shape))
    df.to_csv(RESULT_PATH, index=False, header=False)

def unbalanced_sample(test_dataset):
    """加载模型进行测试集预测
    """
    
    # 模型预测结果
    rrname = test_dataset['rrname']
    X = test_dataset.drop(['rrname'], axis=1, inplace=False).values
    logger.info('isnan[{} {} {}], min_max[{} {}]'.format(np.isnan(X).any(),
        np.isfinite(X).all(), np.isinf(X).all(), X.argmin(), X.argmax()))

    result = np.zeros(test_dataset.shape[0])
    for i in range(8):
        model_path = './model/model_{}.pkl'.format(i)
        score_path = './model/model_{}.score'.format(i)
        model = load_model(model_path)
        pred = model.predict(X)
        result += pred / 8
    
    for threshold in np.arange(0.15, 0.65, 0.1):
        file_name = 'result_{}.csv'.format(threshold)
        tmp = [1 if i > threshold else 0 for i in result] 
        df = pd.DataFrame({
                'rrname': rrname,
                'label' : tmp
           })
        df.to_csv(file_name, index=False, header=False)
    return 0

    result = [1 if i > THRESHOLD else 0 for i in result] 
    #result = [0 if i > 0 else 1 for i in result] 
    
    # 存储结果
    df = pd.DataFrame({
                'rrname': rrname,
                'label' : result
           })
    logger.info('predict result shape: {}'.format(df.shape))
    df.to_csv(RESULT_PATH, index=False, header=False)

def test_dataset_preparation():
    """测试集数据准备，从文件获取或者进行特征工程
    """
    
    if False or not os.path.exists(FEATURE_TEST_DATASET_PATH):
        show_memory_info('initial')
        # 获取数据集
        test_dataset = pd.read_csv(TEST_DATASET_PATH)
        logger.info('test_dataset  shape <{}, {}>'.format(test_dataset.shape[0], test_dataset.shape[1]))

        # 特征工程
        test_dataset = feature_processing(test_dataset)
        logger.info('test_dataset: ({}, {}).'.format(test_dataset.shape[0], test_dataset.shape[1]))
        
        # 保存数据集
        test_dataset.to_csv(FEATURE_TEST_DATASET_PATH, sep=',', encoding='utf-8', index=False)
        logger.info('FEATURE_TEST_DATASET_PATH is saved successfully')
        return test_dataset
    
    test_dataset = pd.read_csv(FEATURE_TEST_DATASET_PATH)
    return test_dataset
    
def main():
    """主函数
    """
    
    # 获取数据集
    test_dataset = test_dataset_preparation()
    logger.info('test_dataset shape: ({}, {}).'.format(test_dataset.shape[0], test_dataset.shape[1]))
    #logger.debug(test_dataset.info())
    
    # 预测结果
    predict(test_dataset, BASELINE_MODEL_PATH)
    #unbalanced_sample(test_dataset)

if __name__ == '__main__':
    #os.system('chcp 936 & cls')
    logger.info('******** starting ********')
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.\n'.format(round(end_time - start_time, 3)))