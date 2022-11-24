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

def load_model(model_path):
    """加载模型
    """

    logger.info('load model[{}]'.format(model_path))
    with open(model_path, 'rb') as fd:
        buffer = fd.read()
    model = pickle.loads(buffer)
    return model

def predict(test_dataset_path, model_path):
    """加载模型进行测试集预测
    """

    # 获取数据集
    test_dataset = pd.read_csv(test_dataset_path)
    logger.info('test dataset shape: ({}, {}).'.format(test_dataset.shape[0], test_dataset.shape[1]))

    #test_dataset = test_dataset.fillna(0)
    #logger.info(test_dataset.info())
    #logger.info(test_dataset.isnull().any())
    
    
    # 模型预测结果
    rrname = test_dataset['rrname']
    X = test_dataset.drop(['rrname'], axis=1, inplace=False).values
    
    logger.info('isnan[{} {} {}], min_max[{} {}]'.format(np.isnan(X).any(),
        np.isfinite(X).all(), np.isinf(X).all(), X.argmin(), X.argmax()))

    model = load_model(model_path)
    result = model.predict(X)
    
    result = [1 if i > 0.25 else 0 for i in result] 
    #result = [0 if i > 0 else 1 for i in result] 
    
    # 存储结果
    df = pd.DataFrame({
                'rrname': rrname,
                'label' : result
           })
    logger.info('predict result shape: {}'.format(df.shape))
    df.to_csv(RESULT_PATH, index=False, header=False)
    
def main():
    predict(FEATURE_TEST_DATASET_PATH, BASELINE_MODEL_PATH)

if __name__ == '__main__':
    #os.system('chcp 936 & cls')
    logger.info('******** starting ********')
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.\n'.format(round(end_time - start_time, 3)))