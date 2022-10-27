# -*- coding: utf-8 -*-
"""
文 件 名: predict_result.py
文件描述: 预测结果
作    者: HanKin
创建日期: 2022.10.26
修改日期：2022.10.26

Copyright (c) 2022 HanKin. All rights reserved.
"""

from common import *

def load_model(model_path):
    """加载模型
    """

    with open(model_path, 'rb') as fd:
        buffer = fd.read()
    model = pickle.loads(buffer)
    return model

def result_processing(df):
    """
    """
    
    domain = []
    label = []
    for rrname in set(df['domain']):
        domain.append(rrname)
        tmp = df[df['domain'] == rrname]
        if tmp[tmp['label'] == 0].shape[0] > tmp[tmp['label'] == 1].shape[0]:
            label.append(0)
        else:
            label.append(1)
            
    result = pd.DataFrame({
                'domain': domain,
                'label' : label
           })
    logger.info('predict result shape: {}'.format(result.shape))
    result.to_csv(RESULT_PATH, index=False, header=False)

def predict(test_dataset_path, model_path):
    """加载模型进行测试集预测
    """

    # 获取数据集
    test_dataset = pd.read_csv(test_dataset_path)
    logger.info('test dataset shape: ({}, {}).'.format(test_dataset.shape[0], test_dataset.shape[1]))

    # 模型预测结果
    rrname = test_dataset['rrname']
    X = test_dataset.drop(['rrname'], axis=1, inplace=False).values
    #selector = load_model(MALICIOUS_SAMPLE_DETECTION_SELECTOR_PATH)
    #X = selector.transform(X)
    model = load_model(model_path)
    result = model.predict(X)
    
    # 存储结果
    df = pd.DataFrame({
                'domain': rrname,
                'label' : result
           })
    logger.info('predict result shape: {}'.format(df.shape))
    result_processing(df)
    
def main():
    predict(TEST_DATASET_PATH, RFC_MODEL_PATH)

if __name__ == '__main__':
    #os.system('chcp 936 & cls')
    logger.info('******** starting ********')
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.\n'.format(round(end_time - start_time, 3)))