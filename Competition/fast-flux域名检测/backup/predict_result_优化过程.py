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

def voting_decision(df, rrname):
    """投票决定最终结果
    训练集标签结果中有138144个0，2149个1
    """
    
    tmp = df[df['domain'] == rrname]
    if tmp[tmp['label'] == 0].shape[0] >= tmp[tmp['label'] == 1].shape[0]:
        return {'domain': rrname, 'label': 0}
    else:
        return {'domain': rrname, 'label': 1}

def result_processing(df):
    """
    """
    
    tasks = []
    
    with ThreadPoolExecutor(max_workers=THREAD_NUM) as pool:
        for rrname in set(df['domain']):
            task = pool.submit(voting_decision, df, rrname)
            tasks.append(task)
            
    result = pd.DataFrame(columns=['domain', 'label'])
    logger.debug('tasks count: {}'.format(len(tasks)))
    for task in tasks:
        logger.debug(task.result())
        #result = result.append(task.result(), ignore_index=True)
        result = pd.concat([result, pd.DataFrame([task.result()])], ignore_index=True)
    logger.info('predict result shape: {}'.format(result.shape))
    result.to_csv(RESULT_PATH, index=False, header=False)

def result_processing_(df):
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
    
def result_processing_v3(df):
    """
    """
    
    label0_list = set(df[df['label'] == 0]['domain'])
    label1_list = set(df[df['label'] == 1]['domain'])
    intersection = list(label0_list & label1_list)
    logger.info('{}, {}, {}'.format(len(label0_list), len(label1_list), len(intersection)))
    
    domain = []
    label = []
    for rrname in label0_list:
        if rrname not in intersection:
            domain.append(rrname)
            label.append(0)
            
    for rrname in label1_list:
        if rrname not in intersection:
            domain.append(rrname)
            label.append(1)

    result = pd.DataFrame({
                'domain': domain,
                'label' : label
           })

    tasks = []
    with ThreadPoolExecutor(max_workers=THREAD_NUM) as pool:
        for rrname in intersection:
            task = pool.submit(voting_decision, df, rrname)
            tasks.append(task)

    for task in tasks:
        #result = result.append(task.result(), ignore_index=True)
        result = pd.concat([result, pd.DataFrame([task.result()])], ignore_index=True)
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
    result_processing_v3(df)
    
def main():
    predict(TEST_DATASET_PATH, RFC_MODEL_PATH)

if __name__ == '__main__':
    #os.system('chcp 936 & cls')
    logger.info('******** starting ********')
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.\n'.format(round(end_time - start_time, 3)))