import pandas as pd
import numpy as np
import time
import datetime
from log import logger
import pickle
import os
os.environ['NUMEXPR_MAX_THREADS'] = '64'

# 数据集路径
DATASET_PATH = './dataset/'
# 测试集样本数据集路径
TEST_DATASET_PATH = DATASET_PATH+'test_dataset.csv'
# 模型路径
MODEL_PATH = './model/malicious_sample_detection.model'
# 预测结果存储路径
RESULT_PATH = './result.csv'

def load_model(model_path):
    """加载模型
    """

    with open(model_path, 'rb') as fd:
        buffer = fd.read()
    model = pickle.loads(buffer)
    return model

def special_treatment(dataset):
    return np.ones(dataset.shape[0], dtype=int)

def main():
    # 获取数据集
    test_dataset = pd.read_csv(TEST_DATASET_PATH)
    logger.info('test dataset shape: ({}, {}).'.format(test_dataset.shape[0], test_dataset.shape[1]))
    special_dataset = test_dataset.query('ExceptionError > 0')
    test_dataset = test_dataset.query('ExceptionError == 0')
    
    # 模型预测结果
    file_name1 = test_dataset['FileName']
    x = test_dataset.drop(['FileName', 'ExceptionError'], axis=1, inplace=False)
    X = np.asarray(x)
    model = load_model(MODEL_PATH)
    result1 = model.predict(X)
    
    # 异常样本特殊处理
    file_name2 = special_dataset['FileName']
    result2 = special_treatment(special_dataset)
    
    # 存储结果
    file_name = file_name1.append(file_name2)
    result = np.append(result1, result2)
    df = pd.DataFrame({
                'md5'  : file_name,
                'label': result
           })
    logger.info('predict result shape: {}'.format(df.shape))
    df.to_csv(RESULT_PATH, index=False, header=False)
    
if __name__ == '__main__':
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))