# -*- coding: utf-8 -*-
"""
文 件 名: get_data.py
文件描述: 获取数据
作    者: HanKin
创建日期: 2022.07.20
修改日期：2022.07.20

Copyright (c) 2022 HanKin. All rights reserved.
"""

from common import *

def save_training_model(model, score, model_path=BASELINE_RFC_MODEL_PATH, score_path=BASELINE_RFC_MODEL_SCORE_PATH):
    """保存训练模型
    """
    
    before_score = 0
    if os.path.exists(score_path):
        with open(score_path, 'r') as fd:
            before_score = fd.read()
            
    if score > float(before_score):
        logger.info('~~~~~[model changed]~~~~~')
        buffer = pickle.dumps(model)
        with open(model_path, "wb+") as fd:
            fd.write(buffer)
        with open(score_path, 'w') as fd:
            fd.write(str(score))

def lightgbm_model(X, y):
    """lightgbm模型训练
    """

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print([X_train.shape, X_test.shape, y_train.shape, y_test.shape])
    print(np.unique(y_train))
    print(np.unique(y_test))
    
    LGB = lgb.LGBMClassifier().fit(X_train, y_train)
    y_pred = LGB.predict(X_test).astype(int)
    
    score = model_score('LGBMClassifier', y_test, y_pred)
    return LGB, score
    
def main():
    # 获取数据集
    train_dataset = pd.read_csv(TRAIN_DATASET_PATH)
    logger.info('train dataset shape: ({}, {}).'.format(train_dataset.shape[0], train_dataset.shape[1]))
    
    X = train_dataset.drop(['ids', 'label'], axis=1).values
    y = train_dataset['label'].values
        
    # 模型训练
    model, score = lightgbm_model(X, y)
    save_training_model(model, score, CUSTOM_STRING_RFC_MODEL_PATH, CUSTOM_STRING_RFC_MODEL_SCORE_PATH)

if __name__ == '__main__':
    start_time = time.time()

    #debug()
    main()

    end_time = time.time()
    print('process spend {} s.'.format(round(end_time - start_time, 3)))