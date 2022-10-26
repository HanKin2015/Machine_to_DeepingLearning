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

def lightgbm_model(X, y):
    """lightgbm模型训练
    """

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print([X_train.shape, X_test.shape, y_train.shape, y_test.shape])
    print(np.unique(y_train))
    print(np.unique(y_test))
    
    LGB = lgb.LGBMClassifier().fit(X_train, y_train)
    y_pred = LGB.predict(X_test).astype(int)
    
    # 模型评估 （测试集）
    print(classification_report(test_y,y_test_pred))  # 评估模型（真实的值与预测的结果对比）
    
    return LGB

def random_forest_model(X, y):
    """随机森林模型
    """

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print([X_train.shape, X_test.shape, y_train.shape, y_test.shape])

    RFC = RandomForestClassifier(n_jobs=-1)
    RFC.fit(X_train, y_train)
    y_pred = RFC.predict(X_test)
    print(classification_report(y_test, y_pred))  # 评估模型（真实的值与预测的结果对比）
    
    return RFC

def predict(model):
    """加载模型进行测试集预测
    """

    # 获取数据集
    test_dataset = pd.read_csv(TEST_DATA_PATH)
    print('test dataset shape: ({}, {}).'.format(test_dataset.shape[0], test_dataset.shape[1]))

    # 模型预测结果
    ids = test_dataset['ids']
    X = test_dataset.drop(['ids'], axis=1, inplace=False).values

    result = model.predict(X)
    
    # 存储结果
    df = pd.DataFrame({
                'ids'  : ids,
                'label': result
           })
    print('predict result shape: {}'.format(df.shape))
    df.to_csv(RESULT_PATH, index=False, header=False)

def main():
    # 获取数据集
    train_dataset = pd.read_csv(TRAIN_DATASET_PATH)
    print('train dataset shape: ({}, {}).'.format(train_dataset.shape[0], train_dataset.shape[1]))
    
    X = train_dataset.drop(['ids', 'label'], axis=1).values
    y = train_dataset['label'].values
        
    # 模型训练
    #model = lightgbm_model(X, y)
    model = random_forest_model(X, y)
    #save_training_model(model, score, CUSTOM_STRING_RFC_MODEL_PATH, CUSTOM_STRING_RFC_MODEL_SCORE_PATH)

    predict(model)

if __name__ == '__main__':
    start_time = time.time()

    #debug()
    main()

    end_time = time.time()
    print('process spend {} s.'.format(round(end_time - start_time, 3)))