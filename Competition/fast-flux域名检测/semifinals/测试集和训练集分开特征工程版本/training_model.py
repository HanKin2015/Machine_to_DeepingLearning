# -*- coding: utf-8 -*-
"""
文 件 名: training_model.py
文件描述: 训练模型
作    者: 重在参与快乐加倍队
创建日期: 2022.10.26
修改日期：2022.11.21

Copyright (c) 2022 ParticipationDoubled. All rights reserved.
"""

from common import *
from feature_engineering import feature_processing

def model_score(model_name, y_test, y_pred):
    """模型得分
    
    根据比赛规则计算
    
    Parameters
    ------------
    model_name : str
        模型名字
    y_test : pandas.Series
        验证集结果
    y_pred : pandas.Series
        预测结果
        
    Returns
    -------
    """
    
    logger.info('model {}:'.format(model_name))
    black_is_black, black_is_white, white_is_black, white_is_white = confusion_matrix(y_test, y_pred).ravel()
    logger.info('black_is_black = {}'.format(black_is_black))
    logger.info('black_is_white = {}'.format(black_is_white))
    logger.info('white_is_black = {}'.format(white_is_black))
    logger.info('white_is_white = {}'.format(white_is_white))
    
    # 召回率
    recall = black_is_black / (black_is_black + black_is_white)
    # 误报率
    #error_ratio = white_is_black / (white_is_black + white_is_white)
    # 准确率
    precision = black_is_black / (black_is_black + white_is_black)
    # 惩罚系数
    #alpha = 1.2
    # 分数
    score = 2 * precision * recall / (precision + recall)
    
    logger.info('recall: {}, precision: {}, score: {}'.format(
        round(recall, 4), round(precision, 4), round(score*100, 2)))
    return round(score*100, 2)

def catboost_model(X, y):
    """CatBoostRegressor模型
    """

    cat_grid = {'learning_rate': [0.01]
            ,'depth': [5]
            , 'l2_leaf_reg': [10]
            , 'bootstrap_type': ['Bernoulli']
            , 'od_type': ['Iter']
            , 'od_wait': [50]
            , 'random_seed': [2022]
            , 'allow_writing_files': [False]
           }
    folds = 5
    seed = 2022
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    catgrid = GridSearchCV(estimator=CatBoostRegressor(), param_grid=cat_grid, cv=kf, n_jobs=-1, verbose=folds)
    catgrid.fit(X, y, verbose=500)
    logger.info([catgrid.best_params_, catgrid.best_score_])

    #查看最大值、最小值可以在线下初步确定预测效果
    y_pred = catgrid.predict(X)
    logger.info([y_pred.min(), y_pred.max()])
    y_pred = [1 if i > THRESHOLD else 0 for i in y_pred]
    
    logger.info(classification_report(y, y_pred))
    score = model_score('CatBoostRegressor', y, y_pred)
    return catgrid, score

def save_training_model(model, score, model_path=BASELINE_MODEL_PATH, score_path=BASELINE_MODEL_SCORE_PATH):
    """保存训练模型
    """
    
    before_score = 0
    if os.path.exists(score_path):
        with open(score_path, 'r') as fd:
            before_score = fd.read()
    
    before_score = 0    # 设置一直更新保存模型
    if score > float(before_score):
        logger.info('model need to be changed, old score {}, new score {}'.format(before_score, score))
        logger.info('save model[{}]'.format(model_path))
        buffer = pickle.dumps(model)
        with open(model_path, "wb+") as fd:
            fd.write(buffer)
        with open(score_path, 'w') as fd:
            fd.write(str(score))

def train_dataset_preparation():
    """训练集数据准备，从文件获取或者进行特征工程
    """
    
    if False or not os.path.exists(FEATURE_TRAIN_DATASET_PATH):
        show_memory_info('initial')
        # 获取数据集
        train_dataset = pd.read_csv(TRAIN_DATASET_PATH)
        logger.info('train_dataset  shape <{}, {}>'.format(train_dataset.shape[0], train_dataset.shape[1]))

        # 特征工程
        train_dataset = feature_processing(train_dataset)
        logger.info('train_dataset: ({}, {}).'.format(train_dataset.shape[0], train_dataset.shape[1]))

        # 给训练集添加标签
        train_label = pd.read_csv(TRAIN_LABEL_PATH)
        logger.info('train_label shape <{}, {}>'.format(train_label.shape[0], train_label.shape[1]))
        train_label.columns = ['rrname', 'label']
        train_dataset = train_dataset.merge(train_label, on='rrname', how='left')
        logger.info([train_dataset.shape])
        del train_label
        gc.collect()
        
        # 保存数据集
        train_dataset.to_csv(FEATURE_TRAIN_DATASET_PATH, sep=',', encoding='utf-8', index=False)
        logger.info('FEATURE_TRAIN_DATASET_PATH is saved successfully')
        return train_dataset
    
    train_dataset = pd.read_csv(FEATURE_TRAIN_DATASET_PATH)
    return train_dataset

def main():
    """主函数
    """
    
    # 获取数据集
    train_dataset = train_dataset_preparation()
    logger.info('train dataset shape: ({}, {}).'.format(train_dataset.shape[0], train_dataset.shape[1]))
    logger.debug(train_dataset.info())
    
    # 特征和标签分离
    X = train_dataset.drop(['rrname', 'label'], axis=1).values
    y = train_dataset['label'].values
    logger.info('isnan[{} {} {}], min_max[{} {}]'.format(np.isnan(X).any(),
        np.isfinite(X).all(), np.isinf(X).all(), X.argmin(), X.argmax()))
        
    # 模型训练
    model, score = catboost_model(X, y)
    
    # 保存训练模型
    save_training_model(model, score)

if __name__ == '__main__':
    #os.system('chcp 936 & cls')
    logger.info('******** starting ********')
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.\n'.format(round(end_time - start_time, 3)))