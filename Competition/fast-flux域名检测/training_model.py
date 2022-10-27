# -*- coding: utf-8 -*-
"""
文 件 名: training_model.py
文件描述: 训练模型
作    者: HanKin
创建日期: 2022.10.26
修改日期：2022.10.26

Copyright (c) 2022 HanKin. All rights reserved.
"""

from common import *

def lightgbm_model(X, y):
    """lightgbm模型训练
    """

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logger.info([X_train.shape, X_test.shape, y_train.shape, y_test.shape])
    logger.info(np.unique(y_train))
    logger.info(np.unique(y_test))
    
    # K折交叉验证与学习曲线的联合使用来获取最优K值
    scores_cross = []
    Ks = []
    for k in range(75, 100):
        LGB = lgb.LGBMClassifier(n_estimators=k, n_jobs=-1)
        score_cross = cross_val_score(LGB, X_train, y_train, cv=5).mean()
        scores_cross.append(score_cross)
        Ks.append(k)
    scores_arr = np.array(scores_cross)
    Ks_arr = np.array(Ks)
    score_best = scores_arr.max()  # 在存储评分的array中找到最高评分
    index_best = scores_arr.argmax()  # 找到array中最高评分所对应的下标
    Ks_best = Ks_arr[index_best]  # 根据下标找到最高评分所对应的K值
    logger.info('Ks_best: {}.'.format(Ks_best)) 
    
    LGB = lgb.LGBMClassifier(n_estimators=Ks_best, n_jobs=-1).fit(X_train, y_train)
    y_pred = LGB.predict(X_test).astype(int)
    
    score = model_score('LGBMClassifier', y_test, y_pred)
    return LGB, score

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

def random_forest_model(X, y):
    """随机森林模型
    """

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logger.info([X_train.shape, X_test.shape, y_train.shape, y_test.shape])

    RFC = RandomForestClassifier(n_jobs=-1)
    RFC.fit(X_train, y_train)
    y_pred = RFC.predict(X_test)
    logger.info(classification_report(y_test, y_pred))  # 评估模型（真实的值与预测的结果对比）

    score = model_score('RandomForestClassifier', y_test, y_pred)
    return RFC, score

def save_training_model(model, score, model_path=MODEL_PATH, score_path=MODEL_SCORE_PATH):
    """保存训练模型
    """
    
    before_score = 0
    if os.path.exists(score_path):
        with open(score_path, 'r') as fd:
            before_score = fd.read()
            
    if score > float(before_score):
        logger.info('model need to be changed, old score {}, new score {}'.format(before_score, score))
        buffer = pickle.dumps(model)
        with open(model_path, "wb+") as fd:
            fd.write(buffer)
        with open(score_path, 'w') as fd:
            fd.write(str(score))

def main():
    # 获取数据集
    train_dataset = pd.read_csv(TRAIN_DATASET_PATH)
    logger.info('train dataset shape: ({}, {}).'.format(train_dataset.shape[0], train_dataset.shape[1]))
    
    X = train_dataset.drop(['rrname', 'label'], axis=1).values
    y = train_dataset['label'].values
        
    # 模型训练
    #model = lightgbm_model(X, y)
    model, score = random_forest_model(X, y)
    save_training_model(model, score, RFC_MODEL_PATH, RFC_MODEL_SCORE_PATH)

if __name__ == '__main__':
    #os.system('chcp 936 & cls')
    logger.info('******** starting ********')
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.\n'.format(round(end_time - start_time, 3)))