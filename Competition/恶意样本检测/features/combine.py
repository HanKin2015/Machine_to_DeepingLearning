# -*- coding: utf-8 -*-
"""
文 件 名: combine.py
文件描述: 将操作指令n-gram(84.88)和图像矩阵特征(81.98)结合起来训练
备    注: 只有81.84，运行4216秒左右 
作    者: HeJian
创建日期: 2022.06.24
修改日期：2022.06.24
Copyright (c) 2022 HeJian. All rights reserved.
"""

from common import *

def model_score(model_name, y_test, y_pred):
    """模型得分
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
    error_ratio = white_is_black / (white_is_black + white_is_white)
    # 惩罚系数
    alpha = 1.2
    # 分数
    score = recall - alpha * error_ratio
    
    logger.info('recall: {}, error_ratio: {}, score: {}'.format(
        round(recall, 4), round(error_ratio, 4), round(score*100, 2)))
    return round(score*100, 2)

def random_forest_model(X, y):
    """随机森林模型
    """

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logger.info([X_train.shape, X_test.shape, y_train.shape, y_test.shape])

    RFC = RandomForestClassifier(n_estimators=86, n_jobs=-1)
    RFC.fit(X_train, y_train)
    y_pred = RFC.predict(X_test)

    score = model_score('RandomForestClassifier', y_test, y_pred)
    return RFC, score

def save_training_model(model, score):
    """保存训练模型
    """
    
    before_score = 0
    if os.path.exists(COMBINE_RFC_MODEL_SCORE_PATH):
        with open(COMBINE_RFC_MODEL_SCORE_PATH, 'r') as fd:
            before_score = fd.read()

    if score > float(before_score):
        logger.info('~~~~~[model changed]~~~~~')
        buffer = pickle.dumps(model)
        with open(COMBINE_RFC_MODEL_PATH, "wb+") as fd:
            fd.write(buffer)
        with open(COMBINE_RFC_MODEL_SCORE_PATH, 'w') as fd:
            fd.write(str(score))

def main():
    # 获取数据集
    #train_dataset1 = pd.read_csv(TRAIN_IMAGE_MATRIX_PATH)
    train_dataset1 = pd.read_csv(TRAIN_DATASET_PATH)
    train_dataset2 = pd.read_csv(TRAIN_OPCODE_3_GRAM_PATH)
    logger.info([train_dataset1.shape, train_dataset2.shape])
    
    # 要先去掉Label，否则会变成Label_1, Label_2
    label = train_dataset2[['FileName', 'Label']]
    train_dataset1.drop(['label'], axis=1, inplace=True)
    train_dataset2.drop(['Label'], axis=1, inplace=True)
    logger.info([train_dataset1.shape, train_dataset2.shape])
    
    train_dataset  = pd.merge(train_dataset1, train_dataset2, on='FileName')
    logger.info(train_dataset.shape)
    
    train_dataset  = pd.merge(train_dataset, label, on='FileName')
    logger.info(train_dataset.shape)
    
    # 先保存一下
    #train_dataset.to_csv('./temp.csv', sep=',', encoding='utf-8', index=False)
    logger.info(train_dataset.columns)

    # 划分训练集和测试集
    X = train_dataset.drop(['Label', 'FileName'], axis=1).values
    y = train_dataset['Label'].values
        
    # 模型训练
    model, score = random_forest_model(X, y)
    save_training_model(model, score)

if __name__ == '__main__':
    logger.info('******** starting ********')
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))

