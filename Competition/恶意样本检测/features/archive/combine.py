# -*- coding: utf-8 -*-
"""
文 件 名: combine.py
文件描述: 将操作指令3-gram(84.88)和pe信息以及自定义字符串特征(89.98)结合起来训练
备    注: 
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

def save_feature_selector(selector):
    """保存特征选择模型
    """
    
    buffer = pickle.dumps(selector)
    with open(COMBINE_RFC_SELECTOR_PATH, "wb+") as fd:
        fd.write(buffer)

def lightgbm_model(X, y):
    """lightgbm模型训练
    """

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logger.info([X_train.shape, X_test.shape, y_train.shape, y_test.shape])
    logger.info(np.unique(y_train))
    logger.info(np.unique(y_test))
    
    LGB = lgb.LGBMClassifier(num_leaves=2**5-1, reg_alpha=0.25, reg_lambda=0.25, objective='binary', max_depth=-1, learning_rate=0.005, min_child_samples=3, random_state=2022, n_estimators=200, subsample=1, colsample_bytree=1).fit(X_train, y_train)
    y_pred = LGB.predict(X_test).astype(int)
    
    score = model_score('LGBMClassifier', y_test, y_pred)
    return LGB, score

def load_model(model_path):
    """加载模型
    """

    with open(model_path, 'rb') as fd:
        buffer = fd.read()
    model = pickle.loads(buffer)
    return model

def random_forest_model(X, y):
    """随机森林模型
    """
    
    RFC = RandomForestClassifier().fit(X, y)
    selector = SelectFromModel(RFC, prefit=True)
    save_feature_selector(selector)
    logger.info('X before selector shape: ({}, {}).'.format(X.shape[0], X.shape[1]))
    X = selector.transform(X)
    logger.info('X_train after selector shape: ({}, {}).'.format(X.shape[0], X.shape[1]))

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
    train_dataset1 = pd.read_csv(TRAIN_DATASET_PATH)
    train_dataset2 = pd.read_csv(TRAIN_OPCODE_3_GRAM_PATH)
    logger.info([train_dataset1.shape, train_dataset2.shape])
    
    # 要先去掉Label，否则会变成Label_1, Label_2
    label = train_dataset2[['FileName', 'Label']]
    train_dataset1.drop(['Label'], axis=1, inplace=True)
    train_dataset2.drop(['Label'], axis=1, inplace=True)
    logger.info([train_dataset1.shape, train_dataset2.shape])
    
    train_dataset  = pd.merge(train_dataset1, train_dataset2, on='FileName')
    logger.info(train_dataset.shape)
    
    train_dataset  = pd.merge(train_dataset, label, on='FileName')
    logger.info(train_dataset.shape)
    
    # 先保存一下
    #train_dataset.to_csv('./train_combine_opcode.csv', sep=',', encoding='utf-8', index=False)
    #train_dataset = pd.read_csv('./train_combine_opcode.csv')
    logger.info(train_dataset.columns)

    # 划分训练集和测试集
    X = train_dataset.drop(['Label', 'FileName'], axis=1).values
    y = train_dataset['Label'].values
    # 使用随机森林生成选择器
    #random_forest_model(X, y)
    selector = load_model(COMBINE_RFC_SELECTOR_PATH)
    X = selector.transform(X)
        
    # 模型训练
    #model, score = random_forest_model(X, y)
    model, score = lightgbm_model(X, y)
    save_training_model(model, score)

if __name__ == '__main__':
    logger.info('******** starting ********')
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))

