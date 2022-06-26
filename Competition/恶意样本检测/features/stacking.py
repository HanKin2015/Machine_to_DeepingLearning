# -*- coding: utf-8 -*-
"""
文 件 名: stacking.py
文件描述: 模型融合模型
备    注: pip install安装需要指定安装到用户目录，否则容器重启的时候，非用户目录的修改内容会自动删除，恢复到默认状态。
作    者: HeJian
创建日期: 2022.06.25
修改日期：2022.06.25
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


def stacking_model(X, y):
    """将RF、XGBoost、LightGBM融合（单层Stacking）
    """
    
    RANDOM_SEED = 2022
    LGB  = lgb.LGBMClassifier(random_state=RANDOM_SEED)
    RF   = RandomForestClassifier(n_neighbors=86, random_state=RANDOM_SEED)
    XGB  = xgb.XGBClassifier(random_state=RANDOM_SEED)
    GBDT = GradientBoostingClassifier(random_state=RANDOM_SEED)
    DT   = DecisionTreeClassifier(random_state=RANDOM_SEED)
    SVM  = LinearSVC(random_state=RANDOM_SEED)
    ET   = ExtraTreesClassifier(random_state=RANDOM_SEED)
    LR   = LogisticRegression(random_state=RANDOM_SEED)
    
    sclf = StackingCVClassifier(classifiers=[LGB,RF,XGB,GBDT,DT,SVM,LR], meta_classifier=LR, random_state=RANDOM_SEED)
    #logger.info('3-fold cross validation:')
    #for clf, label in zip([clf1, clf2, clf3, clf4, sclf], ['LGB', 'RF', 'ET', 'XGB', 'StackingClassifier']):
    #    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    #    logger.info('Accuracy: {} (+/- {}) [{}]'.format(scores.mean(), scores.std(), label))
    
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logger.info([X_train.shape, X_test.shape, y_train.shape, y_test.shape])
    sclf.fit(X_train, y_train)
    y_pred = sclf.predict(X_test).astype(int)
    
    score = model_score('StackingClassifier', y_test, y_pred)
    return sclf, score

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


def main():
    # 获取数据集
    train_dataset = pd.read_csv(TRAIN_DATASET_PATH)
    logger.info('train dataset shape: ({}, {}).'.format(train_dataset.shape[0], train_dataset.shape[1]))
    
    # 划分训练集和测试集 80% 20%
    X = train_dataset.drop(['label', 'FileName'], axis=1).values
    y = train_dataset['label'].values
        
    # 模型训练
    model, score = stacking_model(X, y)
    save_training_model(model, score, STACKING_MODEL_PATH, STACKING_MODEL_SCORE_PATH)

if __name__ == '__main__':
    logger.info('******** starting ********')
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))