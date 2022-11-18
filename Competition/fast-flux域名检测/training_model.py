# -*- coding: utf-8 -*-
"""
文 件 名: training_model.py
文件描述: 训练模型
作    者: HanKin
创建日期: 2022.10.26
修改日期：2022.11.17

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
    for k in range(85, 90):
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

def svm_model(X, y):
    """SVM模型
    """
    
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logger.info([X_train.shape, X_test.shape, y_train.shape, y_test.shape])
    
    logger.info('training model......')
    SVM = svm.SVC(kernel='linear').fit(X_train, y_train)
    y_pred = SVM.predict(X_test)
    logger.info(classification_report(y_test, y_pred))
    score = model_score('SVM', y_test, y_pred)
    return SVM, score

def save_training_model(model, score, model_path=BASELINE_MODEL_PATH, score_path=BASELINE_MODEL_SCORE_PATH):
    """保存训练模型
    """
    
    before_score = 0
    if os.path.exists(score_path):
        with open(score_path, 'r') as fd:
            before_score = fd.read()
    
    before_score = 0
    if score > float(before_score):
        logger.info('model need to be changed, old score {}, new score {}'.format(before_score, score))
        logger.info('save model[{}]'.format(model_path))
        buffer = pickle.dumps(model)
        with open(model_path, "wb+") as fd:
            fd.write(buffer)
        with open(score_path, 'w') as fd:
            fd.write(str(score))

def cv_model(clf, train_x, train_y, test_x, clf_name):
    folds = 5
    seed = 2022
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    oof = np.zeros(train_x.shape[0])
    pred = np.zeros(test_x.shape[0])

    cv_scores = []

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]

        if clf_name == "lgb":
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'min_child_weight': 5,
                'num_leaves': 2 ** 5,
                'lambda_l2': 10,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 4,
                'learning_rate': 0.01,
                'seed': 2022,
                'n_jobs':-1,
                'verbose': -1,
            }

            model = clf.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], 
                              categorical_feature=[], verbose_eval=500, early_stopping_rounds=500)
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration)
            
            # print(list(sorted(zip(features, model.feature_importance("gain")), key=lambda x: x[1], reverse=True))[:20])
                
        if clf_name == "xgb":
            train_matrix = clf.DMatrix(trn_x , label=trn_y)
            valid_matrix = clf.DMatrix(val_x , label=val_y)
            test_matrix = clf.DMatrix(test_x)
            
            params = {'booster': 'gbtree',
                      'objective': 'binary:logistic',
                      'eval_metric': 'auc',
                      'gamma': 1,
                      'min_child_weight': 1.5,
                      'max_depth': 5,
                      'lambda': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'eta': 0.01,
                      'tree_method': 'exact',
                      'seed': 2022,
                      'nthread': 36
                      }
            
            watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]
            
            model = clf.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=500, early_stopping_rounds=200)
            val_pred  = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)
            test_pred = model.predict(test_matrix , ntree_limit=model.best_ntree_limit)
            
        if clf_name == "cat":
            params = {'learning_rate': 0.01, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli',
                      'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}
            
            model = clf(iterations=20000, **params)
            model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                      cat_features=[], use_best_model=True, verbose=500)
            
            val_pred  = model.predict(val_x)
            test_pred = model.predict(test_x)
            
        oof[valid_index] = val_pred
        pred += test_pred / kf.n_splits
        cv_scores.append(f1_score(val_y, pd.DataFrame(val_pred)[0].apply(lambda x: 1 if x>0.35 else 0), average='macro'))
        
        print(cv_scores)
        
    logger.info("%s_train_score_list:" % clf_name, cv_scores)
    logger.info("%s_score_mean:" % clf_name, np.mean(cv_scores))
    logger.info("%s_score_std:" % clf_name, np.std(cv_scores))
    return oof, pred

def lgb_model(x_train, y_train, x_test):
    return cv_model(lgb, x_train, y_train, x_test, "lgb")

def xgb_model(x_train, y_train, x_test):
    return  cv_model(xgb, x_train, y_train, x_test, "xgb")

def cat_model(x_train, y_train, x_test):
    return cv_model(CatBoostRegressor, x_train, y_train, x_test, "cat") 

def main():
    # 获取数据集
    train_dataset = pd.read_csv(FEATURE_TRAIN_DATASET_PATH)
    logger.info('train dataset shape: ({}, {}).'.format(train_dataset.shape[0], train_dataset.shape[1]))
    logger.info(train_dataset.info())
    
    X = train_dataset.drop(['rrname', 'label'], axis=1).values
    y = train_dataset['label'].values
        
    # 模型训练
    #model, score = lightgbm_model(X, y)
    #save_training_model(model, score)
    
    test_dataset = pd.read_csv(FEATURE_TEST_DATASET_PATH)
    x_train = train_dataset.drop(['rrname', 'label'], axis=1)
    y_train = train_dataset['label']
    x_test  = test_dataset.drop(['rrname'], axis=1)
    cat_train, cat_test = cat_model(x_train, y_train, x_test)
    logger.info(cat_test.shape)
    
    test_dataset['label'] = cat_test

    for threshold in np.arange(0.15, 0.65, 0.1):
        #print(threshold)
        file_name = 'result_{}.csv'.format(threshold)
        #print(file_name)
        test_dataset['result'] = test_dataset['label'].apply(lambda x: 1 if x > threshold else 0)
        
        if True:
            change_count = 0
            for index, row in test_dataset.iterrows():
                if row['openSource_sum'] > 5000:
                    if row['openSource_sum'] / row['rdatalist_count'] > 0.8:
                        if row['result'] == 0:
                            change_count += 1
                            #row['result'] = 1
                            test_dataset.loc[index, 'result'] = 1
            logger.info('change_count: {}'.format(change_count))
        
        test_dataset[['rrname','result']].to_csv(file_name, index=False, header=False)
    #test_dataset['label'] = test_dataset['label'].apply(lambda x: 1 if x>0.35 else 0)
    # 提交最终结果
    #test_dataset[['rrname','label']].to_csv("result.csv", index=False, header=False)

    #model, score = random_forest_model(X, y)
    #save_training_model(model, score, RFC_MODEL_PATH, RFC_MODEL_SCORE_PATH)

if __name__ == '__main__':
    #os.system('chcp 936 & cls')
    logger.info('******** starting ********')
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.\n'.format(round(end_time - start_time, 3)))