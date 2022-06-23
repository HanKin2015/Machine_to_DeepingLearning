# -*- coding: utf-8 -*-
"""
文 件 名: training.py
文件描述: 训练模型
作    者: HeJian
创建日期: 2022.05.29
修改日期：2022.05.30
Copyright (c) 2022 HeJian. All rights reserved.
"""

from common import *

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
    error_ratio = white_is_black / (white_is_black + white_is_white)
    # 惩罚系数
    alpha = 1.2
    # 分数
    score = recall - alpha * error_ratio
    
    logger.info('recall: {}, error_ratio: {}, score: {}'.format(
        round(recall, 4), round(error_ratio, 4), round(score*100, 2)))
    return round(score*100, 2)

def save_test_pred(X_test, y_test, y_pred, score):
    df = pd.DataFrame(X_test)
    df['y_test'] = y_test.reshape(-1,1)
    df['y_pred'] = y_pred.reshape(-1,1)
    logger.info('test_pred result shape: {}'.format(df.shape))
    df.to_csv('he/{}.csv'.format(score), index=False, header=False)

def feature_selection_model(model):
    """模型特征选优
    """
    
    RFC = RandomForestClassifier().fit(X, y)
    select = SelectFromModel(RFC, prefit=True)
    X = select.transform(X)
    return select

def get_RF_best_params(X, y):
    """获取随机森林最佳参数
    """

    random_seed=44
    random_forest_seed=np.random.randint(low=1,high=230)

    # Search optimal hyperparameter
    n_estimators_range=[int(x) for x in np.linspace(start=50,stop=3000,num=60)]
    max_features_range=['auto','sqrt']
    max_depth_range=[int(x) for x in np.linspace(10,500,num=50)]
    max_depth_range.append(None)
    min_samples_split_range=[2,5,10]
    min_samples_leaf_range=[1,2,4,8]
    bootstrap_range=[True,False]

    random_forest_hp_range={'n_estimators':n_estimators_range,
                            'max_features':max_features_range,
                            'max_depth':max_depth_range,
                            'min_samples_split':min_samples_split_range,
                            'min_samples_leaf':min_samples_leaf_range
                            # 'bootstrap':bootstrap_range
                            }
    logger.info(random_forest_hp_range)

    random_forest_model_test_base=RandomForestRegressor()
    random_forest_model_test_random=RandomizedSearchCV(estimator=random_forest_model_test_base,
                                                       param_distributions=random_forest_hp_range,
                                                       n_iter=200,
                                                       n_jobs=-1,
                                                       cv=3,
                                                       verbose=1,
                                                       random_state=random_forest_seed
                                                       )
    random_forest_model_test_random.fit(X, y)

    best_hp_now = random_forest_model_test_random.best_params_
    logger.info(best_hp_now)
    return best_hp_now

def random_forest_model(X, y):
    """随机森林模型

    根据比赛规则计算
    
    Parameters
    ------------
    black_is_black : str
        表示黑样本被预测为黑样本的数目
    black_is_white : str
        表示黑样本被预测为白样本的数目（漏报）
    white_is_black : str
        表示白样本被预测为黑样本的数目（误报）
    white_is_white : str
        表示白样本被预测为白样本的数目
        
    Returns
    -------
    score : float
        分数
    """

    """
    RFC = RandomForestClassifier().fit(X_train, y_train)
    selector = SelectFromModel(RFC, prefit=True)
    save_feature_selector(selector)
    
    logger.info('X_train before selector shape: ({}, {}).'.format(X_train.shape[0], X_train.shape[1]))
    X_train = selector.transform(X_train)
    X_test  = selector.transform(X_test)
    logger.info('X_train after selector shape: ({}, {}).'.format(X_train.shape[0], X_train.shape[1]))
    """

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logger.info([X_train.shape, X_test.shape, y_train.shape, y_test.shape])
    
    #params = get_RF_best_params(X, y)

    #RFC = RandomForestClassifier(params=params)
    RFC = RandomForestClassifier(n_estimators=86, n_jobs=-1)
    RFC.fit(X_train, y_train)
    y_pred = RFC.predict(X_test)

    score = model_score('RandomForestClassifier', y_test, y_pred)
    
    save_test_pred(X_test, y_test, y_pred, score)
    return RFC, score

def extra_trees_model(X_train, X_test, y_train, y_test):
    """使用极端随机树训练
    """
    
    ETC = ExtraTreesClassifier(random_state=0).fit(X_train, y_train)
    y_pred = ETC.predict(X_test)
    
    score = model_score('ExtraTreesClassifier', y_test, y_pred)
    return ETC, score

def XGB_model(X_train, X_test, y_train, y_test):
    """使用XGB模型训练
    """
    
    XGB = xgb.XGBClassifier(n_estimators=110, nthread=-1, max_depth=4, seed=0).fit(X_train, y_train,
                            eval_metric="auc", verbose = False, eval_set=[(X_test, y_test)])
    y_pred = XGB.predict(X_test)
    
    score = model_score('XGBClassifier', y_test, y_pred)
    return XGB, score

def lightgbm_model(X, y):
    """lightgbm模型训练
    """

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logger.info([X_train.shape, X_test.shape, y_train.shape, y_test.shape])
    
    LGB = lgb.LGBMClassifier().fit(X_train, y_train)
    y_pred = LGB.predict(X_test).astype(int)
    
    score = model_score('LGBMClassifier', y_test, y_pred)
    return LGB, score


def lightgbm_model_(X_train, X_test, y_train, y_test):
    """lightgbm模型训练
    """
    
    params = {
          # 这些参数需要学习
          'boosting_type': 'gbdt',
          #'boosting_type': 'dart',
          'objective': 'multiclass',
          'metric': 'multi_logloss',  # 评测函数，这个比较重要
          'min_child_weight': 1.5,
          'num_leaves': 2**5,
          'lambda_l2': 10,
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'tree_method': 'exact',
          'seed': 2022,
          'learning_rate': 0.01, # 学习率 重要
          'num_class': 2,  # 重要
          'silent': True,
          }
    #LGB = lgb.LGBMClassifier(params).fit(X_train, y_train)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=0)

    train_matrix = lgb.Dataset(X_train, label=y_train)
    test_matrix = lgb.Dataset(X_test, label=y_test)

    num_round = 200  # 训练的轮数
    early_stopping_rounds = 10
    LGB = lgb.train(params, 
                  train_matrix,
                  num_round,
                  valid_sets=test_matrix,
                  early_stopping_rounds=early_stopping_rounds)
    y_pred = LGB.predict(X_test, num_iteration=LGB.best_iteration).astype(int)
    
    logger.info('score : ', np.mean((y_pred[:,1]>0.5)==y_valid))
    #score = model_score('LGBMClassifier', y_test, y_pred)
    score = np.mean((y_pred[:,1]>0.5)==y_valid)
    return LGB, score

def MLP_model(X_train, X_test, y_train, y_test):
    """多层感知器模型训练
    """
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    MLP = MLPClassifier(hidden_layer_sizes=(12, 12, 12, 12, 12, 12))

    MLP.fit(X_train,y_train)
    y_pred = MLP.predict(X_test)
    
    score = model_score('MLPClassifier', y_test, y_pred)
    return MLP, score

def gradient_boosting_model(X_train, X_test, y_train, y_test):
    """梯度提升决策树模型训练
    """
    
    original_params = {'n_estimators': 1000, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': 2, 'min_samples_split': 5}
    setting = {'learning_rate': 0.1, 'max_features': 2}
    params = dict(original_params)
    params.update(setting)
 
    GBDT = GradientBoostingClassifier(**params)
    GBDT.fit(X_train, y_train)
    y_pred = GBDT.predict(X_test)
    
    score = model_score('GradientBoostingClassifier', y_test, y_pred)
    return GBDT, score

def fusion_model(X, y):
    """将RF、XGBoost、LightGBM融合（单层Stacking）
    """
    
    # 模型列表
    models = [('RFC', RandomForestClassifier()), 
              ('XGB', xgb.XGBClassifier()),
              ('LGB', lgb.LGBMClassifier())
             ]
             
    # 创建stacking模型
    model = StackingClassifier(models)
    # 设置验证集数据划分方式
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
    # 查看单一模型精度
    logger.info('RFC: {}.'.format(cross_val_score(RandomForestClassifier(), scoring='accuracy', cv=cv, n_jobs=-1).mean()))
    logger.info('XGB: {}.'.format(cross_val_score(xgb.XGBClassifier(), scoring='accuracy', cv=cv, n_jobs=-1).mean()))
    logger.info('LGB: {}.'.format(cross_val_score(lgb.LGBMClassifier(), scoring='accuracy', cv=cv, n_jobs=-1).mean()))
    # 验证模型精度
    n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # 打印模型的精度
    logger.info('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

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

def save_feature_selector(selector):
    """保存特征选择模型
    """
    
    buffer = pickle.dumps(selector)
    with open(MALICIOUS_SAMPLE_DETECTION_SELECTOR_PATH, "wb+") as fd:
        fd.write(buffer)

def main():
    # 获取数据集
    train_dataset = pd.read_csv(TRAIN_DATASET_PATH)
    logger.info('train dataset shape: ({}, {}).'.format(train_dataset.shape[0], train_dataset.shape[1]))
    
    # 划分训练集和测试集 80% 20%
    X = train_dataset.drop(['label', 'FileName'], axis=1).values
    y = train_dataset['label'].values
        
    # 模型训练
    model, score = random_forest_model(X, y)
    #model, score = XGB_model(X_train, X_test, y_train, y_test)
    #model, score = lightgbm_model(X, y)
    #model, score = extra_trees_model(X_train, X_test, y_train, y_test)
    #model, score = MLP_model(X_train, X_test, y_train, y_test)
    #model, score = gradient_boosting_model(X_train, X_test, y_train, y_test)
    #fusion_model(X, y)
    save_training_model(model, score, CUSTOM_STRING_RFC_MODEL_PATH, CUSTOM_STRING_RFC_MODEL_SCORE_PATH)

if __name__ == '__main__':
    logger.info('******** starting ********')
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))