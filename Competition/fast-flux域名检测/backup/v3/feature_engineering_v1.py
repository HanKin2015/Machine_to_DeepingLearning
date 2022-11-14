# -*- coding: utf-8 -*-
"""
文 件 名: feature_engineering.py
文件描述: 特征工程
作    者: HanKin
创建日期: 2022.10.18
修改日期：2022.10.18

Copyright (c) 2022 HanKin. All rights reserved.
"""

from common import *

def delete_uncorrelated_features(dataset):
    """删除相关性低的特征
    """
    
    uncorrelated_features = ['time_first', 'time_last', 'rrtype', 'rdata', 'bailiwick', 'rrname_label', 'bailiwick_label']
    dataset.drop(uncorrelated_features, axis=1, inplace=True)
    return dataset

def discrete_value_processing(dataset):
    """离散值处理
    """
    
    gle = LabelEncoder()
    
    rrname_label = gle.fit_transform(dataset['rrname'])
    rrname_mapping = {index: label for index, label in enumerate(gle.classes_)}
    #logger.info(rrname_mapping)
    dataset['rrname_label'] = rrname_label
    
    bailiwick_label = gle.fit_transform(dataset['bailiwick'])
    bailiwick_mapping = {index: label for index, label in enumerate(gle.classes_)}
    #logger.info(bailiwick_mapping)
    dataset['bailiwick_label'] = bailiwick_label
    
    dataset['rdata_count'] = dataset['rdata'].apply(lambda x: len(x.split(',')))
    dataset['rrname_bailiwick'] = (dataset['rrname'] == dataset['bailiwick'])
    dataset['rrname_bailiwick'] = dataset['rrname_bailiwick'].astype(int)
    return dataset

def time_processing(dataset):
    """观察时间处理
    """
    
    dataset['time_interval'] = dataset['time_last'] - dataset['time_first']
    dataset['time_interval'] = dataset['time_interval'].apply(lambda x: int(x / 3600 / 24))
    return dataset

def label_encode(series):
    """自然数编码
    """
    
    unique = list(series.unique())
    #unique.sort()
    return series.map(dict(zip(unique, range(series.nunique()))))

def feature_extraction(dataset):
    """特征提取
    """

    # 聚合
    agg_df = dataset.groupby(['rrname']).agg({'rdata':[list],
                                            'count':['sum','max', 'min'],
                                            'time_first':['max', 'min'],
                                            'time_last':['max', 'min'],
                                            'rrtype':['unique'],
                                            'bailiwick':[list, 'nunique'],
                                            'time_diff':['sum', 'max', 'min', 'std']}).reset_index()

    # 重置列名
    agg_df.columns = [''.join(col).strip() for col in agg_df.columns.values]
    agg_df.reset_index(drop=True, inplace=True)
    
    # 去掉两边引号
    agg_df['rdatalist'] = agg_df['rdatalist'].apply(lambda x:','.join([i[1:-1] for i in x]).replace('\'','').split(','))

    agg_df['rdatalist_count']     = agg_df['rdatalist'].apply(lambda x:len(x))
    agg_df['rdatalist_nunique']   = agg_df['rdatalist'].apply(lambda x:len(set(x)))
    agg_df['bailiwicklist_count'] = agg_df['bailiwicklist'].apply(lambda x:len(x))
    
    # 自然数编码
    for col in ['rrtypeunique']:
        agg_df[col] = label_encode(agg_df[col].astype(str))
    
    return agg_df
    
def dataset_pre_processing(dataset):
    """数据集预处理
    """
    
    # 去掉列表的左右中括号
    dataset['rdata'] = dataset['rdata'].apply(lambda x:x[1:-1])

    # 去掉空格
    dataset['rdata'] = dataset['rdata'].apply(lambda x:x.replace(' ', ''))

    # 时间转换成天数
    #dataset['time_first'] = dataset['time_first'].apply(lambda x: int(x / 3600 / 24))
    #dataset['time_last'] = dataset['time_last'].apply(lambda x: int(x / 3600 / 24))

    # 时间差
    dataset['time_diff'] = dataset['time_last'] - dataset['time_first']

    return dataset

def feature_processing(dataset, ip_info):
    """特征工程
    """
    
    # 数据集预处理
    dataset = dataset_pre_processing(dataset)
    # 特征提取
    agg_df = feature_extraction(dataset)
    # pdns数据集和ip_info数据集结合
    agg_df = pdns_concat_ip_info_v2(agg_df, ip_info)
    # 通过ip地址获取洲和国家的数量
    #dataset = get_ip_info(dataset, ip_info)
    # 删除不相关的特征（相关性低）
    #dataset = delete_uncorrelated_features(dataset)
    
    # 特殊处理
    agg_df = other(agg_df)
    return agg_df

def pdns_concat_ip_info(agg_df, ip_info):
    """pdns数据集和ip_info数据集结合
    """
    
    logger.info([ip_info.shape])
    
    # 创建ip_info字典
    col_dicts = ['judgement', 'networkTags', 'threatTags', 'expired', 'continent',
             'country', 'province', 'city', 'district', 'timeZone', 'organization', 'operator']
    for col_dict in col_dicts:
        exec('{}_dict = dict()'.format(col_dict), globals())
        exec("{}_dict = dict(zip(ip_info['ip'], ip_info['{}']))".format(col_dict, col_dict))

    # 初始化列表
    for col_dict in col_dicts:
        exec('{}_li = []'.format(col_dict), globals())

    np.array(judgement_dict)

    not_found_ip_count = 0
    for items in tqdm(agg_df['rdatalist'].values):
        # 置空临时列表
        for col_dict in col_dicts:
            exec('{}_tmp = []'.format(col_dict), globals())
        
        for ip in items:
            try:
                for col_dict in col_dicts:
                    exec('{}_tmp.append({}_dict[ip])'.format(col_dict, col_dict))
            except:
                logger.debug('failed, ip {}'.format(ip))
                not_found_ip_count += 1
                pass
        
        # 将临时列表添加到主列表
        for col_dict in col_dicts:
            exec('{}_li.append({}_tmp)'.format(col_dict, col_dict))
    
    logger.info('here is {} ips which are not found'.format(not_found_ip_count))
    
    # 将ip_info提取的特征添加到数据集中
    info_df = pd.DataFrame({'judgement':np.array(judgement_li),'networkTags':np.array(networkTags_li),'threatTags':np.array(threatTags_li),
                        'expired':np.array(expired_li),'continent':np.array(continent_li),'country':np.array(country_li),
                        'province':np.array(province_li),'city':np.array(city_li),'district':np.array(district_li),
                        'timeZone':np.array(timeZone_li),'organization':np.array(organization_li),'operator':np.array(operator_li)})
    agg_df = pd.concat([agg_df, info_df], axis=1)
    return agg_df

def pdns_concat_ip_info_v2(agg_df, ip_info):
    """pdns数据集和ip_info数据集结合
    """
    
    logger.info([ip_info.shape])
    
    # 创建ip_info字典
    judgement_dict = dict(zip(ip_info['ip'], ip_info['judgement']))
    networkTags_dict = dict(zip(ip_info['ip'], ip_info['networkTags']))
    threatTags_dict = dict(zip(ip_info['ip'], ip_info['threatTags']))
    expired_dict = dict(zip(ip_info['ip'], ip_info['expired']))

    continent_dict = dict(zip(ip_info['ip'], ip_info['continent']))
    country_dict = dict(zip(ip_info['ip'], ip_info['country']))
    province_dict = dict(zip(ip_info['ip'], ip_info['province']))
    city_dict = dict(zip(ip_info['ip'], ip_info['city']))
    district_dict = dict(zip(ip_info['ip'], ip_info['district']))
    timeZone_dict = dict(zip(ip_info['ip'], ip_info['timeZone']))

    organization_dict = dict(zip(ip_info['ip'], ip_info['organization']))
    operator_dict = dict(zip(ip_info['ip'], ip_info['operator']))

    # 初始化列表
    judgement_li = []
    networkTags_li = []
    threatTags_li = []
    expired_li = []
    continent_li = []
    country_li = []
    province_li = []
    city_li = []
    district_li = []
    timeZone_li = []
    organization_li = []
    operator_li = []

    not_found_ip_count = 0
    for items in tqdm(agg_df['rdatalist'].values):
        # 置空临时列表
        judgement_tmp = []
        networkTags_tmp = []
        threatTags_tmp = []
        expired_tmp = []
        continent_tmp = []
        country_tmp = []
        province_tmp = []
        city_tmp = []
        district_tmp = []
        timeZone_tmp = []
        organization_tmp = []
        operator_tmp = []
        
        for ip in items:
            try:
                judgement_tmp.append(judgement_dict[ip])
                networkTags_tmp.append(networkTags_dict[ip])
                threatTags_tmp.append(threatTags_dict[ip])
                expired_tmp.append(expired_dict[ip])
                continent_tmp.append(continent_dict[ip])
                country_tmp.append(country_dict[ip])
                province_tmp.append(province_dict[ip])
                city_tmp.append(city_dict[ip])
                district_tmp.append(district_dict[ip])
                timeZone_tmp.append(timeZone_dict[ip])
                organization_tmp.append(organization_dict[ip])
                operator_tmp.append(operator_dict[ip])
            except:
                logger.debug('failed, ip {}'.format(ip))
                not_found_ip_count += 1
                pass
        
        # 将临时列表添加到主列表
        judgement_li.append(judgement_tmp)
        networkTags_li.append(networkTags_tmp)
        threatTags_li.append(threatTags_tmp)
        expired_li.append(expired_tmp)
        continent_li.append(continent_tmp)
        country_li.append(country_tmp)
        province_li.append(province_tmp)
        city_li.append(city_tmp)
        district_li.append(district_tmp)
        timeZone_li.append(timeZone_tmp)
        organization_li.append(organization_tmp)
        operator_li.append(operator_tmp)
    
    logger.info('here is {} ips which are not found'.format(not_found_ip_count))
    
    # 将ip_info提取的特征添加到数据集中
    info_df = pd.DataFrame({'judgement':np.array(judgement_li),'networkTags':np.array(networkTags_li),'threatTags':np.array(threatTags_li),
                        'expired':np.array(expired_li),'continent':np.array(continent_li),'country':np.array(country_li),
                        'province':np.array(province_li),'city':np.array(city_li),'district':np.array(district_li),
                        'timeZone':np.array(timeZone_li),'organization':np.array(organization_li),'operator':np.array(operator_li)})
    agg_df = pd.concat([agg_df, info_df], axis=1)
    return agg_df

def ip_info_processing():
    """处理ip_info数据集
    """
    
    ip_info = pd.read_csv(RAW_IP_INFO_PATH)
    logger.info('ip_info shape <{}, {}>'.format(ip_info.shape[0], ip_info.shape[1]))
    logger.info(ip_info.columns)
    
    ip_info['location'] = ip_info['location'].apply(lambda x: eval(x))
    for col in ['continent', 'country', 'province', 'city', 'district', 'lng', 'lat', 'timeZone']:
        ip_info[col] = ip_info['location'].apply(lambda x: x[col])
     
    ip_info['asn'] = ip_info['asn'].apply(lambda x: eval(x))
    for col in ['number', 'organization', 'operator']:
        ip_info[col] = ip_info['asn'].apply(lambda x: x[col])

    ip_info['networkTags'] = ip_info['networkTags'].apply(lambda x:str(x).replace('[',''))
    ip_info['networkTags'] = ip_info['networkTags'].apply(lambda x:str(x).replace(']',''))
    ip_info['threatTags'] = ip_info['threatTags'].apply(lambda x:str(x).replace('[',''))
    ip_info['threatTags'] = ip_info['threatTags'].apply(lambda x:str(x).replace(']',''))

    info_cols = ['judgement','networkTags','threatTags','expired','continent','country','province','city',
            'district','timeZone','organization','operator']

    for col in info_cols:
        ip_info[col] = ip_info[col].fillna('##')
        ip_info[col] = label_encode(ip_info[col].astype(str))
        ip_info[col] = ip_info[col].astype(str)

    ip_info.to_csv(IP_INFO_PATH, sep=',', encoding='utf-8', index=False)

def other(agg_df):
    small_cols = ['judgement','expired','continent','timeZone']
    for col in tqdm(small_cols):
        logger.debug(col)
        agg_df[col] = agg_df[col].apply(lambda x: ' '.join([str(i) for i in x]))

        # CountVectorizer
        countv = CountVectorizer(analyzer='char', token_pattern=u"(?u)\b\w+\b")
        cv = countv.fit_transform(agg_df[col].fillna("##").values)
        cv_df = pd.DataFrame(cv.toarray())
        cv_df.columns = [col + '_cv_' + str(i) for i in range(cv_df.shape[1])]
        cv_df['rrname'] = agg_df['rrname']
        
        agg_df = agg_df.merge(cv_df, on=['rrname'], how='left')

    info_cols = ['judgement','networkTags','threatTags','expired','continent','country','province','city',
            'district','timeZone','organization','operator']
    big_cols = [i for i in info_cols if i not in small_cols]
    logger.info(big_cols)
    for col in tqdm(['rdatalist','bailiwicklist'] + big_cols):
        logger.debug(col)
        agg_df[col] = agg_df[col].apply(lambda x: ' '.join([str(i) for i in x]))

        # TfidfVectorizer
        tfidf = TfidfVectorizer()
        tf = tfidf.fit_transform(agg_df[col].fillna("##").values)

        # TruncatedSVD
        decom = TruncatedSVD(random_state=1024)
        decom_x = decom.fit_transform(tf)
        decom_feas = pd.DataFrame(decom_x)
        decom_feas.columns = [col + '_svd_'+str(i) for i in range(decom_feas.shape[1])]
        decom_feas['rrname'] = agg_df['rrname']

        agg_df = agg_df.merge(decom_feas, on=['rrname'], how='left')
    
    #agg_df['country_count'] = agg_df['country'].apply(lambda x: len(set(x)))
    return agg_df

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
        
    print("%s_train_score_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    return oof, pred

def lgb_model(x_train, y_train, x_test):
    return cv_model(lgb, x_train, y_train, x_test, "lgb")

def xgb_model(x_train, y_train, x_test):
    return  cv_model(xgb, x_train, y_train, x_test, "xgb")

def cat_model(x_train, y_train, x_test):
    return cv_model(CatBoostRegressor, x_train, y_train, x_test, "cat") 

def main():
    """主函数
    """

    # 处理ip_info数据集
    if not os.path.exists(IP_INFO_PATH):
        ip_info_processing()

    # 获取数据集
    train_dataset = pd.read_csv(TRAIN_DATASET_PATH)
    train_label   = pd.read_csv(TRAIN_LABEL_PATH)
    test_dataset  = pd.read_csv(TEST_DATASET_PATH)
    ip_info       = pd.read_csv(IP_INFO_PATH)
    logger.info('train_dataset shape <{}, {}>'.format(train_dataset.shape[0], train_dataset.shape[1]))
    logger.info('train_label   shape <{}, {}>'.format(train_label.shape[0], train_label.shape[1]))
    logger.info('test_dataset  shape <{}, {}>'.format(test_dataset.shape[0], test_dataset.shape[1]))
    logger.info('ip_info       shape <{}, {}>'.format(ip_info.shape[0], ip_info.shape[1]))
    
    # 特征工程
    train_dataset = feature_processing(train_dataset, ip_info)
    test_dataset  = feature_processing(test_dataset, ip_info)
    logger.info('train_dataset: ({}, {}), test_dataset: ({}, {}).'.format(
        train_dataset.shape[0], train_dataset.shape[1],
        test_dataset.shape[0], test_dataset.shape[1],))
    
    # 挑选特征
    logger.info([train_dataset.shape, test_dataset.shape])
    info_cols = ['judgement','networkTags','threatTags','expired','continent','country','province','city',
            'district','timeZone','organization','operator']
    features = [f for f in train_dataset.columns if f not in ['rdatalist','bailiwicklist']+info_cols]
    logger.info('features count {}'.format(len(features)))
    train_dataset = train_dataset[features]
    test_dataset  = test_dataset[features]
    logger.info([train_dataset.shape, test_dataset.shape])

    # 添加标签
    train_label.columns = ['rrname', 'label']
    train_dataset = train_dataset.merge(train_label, on='rrname', how='left')
    logger.info([train_dataset.shape])
    logger.info(train_dataset.info())

    # 保存数据集
    #train_dataset.to_csv(FEATURE_TRAIN_DATASET_PATH, sep=',', encoding='utf-8', index=False)
    #test_dataset.to_csv(FEATURE_TEST_DATASET_PATH, sep=',', encoding='utf-8', index=False)
    x_train = train_dataset.drop(['rrname', 'label'], axis=1)
    y_train = train_dataset['label']
    x_test  = test_dataset.drop(['rrname'], axis=1)
    cat_train, cat_test = cat_model(x_train, y_train, x_test)
    
    logger.info(cat_test.shape)
    
    test_dataset['label'] = cat_test
    test_dataset['label'] = test_dataset['label'].apply(lambda x: 1 if x > 0.35 else 0)
    test_dataset[['rrname','label']].to_csv("result.csv", index=False, header=False)

def debug():
    """debug测试
    """
    
    cols = ['a']
    num = 233
    for col in cols:
        exec('{}_li = num'.format(col), globals())
    #exec('a_li.append({})'.format(num), globals())
    logger.info(a_li)

if __name__ == '__main__':
    """程序入口
    """
    
    #os.system('chcp 936 & cls')
    logger.info('******** starting ********')
    start_time = time.time()

    main()
    #debug()

    end_time = time.time()
    logger.info('process spend {} s.\n'.format(round(end_time - start_time, 3)))

