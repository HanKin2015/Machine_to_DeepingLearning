# -*- coding: utf-8 -*-
"""
文 件 名: feature_engineering.py
文件描述: 特征工程
作    者: 重在参与快乐加倍队
创建日期: 2022.10.18
修改日期：2022.11.24

Copyright (c) 2022 ParticipationDoubled. All rights reserved.
"""

from common import *

def dataset_pre_processing(dataset):
    """数据集预处理
    """
    
    # 去掉列表的左右中括号
    dataset['rdata'] = dataset['rdata'].apply(lambda x:x[1:-1])

    # 去掉空格
    dataset['rdata'] = dataset['rdata'].apply(lambda x:x.replace(' ', ''))

    # 时间转换成天数
    dataset['time_first'] = dataset['time_first'].apply(lambda x: int(x / 3600 / 24))
    dataset['time_last'] = dataset['time_last'].apply(lambda x: int(x / 3600 / 24))

    # 时间差
    dataset['time_diff'] = dataset['time_last'] - dataset['time_first']
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
                                            'count':['sum', 'max', 'min', 'std'],
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
    
    # 域名长度
    agg_df['rrname_length'] = agg_df['rrname'].apply(lambda x: len(x))
    
    # 域名中数字数量
    agg_df['rrname_number_count'] = agg_df['rrname'].apply(lambda x: len([ch for ch in x if ch.isdigit()]))
    return agg_df
    
def feature_processing(dataset):
    """特征工程
    """
    
    show_memory_info('feature_processing initial')
    # 数据集预处理
    dataset = dataset_pre_processing(dataset)
    show_memory_info('dataset_pre_processing done')

    # 聚合数据，特征初提取
    agg_df = feature_extraction(dataset)
    logger.info('agg_df shape <{}, {}>'.format(agg_df.shape[0], agg_df.shape[1]))
    del dataset
    show_memory_info('feature_processing done')
    
    # pdns数据集和ip_info数据集结合
    agg_df = pdns_concat_ip_info(agg_df)
    show_memory_info('pdns_concat_ip_info done')
    
    # ip_info文本特征处理
    agg_df = text_feture_processing(agg_df)
    show_memory_info('text_feture_processing done')
    
    # ip_info数值特征处理
    agg_df = numerical_feature_processing(agg_df)
    logger.debug(agg_df.columns)
    
    # 调用的聚合函数可能存在NaN值
    agg_df = agg_df.fillna(0)
    
    # 挑选特征
    info_cols = ['judgement','networkTags','threatTags','expired','openSource',
            'continent','country','countryCode','province','city',
            'district','timeZone','number','organization','operator']
    features = [f for f in agg_df.columns if f not in ['rrname', 'label', 'rdatalist', 'bailiwicklist', 'openSource_sum']+info_cols]
    agg_df = agg_df[features+['rrname']]
    logger.info('agg_df shape ({}, {})'.format(agg_df.shape[0], agg_df.shape[1]))
    #logger.debug(agg_df.info())
    logger.info('features count {}'.format(len(features)))
    return agg_df

def pdns_concat_ip_info(agg_df):
    """pdns数据集和ip_info数据集结合
    """
    
    show_memory_info('pdns_concat_ip_info initial')
    ip_info = pd.read_csv(IP_INFO_PATH)
    logger.info('ip_info shape <{}, {}>'.format(ip_info.shape[0], ip_info.shape[1]))
    
    # 创建ip_info字典
    judgement_dict = dict(zip(ip_info['ip'], ip_info['judgement']))
    networkTags_dict = dict(zip(ip_info['ip'], ip_info['networkTags']))
    threatTags_dict = dict(zip(ip_info['ip'], ip_info['threatTags']))
    expired_dict = dict(zip(ip_info['ip'], ip_info['expired']))
    openSource_dict = dict(zip(ip_info['ip'], ip_info['openSource']))

    continent_dict = dict(zip(ip_info['ip'], ip_info['continent']))
    country_dict = dict(zip(ip_info['ip'], ip_info['country']))
    countryCode_dict = dict(zip(ip_info['ip'], ip_info['countryCode']))
    province_dict = dict(zip(ip_info['ip'], ip_info['province']))
    city_dict = dict(zip(ip_info['ip'], ip_info['city']))
    district_dict = dict(zip(ip_info['ip'], ip_info['district']))
    timeZone_dict = dict(zip(ip_info['ip'], ip_info['timeZone']))

    number_dict = dict(zip(ip_info['ip'], ip_info['number']))
    organization_dict = dict(zip(ip_info['ip'], ip_info['organization']))
    operator_dict = dict(zip(ip_info['ip'], ip_info['operator']))
    
    show_memory_info('gc ip_info before')
    del ip_info
    gc.collect()
    show_memory_info('gc ip_info after')

    # 初始化列表
    judgement_li = []
    networkTags_li = []
    threatTags_li = []
    expired_li = []
    openSource_li = []
    continent_li = []
    country_li = []
    countryCode_li = []
    province_li = []
    city_li = []
    district_li = []
    timeZone_li = []
    number_li = []
    organization_li = []
    operator_li = []

    not_found_ip_count = 0
    #agg_df['rdatalist'] = agg_df['rdatalist'].apply(lambda x: eval(x))
    for items in tqdm(agg_df['rdatalist'].values):
        # 置空临时列表
        judgement_tmp = []
        networkTags_tmp = []
        threatTags_tmp = []
        expired_tmp = []
        openSource_tmp = []
        continent_tmp = []
        country_tmp = []
        countryCode_tmp = []
        province_tmp = []
        city_tmp = []
        district_tmp = []
        timeZone_tmp = []
        number_tmp = []
        organization_tmp = []
        operator_tmp = []
        
        for ip in items:
            try:
                judgement_tmp.append(judgement_dict[ip])
                networkTags_tmp.append(networkTags_dict[ip])
                threatTags_tmp.append(threatTags_dict[ip])
                expired_tmp.append(expired_dict[ip])
                openSource_tmp.append(int(openSource_dict[ip]))
                continent_tmp.append(continent_dict[ip])
                country_tmp.append(country_dict[ip])
                countryCode_tmp.append(countryCode_dict[ip])
                province_tmp.append(province_dict[ip])
                city_tmp.append(city_dict[ip])
                district_tmp.append(district_dict[ip])
                timeZone_tmp.append(timeZone_dict[ip])
                number_tmp.append(number_dict[ip])
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
        openSource_li.append(openSource_tmp)
        continent_li.append(continent_tmp)
        country_li.append(country_tmp)
        countryCode_li.append(countryCode_tmp)
        province_li.append(province_tmp)
        city_li.append(city_tmp)
        district_li.append(district_tmp)
        timeZone_li.append(timeZone_tmp)
        number_li.append(number_tmp)
        organization_li.append(organization_tmp)
        operator_li.append(operator_tmp)
    
    logger.info('here is {} ips which are not found'.format(not_found_ip_count))

    # 将ip_info提取的特征添加到数据集中
    info_df = pd.DataFrame({'judgement':np.array(judgement_li),'networkTags':np.array(networkTags_li),'threatTags':np.array(threatTags_li),
                        'expired':np.array(expired_li),'openSource':np.array(openSource_li),
                        'continent':np.array(continent_li),'country':np.array(country_li),'countryCode':np.array(countryCode_li),
                        'province':np.array(province_li),'city':np.array(city_li),'district':np.array(district_li),
                        'timeZone':np.array(timeZone_li),'number':np.array(number_li),'organization':np.array(organization_li),'operator':np.array(operator_li)})
    agg_df = pd.concat([agg_df, info_df], axis=1)
    
    show_memory_info('gc info_df before')

    del info_df
    gc.collect()
    show_memory_info('gc info_df after')
    return agg_df

def save_training_model(model, score, model_path=BASELINE_MODEL_PATH, score_path=BASELINE_MODEL_SCORE_PATH):
    """保存训练模型
    """
    
    before_score = 0
    if os.path.exists(score_path):
        buffer = pickle.dumps(model)
        with open(model_path, "wb+") as fd:
            fd.write(buffer)
        with open(score_path, 'w') as fd:
            fd.write(str(score))

def text_feture_processing(agg_df):
    """文本特征处理
    """
    
    show_memory_info('low_dimension_cols before')
    # 低维度
    low_dimension_cols = ['judgement', 'expired', 'continent', 'timeZone']
    for col in tqdm(low_dimension_cols):
        logger.debug(col)
        agg_df[col] = agg_df[col].apply(lambda x: ' '.join([str(i) for i in x]))

        # CountVectorizer
        countv = CountVectorizer(analyzer='char', token_pattern=u"(?u)\b\w+\b")
        countv_file_path = './model/countv_{}.pkl'.format(col)
        if os.path.exists(countv_file_path):
            with open(countv_file_path, 'rb') as fd:
                buffer = fd.read()
            vocabulary_ = pickle.loads(buffer)
            countv = CountVectorizer(vocabulary=vocabulary_)
            cv = countv.transform(agg_df[col].fillna("##").values)
        else:
            cv = countv.fit_transform(agg_df[col].fillna("##").values)
            buffer = pickle.dumps(countv.vocabulary_)
            with open(countv_file_path, "wb+") as fd:
                fd.write(buffer)
                
        cv_df = pd.DataFrame(cv.toarray())
        cv_df.columns = [col + '_cv_' + str(i) for i in range(cv_df.shape[1])]
        cv_df['rrname'] = agg_df['rrname']
        
        agg_df = agg_df.merge(cv_df, on=['rrname'], how='left')

    show_memory_info('high_dimension_cols before')
    info_cols = ['judgement', 'networkTags', 'threatTags', 'expired', 'continent', 'country',
            'countryCode', 'province', 'city', 'district', 'timeZone', 'organization', 'operator']
    # 高维度
    high_dimension_cols = [i for i in info_cols if i not in low_dimension_cols]
    logger.info(high_dimension_cols)
    for col in tqdm(['rdatalist','bailiwicklist'] + high_dimension_cols):
        logger.debug(col)
        agg_df[col] = agg_df[col].apply(lambda x: ' '.join([str(i) for i in x]))

        # TfidfVectorizer
        tfidf = TfidfVectorizer()
        tfidf_file_path = './model/tfidf_{}.pkl'.format(col)
        if os.path.exists(tfidf_file_path):
            with open(tfidf_file_path, 'rb') as fd:
                buffer = fd.read()
            tfidf = pickle.loads(buffer)
            tf = tfidf.transform(agg_df[col].fillna("##").values)
        else:
            tf = tfidf.fit_transform(agg_df[col].fillna("##").values)
            buffer = pickle.dumps(tfidf)
            with open(tfidf_file_path, "wb+") as fd:
                fd.write(buffer)

        # TruncatedSVD降维
        #decom = TruncatedSVD(n_components=32, random_state=1024)
        decom = TruncatedSVD(random_state=1024)
        decom_file_path = './model/decom_{}.pkl'.format(col)
        if os.path.exists(decom_file_path):
            with open(decom_file_path, 'rb') as fd:
                buffer = fd.read()
            decom = pickle.loads(buffer)
            decom_x = decom.transform(tf)
        else:
            decom_x = decom.fit_transform(tf)
            buffer = pickle.dumps(decom)
            with open(decom_file_path, "wb+") as fd:
                fd.write(buffer)
        
        decom_feas = pd.DataFrame(decom_x)
        decom_feas.columns = [col + '_svd_'+str(i) for i in range(decom_feas.shape[1])]
        decom_feas['rrname'] = agg_df['rrname']

        agg_df = agg_df.merge(decom_feas, on=['rrname'], how='left')

    return agg_df

def numerical_feature_processing(agg_df):
    """数值特征处理
    """

    agg_df['country_count'] = agg_df['country'].apply(lambda x: len(set(x)))
    agg_df['number_count'] = agg_df['number'].apply(lambda x: len(set(x)))
    agg_df['openSource_sum'] = agg_df['openSource'].apply(lambda x: sum(x))
    agg_df['ip_asn'] = agg_df['rdatalist_nunique'] / agg_df['number_count']
    return agg_df
    
def debug():
    """调试函数，此文件主要给外部调用使用
    """

    show_memory_info('initial')
    # 获取数据集
    train_dataset = pd.read_csv(TRAIN_DATASET_PATH)
    logger.info('train_dataset shape <{}, {}>'.format(train_dataset.shape[0], train_dataset.shape[1]))

    # 特征工程
    train_dataset = feature_processing(train_dataset)
    logger.info('train_dataset: ({}, {}).'.format(train_dataset.shape[0], train_dataset.shape[1]))

    # 给训练集添加标签
    train_label   = pd.read_csv(TRAIN_LABEL_PATH)
    logger.info('train_label shape <{}, {}>'.format(train_label.shape[0], train_label.shape[1]))
    train_label.columns = ['rrname', 'label']
    dataset = dataset.merge(train_label, on='rrname', how='left')
    logger.info([dataset.shape])
    del train_label
    gc.collect()
    
    # 保存数据集
    train_dataset.to_csv(FEATURE_TRAIN_DATASET_PATH, sep=',', encoding='utf-8', index=False)

if __name__ == '__main__':
    """程序入口
    """
    
    #os.system('chcp 936 & cls')
    logger.info('******** starting ********')
    start_time = time.time()

    debug()

    end_time = time.time()
    logger.info('process spend {} s.\n'.format(round(end_time - start_time, 3)))

