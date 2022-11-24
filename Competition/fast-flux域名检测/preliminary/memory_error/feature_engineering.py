# -*- coding: utf-8 -*-
"""
文 件 名: feature_engineering.py
文件描述: 特征工程
作    者: 重在参与快乐加倍队
创建日期: 2022.10.18
修改日期：2022.11.21

Copyright (c) 2022 ParticipationDoubled. All rights reserved.
"""

from common import *

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
                                            'count':['sum', 'max', 'min', 'std', 'mean'],
                                            'time_first':['max', 'min'],
                                            'time_last':['max', 'min'],
                                            'rrtype':['unique'],
                                            'bailiwick':[list, 'nunique'],
                                            'ttl':['max', 'min', 'std', 'mean'],
                                            'time_diff':['sum', 'max', 'min', 'std', 'mean']}).reset_index()

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
    
    # 域名中点数量
    agg_df['rrname_dot_count'] = agg_df['rrname'].apply(lambda x: x.count('.'))
    
    agg_df['rdatalist_count_rdatalist_nunique'] = agg_df['rdatalist_count'] / agg_df['rdatalist_nunique']
    agg_df['rdatalist_count_bailiwicklist_count'] = agg_df['rdatalist_count'] / agg_df['bailiwicklist_count']
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
    
    dataset['ttl'] = dataset['time_diff'] / dataset['count']
    return dataset

def feature_processing(dataset, ip_info):
    """特征工程
    """
    
    # 数据集预处理
    dataset = dataset_pre_processing(dataset)
    # 特征提取
    agg_df = feature_extraction(dataset)
    # pdns数据集和ip_info数据集结合
    agg_df = pdns_concat_ip_info(agg_df, ip_info)
    # ip_info文本特征处理
    agg_df = text_feture_processing(agg_df)
    # ip_info数值特征处理
    agg_df = numerical_feature_processing(agg_df)
    return agg_df

def pdns_concat_ip_info(agg_df, ip_info):
    """pdns数据集和ip_info数据集结合
    """
    
    logger.info([ip_info.shape])
    
    # 创建ip_info字典
    judgement_dict = dict(zip(ip_info['ip'], ip_info['judgement']))
    networkTags_dict = dict(zip(ip_info['ip'], ip_info['networkTags']))
    threatTags_dict = dict(zip(ip_info['ip'], ip_info['threatTags']))
    expired_dict = dict(zip(ip_info['ip'], ip_info['expired']))
    opensource_dict = dict(zip(ip_info['ip'], ip_info['openSource']))
    updateTime_diff_dict = dict(zip(ip_info['ip'], ip_info['updateTime_diff'])) 

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

    # 初始化列表
    judgement_li = []
    networkTags_li = []
    threatTags_li = []
    expired_li = []
    opensource_li = []
    updateTime_diff_li = []
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
    for items in tqdm(agg_df['rdatalist'].values):
        # 置空临时列表
        judgement_tmp = []
        networkTags_tmp = []
        threatTags_tmp = []
        expired_tmp = []
        opensource_tmp = []
        updateTime_diff_tmp = []
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
                opensource_tmp.append(int(opensource_dict[ip]))
                updateTime_diff_tmp.append(int(updateTime_diff_dict[ip]))
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
        opensource_li.append(opensource_tmp)
        updateTime_diff_li.append(updateTime_diff_tmp)
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
                        'expired':np.array(expired_li),'openSource':np.array(opensource_li),'updateTime_diff':np.array(updateTime_diff_li),
                        'continent':np.array(continent_li),'country':np.array(country_li),'countryCode':np.array(countryCode_li),
                        'province':np.array(province_li),'city':np.array(city_li),'district':np.array(district_li),
                        'timeZone':np.array(timeZone_li),'number':np.array(number_li),'organization':np.array(organization_li),'operator':np.array(operator_li)})
    agg_df = pd.concat([agg_df, info_df], axis=1)
    return agg_df

def ip_info_processing():
    """处理ip_info数据集
    """
    
    ip_info = pd.read_csv(RAW_IP_INFO_PATH)
    logger.info('ip_info shape <{}, {}>'.format(ip_info.shape[0], ip_info.shape[1]))
    logger.info(ip_info.columns)
    
    ip_info['location'] = ip_info['location'].apply(lambda x: eval(x))
    for col in ['continent', 'country', 'countryCode', 'province', 'city', 'district', 'lng', 'lat', 'timeZone']:
        ip_info[col] = ip_info['location'].apply(lambda x: x[col])
     
    ip_info['asn'] = ip_info['asn'].apply(lambda x: eval(x))
    for col in ['number', 'organization', 'operator']:
        ip_info[col] = ip_info['asn'].apply(lambda x: x[col])

    ip_info['networkTags'] = ip_info['networkTags'].apply(lambda x:str(x).replace('[',''))
    ip_info['networkTags'] = ip_info['networkTags'].apply(lambda x:str(x).replace(']',''))
    ip_info['threatTags'] = ip_info['threatTags'].apply(lambda x:str(x).replace('[',''))
    ip_info['threatTags'] = ip_info['threatTags'].apply(lambda x:str(x).replace(']',''))

    info_cols = ['judgement','networkTags','threatTags','expired','continent','country',
            'countryCode','province','city','district','timeZone','organization','operator']

    # 填充缺失值
    for col in info_cols:
        ip_info[col] = ip_info[col].fillna('##')
        ip_info[col] = label_encode(ip_info[col].astype(str))
        ip_info[col] = ip_info[col].astype(str)

    ip_info['openSource'] = ip_info['openSource'].apply(lambda x: 1 if len(str(x)) > 2 else 0)
    ip_info['firstFoundTime'] = ip_info['firstFoundTime'].fillna('1970-01-01T08:00:00')
    ip_info['firstFoundTime'] = ip_info['firstFoundTime'].apply(lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%dT%H:%M:%S"))))
    ip_info['updateTime'] = ip_info['updateTime'].fillna('1970-01-01T08:00:00')
    ip_info['updateTime'] = ip_info['updateTime'].apply(lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%dT%H:%M:%S"))))
    ip_info['updateTime_diff'] = ip_info['updateTime'] - ip_info['firstFoundTime']

    ip_info.to_csv(IP_INFO_PATH, sep=',', encoding='utf-8', index=False)

def text_feture_processing(agg_df):
    """文本特征处理
    """
    
    # 低维度
    low_dimension_cols = ['judgement', 'expired', 'continent', 'timeZone']
    for col in tqdm(low_dimension_cols):
        logger.debug(col)
        agg_df[col] = agg_df[col].apply(lambda x: ' '.join([str(i) for i in x]))

        # CountVectorizer
        countv = CountVectorizer(analyzer='char', token_pattern=u"(?u)\b\w+\b")
        cv = countv.fit_transform(agg_df[col].fillna("##").values)
        cv_df = pd.DataFrame(cv.toarray())
        cv_df.columns = [col + '_cv_' + str(i) for i in range(cv_df.shape[1])]
        cv_df['rrname'] = agg_df['rrname']
        
        agg_df = agg_df.merge(cv_df, on=['rrname'], how='left')

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
        tf = tfidf.fit_transform(agg_df[col].fillna("##").values)

        # TruncatedSVD降维
        #decom = TruncatedSVD(n_components=32, random_state=1024)
        decom = TruncatedSVD(random_state=1024)
        decom_x = decom.fit_transform(tf)
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
    agg_df['updateTime_diff_max'] = agg_df['updateTime_diff'].apply(lambda x: max(x) if x else 0)
    agg_df['updateTime_diff_min'] = agg_df['updateTime_diff'].apply(lambda x: min(x) if x else 0)
    agg_df['updateTime_diff_mean'] = agg_df['updateTime_diff'].apply(lambda x: np.mean(x) if x else 0)
    agg_df['updateTime_diff_sum'] = agg_df['updateTime_diff'].apply(lambda x: sum(x) if x else 0)
    agg_df['updateTime_diff_std'] = agg_df['updateTime_diff'].apply(lambda x: np.std(x) if x else 0)
    agg_df['ip_asn'] = agg_df['rdatalist_nunique'] / agg_df['number_count']
    return agg_df

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
    
    # 合并训练集和测试集
    dataset = pd.concat([train_dataset, test_dataset], axis=0, ignore_index=True)
    del train_dataset, test_dataset
    gc.collect()
    
    # 特征工程
    dataset = feature_processing(dataset, ip_info)
    logger.info('dataset: ({}, {}).'.format(dataset.shape[0], dataset.shape[1]))

    # 调用的聚合函数可能存在NaN值
    dataset = dataset.fillna(0)

    # 添加标签
    train_label.columns = ['rrname', 'label']
    dataset = dataset.merge(train_label, on='rrname', how='left')
    logger.info([dataset.shape])

    # 分离训练集和测试集
    train_dataset = dataset[~dataset.label.isnull()].reset_index(drop=True)
    test_dataset  = dataset[dataset.label.isnull()].reset_index(drop=True)
    logger.info([train_dataset.shape, test_dataset.shape])
    
    # 挑选特征
    info_cols = ['judgement','networkTags','threatTags','expired','openSource',
            'updateTime_diff','continent','country','countryCode','province','city',
            'district','timeZone','number','organization','operator']
    features = [f for f in dataset.columns if f not in ['rrname', 'label', 'rdatalist', 'bailiwicklist', 'openSource_sum']+info_cols]
    train_dataset = train_dataset[features+['rrname', 'label']]
    test_dataset = test_dataset[features+['rrname']]
    logger.info([train_dataset.shape, test_dataset.shape])
    logger.info(train_dataset.info())
    logger.info('features count {}'.format(len(features)))

    # 保存数据集
    train_dataset.to_csv(FEATURE_TRAIN_DATASET_PATH, sep=',', encoding='utf-8', index=False)
    test_dataset.to_csv(FEATURE_TEST_DATASET_PATH, sep=',', encoding='utf-8', index=False)

if __name__ == '__main__':
    """程序入口
    """
    
    #os.system('chcp 936 & cls')
    logger.info('******** starting ********')
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.\n'.format(round(end_time - start_time, 3)))

