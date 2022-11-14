import pandas as pd
import os, gc, math, time
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD,SparsePCA
from sklearn.preprocessing import MinMaxScaler
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from log import *

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

train_pdns = pd.read_csv('fastflux_dataset/train/pdns.csv')
train_tags = pd.read_csv('fastflux_dataset/train/fastflux_tag.csv')
test_pdns = pd.read_csv('fastflux_dataset/test/pdns.csv')
ip_info = pd.read_csv('ip_info/ip_info.csv')
logger.info('read done')

train_pdns['rdata'] = train_pdns['rdata'].apply(lambda x:x[1:-1])
test_pdns['rdata'] = test_pdns['rdata'].apply(lambda x:x[1:-1])

train_pdns['time_diff'] = train_pdns['time_last'] - train_pdns['time_first']
test_pdns['time_diff'] = test_pdns['time_last'] - test_pdns['time_first']

data_pdns = pd.concat([train_pdns, test_pdns], axis=0, ignore_index=True)
del train_pdns, test_pdns
gc.collect()

data_pdns['rdata'] = data_pdns['rdata'].apply(lambda x:x.replace(' ',''))
logger.info('pre done')

agg_df = data_pdns.groupby(['rrname']).agg({'rdata':[list],
                                            'count':['sum','max','min'],
                                            'time_first':['max','min'],
                                            'time_last':['max','min'],
                                            'rrtype':['unique'],
                                            'bailiwick':[list,'nunique'],
                                            'time_diff':['sum','max','min','std']}).reset_index()

agg_df.columns = [''.join(col).strip() for col in agg_df.columns.values]
agg_df.reset_index(drop=True, inplace=True)

agg_df['rdatalist'] = agg_df['rdatalist'].apply(lambda x:','.join([i[1:-1] for i in x]).replace('\'','').split(','))


agg_df['rdatalist_count'] = agg_df['rdatalist'].apply(lambda x:len(x))
agg_df['rdatalist_nunique'] = agg_df['rdatalist'].apply(lambda x:len(set(x)))

agg_df['bailiwicklist_count'] = agg_df['bailiwicklist'].apply(lambda x:len(x))
logger.info('merage done')

# 自然数编码
def label_encode(series):
    unique = list(series.unique())
    # unique.sort()
    return series.map(dict(zip(unique, range(series.nunique()))))

for col in ['rrtypeunique']:
    agg_df[col] = label_encode(agg_df[col].astype(str))


ip_info['location'] = ip_info['location'].apply(lambda x: eval(x))
ip_info['continent'] = ip_info['location'].apply(lambda x: x['continent'])
ip_info['country'] = ip_info['location'].apply(lambda x: x['country'])
ip_info['province'] = ip_info['location'].apply(lambda x: x['province'])
ip_info['city'] = ip_info['location'].apply(lambda x: x['city'])
ip_info['district'] = ip_info['location'].apply(lambda x: x['district'])
ip_info['lng'] = ip_info['location'].apply(lambda x: x['lng'])
ip_info['lat'] = ip_info['location'].apply(lambda x: x['lat'])
ip_info['timeZone'] = ip_info['location'].apply(lambda x: x['timeZone'])

ip_info['asn'] = ip_info['asn'].apply(lambda x: eval(x))
ip_info['number'] = ip_info['asn'].apply(lambda x: x['number'])
ip_info['organization'] = ip_info['asn'].apply(lambda x: x['organization'])
ip_info['operator'] = ip_info['asn'].apply(lambda x: x['operator'])

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
logger.info('ip info done')

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
for items in tqdm(agg_df['rdatalist'].values):
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
            pass
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

info_df = pd.DataFrame({'judgement':np.array(judgement_li),'networkTags':np.array(networkTags_li),'threatTags':np.array(threatTags_li),
                        'expired':np.array(expired_li),'continent':np.array(continent_li),'country':np.array(country_li),
                        'province':np.array(province_li),'city':np.array(city_li),'district':np.array(district_li),
                        'timeZone':np.array(timeZone_li),'organization':np.array(organization_li),'operator':np.array(operator_li)})

agg_df = pd.concat([agg_df, info_df], axis=1)
logger.info('deal ip info done')

small_cols = ['judgement','expired','continent','timeZone']
for col in small_cols:
    print(col)
    agg_df[col] = agg_df[col].apply(lambda x: ' '.join([str(i) for i in x]))

    # CountVectorizer
    countv = CountVectorizer(analyzer='char',token_pattern=u"(?u)\b\w+\b")
    cv = countv.fit_transform(agg_df[col].fillna("##").values)
    cv_df = pd.DataFrame(cv.toarray())
    cv_df.columns = [col + '_cv_' + str(i) for i in range(cv_df.shape[1])]
    cv_df['rrname'] = agg_df['rrname']
    
    agg_df = agg_df.merge(cv_df, on=['rrname'], how='left')

big_cols = [i for i in info_cols if i not in small_cols]
for col in ['rdatalist','bailiwicklist'] + big_cols:
    print(col)
    agg_df[col] = agg_df[col].apply(lambda x: ' '.join([str(i) for i in x]))

    # TfidfVectorizer
    tfidf = TfidfVectorizer()
    tf = tfidf.fit_transform(agg_df[col].fillna("##").values)

    #### TruncatedSVD
    decom = TruncatedSVD(n_components=32, random_state=1024)
    decom_x = decom.fit_transform(tf)
    decom_feas = pd.DataFrame(decom_x)
    decom_feas.columns = [col + '_svd_'+str(i) for i in range(decom_feas.shape[1])]
    decom_feas['rrname'] = agg_df['rrname']

    agg_df = agg_df.merge(decom_feas, on=['rrname'], how='left')

train_tags.columns = ['rrname','label']
agg_df = agg_df.merge(train_tags, on=['rrname'], how='left')

agg_df = reduce_mem_usage(agg_df)

features = [f for f in agg_df.columns if f not in ['rrname','label','rdatalist','bailiwicklist']+info_cols]

train = agg_df[~agg_df.label.isnull()].reset_index(drop=True)
test = agg_df[agg_df.label.isnull()].reset_index(drop=True)

x_train = train[features]
x_test = test[features]

y_train = train['label']

del agg_df
gc.collect()

print(x_train.shape, x_train.shape)

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

cat_train, cat_test = cat_model(x_train, y_train, x_test)

test['label'] = cat_test
test['label'] = test['label'].apply(lambda x: 1 if x>0.35 else 0)
# 提交最终结果
test[['rrname','label']].to_csv("result.csv", index=False)