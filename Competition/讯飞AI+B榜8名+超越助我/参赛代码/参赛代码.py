############ 1、导入依赖包 ############
import numpy as np            # ndarray数组
import pandas as pd           # DataFrame表格
from sklearn import datasets  # 自带的数据集
from sklearn.model_selection import train_test_split # 随机划分为训练子集和测试子集
from sklearn.model_selection import cross_val_score  # 模型评价：训练误差和测试误差
from sklearn.feature_selection import SelectFromModel, chi2, SelectPercentile # 特征选择(三种方法)
from sklearn.metrics import roc_auc_score            # 评价指标
from sklearn.metrics import f1_score 
#from sklearn.cross_validation import StratifiedKFold # K折交叉验证
from sklearn.model_selection import StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier   # KNN
from sklearn.linear_model import LogisticRegression  # 逻辑斯特回归LR
from sklearn.tree import DecisionTreeClassifier      # DT
from sklearn.ensemble import RandomForestClassifier  # RFC随机森林分类
from sklearn.ensemble import RandomForestRegressor   # RFR随机森林回归
from sklearn.ensemble import ExtraTreesClassifier    # ETC极端随机树分类
from sklearn.ensemble import ExtraTreesRegressor     # ETR极端随机树回归
from sklearn.naive_bayes import GaussianNB           # GNB朴素贝叶斯
from sklearn import svm                            # SVM支持向量机
import xgboost as xgb                                # XGB
import lightgbm as lgb                               # LGB
from scipy import sparse                           # 对稀疏数据进行标准化        
from sklearn.preprocessing import OneHotEncoder, LabelEncoder  # 独热编码、标签编码
from sklearn.feature_extraction.text import CountVectorizer   # 文本特征提取
import time
import datetime
import os      
from pandas.api.types import is_bool_dtype, is_string_dtype  # 字段类型判断

import matplotlib as mpl
import matplotlib.pyplot as plt      # 作图
import seaborn as sns                # 作图
plt.style.use("fivethirtyeight")
sns.set_style({'font.sans-serif':['simhei','Arial']})
from IPython.display import display  # 输出语句

import warnings                      # 消除警告
warnings.filterwarnings("ignore")

#%matplotlib inline  
#import mxnet                          # 深度学习框架
import tensorflow as tf

from sys import version_info  # 检查Python版本
if version_info.major != 3:
    raise Exception('请使用 Python 3 来完成此项目')
	
############ 2、加载数据集 ############
if not os.path.exists('data'):
    raise Exception('请将数据集放在data文件夹里面')
train1 = pd.read_table('./data/round1_iflyad_train.txt')
train2 = pd.read_table('./data/round2_iflyad_train.txt')

train = pd.concat([train1, train2], axis=0, ignore_index=True)
test = pd.read_table('./data/round2_iflyad_test_feature.txt')

# 对训练集去重
train.drop_duplicates(subset=None, keep='first', inplace=False)
# 合并训练集，验证集
data = pd.concat([train,test],axis=0,ignore_index=True) 
# 原始特征
originFeature = data.columns.tolist()  
originFeature.remove('click')

############ 3、特征工程 ############
# 3-1、缺失值填充
data = data.fillna(-1)

# 3-2、bool布尔类型特征转换为int类型
for feat in originFeature:
    if is_bool_dtype(data[feat]):
        data[feat] = data[feat].astype(int)

# 3-3、字符类型特征处理
data['advert_industry_inner_0'] = data['advert_industry_inner'].apply(lambda x: x.split('_')[0])
data['advert_industry_inner_1'] = data['advert_industry_inner'].apply(lambda x: x.split('_')[1])
data['inner_slot_id_0'] = data['inner_slot_id'].apply(lambda x:x.split('_')[0])
data['model'] = data['make'].astype(str).values + '_' + data['model'].astype(str).values
data['osv'] = data['os'].astype(str).values + '_' + data['osv'].astype(str).values

# 3-4、时间time
data['day'] = data['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
data['hour'] = data['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))

# 3-5、城市city
data['county'] = data['city'].apply(lambda x: str(x)[5:7])
data['province'] = data['city'].apply(lambda x: str(x)[5:9])
data['city'] = data['city'].apply(lambda x: str(x)[5:11])
data['level'] = data['city'].apply(lambda x: str(x)[11:12])

# 新添的特征
addFeature = ['advert_industry_inner_0', 'inner_slot_id_0',  'county', 'level']

# 特征选择
advertFeature = ['adid', 'advert_id', 'orderid',  'advert_industry_inner', 'advert_name', 'campaign_id', 'creative_id', 'creative_type', 'creative_tp_dnf', 'creative_has_deeplink', 'creative_is_jump', 'creative_is_download']  # 没有后4个特征
mediaFeature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']   # 没有app_paid
contentFeature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'os_name', 'osv', 'os', 'make', 'model']  # 没有time
otherFeature = ['creative_width', 'creative_height', 'hour']

feature = advertFeature + mediaFeature + contentFeature + addFeature

# 标签编码（labelencoder）
for feat in feature:
    data[feat] = data[feat].map(dict(zip(data[feat].unique(), range(0, data[feat].nunique()))))

############ 4、分离训练集和测试集 ############
data['click'] = data.click.astype(int)
test = data[data.click == -1]      # 测试集
x_test = test.drop('click', axis=1)
y_test = test[['instance_id']]
y_test['predicted_score'] = 0

x_train = data[data.click != -1]  # 训练集
y_train = data[data.click != -1].click.values	

############ 5、对特征进行OneHot编码、user_tags文本特征提取并对稀疏数据进行标准化 ############
trainCsr = sparse.csr_matrix((len(train), 0))
testCsr = sparse.csr_matrix((len(x_test), 0))

print('Start OneHotEncoder!')
ohe = OneHotEncoder()
for feat in feature:
	ohe.fit(data[feat].values.reshape(-1, 1))
	trainCsr = sparse.hstack((trainCsr, ohe.transform(x_train[feat].values.reshape(-1, 1))), 'csr', 'bool')
	testCsr = sparse.hstack((testCsr, ohe.transform(test[feat].values.reshape(-1, 1))),  'csr', 'bool')
	
print('Start CountVectorizer!')
cv = CountVectorizer(min_df=20)
data['user_tags'] = data['user_tags'].astype(str)
cv.fit(data['user_tags'])
trainCsr = sparse.hstack((trainCsr, cv.transform(x_train['user_tags'].astype(str))), 'csr', 'bool')
testCsr = sparse.hstack((testCsr, cv.transform(x_test['user_tags'].astype(str))), 'csr', 'bool')

sparse.save_npz( './data/trainCsr.npz', trainCsr)
sparse.save_npz('./data/testCsr.npz', testCsr)

# 添加其他特征（int类型）
trainCsr = sparse.hstack((sparse.csr_matrix(x_train[otherFeature]), trainCsr), 'csr').astype('float32')
testCsr = sparse.hstack((sparse.csr_matrix(x_test[otherFeature]), testCsr), 'csr').astype('float32')

############ 6、使用卡方进行特征选择 ############# 
print('before feature select:', trainCsr.shape)
SP = SelectPercentile(chi2, percentile=96)
SP.fit(trainCsr, y_train)
trainCsr = SP.transform(trainCsr)
testCsr = SP.transform(testCsr)
print('feature select chi2:', trainCsr.shape)

############ 6、添加统计特征 ############# 
countFeatureList = []
for feat in feature:
    n = data[feat].nunique()
    if n > 5:
        newFeat = 'count_' + feat
        try:
            del data[newFeat]
        except:
            pass
        tmp = data.groupby(feat).size().reset_index().rename(columns={0: newFeat})
        data = data.merge(tmp, 'left', on=feat)
        countFeatureList.append(newFeat)
    else:
        print(feat, ':', n)
trainCount = data[countFeatureList][data.click != -1].values
testCount = data[countFeatureList][data.click == -1].values
trainCsr = sparse.hstack((trainCsr, trainCount), 'csr')
testCsr = sparse.hstack((testCsr, testCount), 'csr')

############ 7、添加部分特征的类别个数 ############# 
# adid广告id
uniqueFeatureList = []
adid_nuq=['model','make','os','city','province','user_tags','f_channel','app_id','carrier','nnt', 'devtype', 'app_cate_id','inner_slot_id']
for feat in adid_nuq:
    columsName = "adid_%s_nuq_num"%feat
    gp1 = data.groupby('adid')[feat].nunique().reset_index().rename(columns={feat: columsName})
    uniqueFeatureList.append(columsName)
    
    columsName = "%s_adid_nuq_num"%feat
    gp2=data.groupby(feat)['adid'].nunique().reset_index().rename(columns={'adid': columsName})
    uniqueFeatureList.append(columsName)
    data = pd.merge(data,gp1, how='left', on=['adid'])
    data = pd.merge(data,gp2, how='left', on=[feat])

# app_id媒体id
app_id_nuq=['model','make','os','city','province','user_tags','f_channel','carrier','nnt', 'devtype', 'app_cate_id','inner_slot_id']
for feat in app_id_nuq:
    columsName = "app_id_%s_nuq_num"%feat
    gp1=data.groupby('app_id')[feat].nunique().reset_index().rename(columns={feat: columsName})
    uniqueFeatureList.append(columsName)   
    
    columsName = "%s_app_id_nuq_num"%feat
    gp2=data.groupby(feat)['app_id'].nunique().reset_index().rename(columns={'app_id':columsName})
    uniqueFeatureList.append(columsName)   
    data=pd.merge(data,gp1,how='left',on=['app_id'])
    data=pd.merge(data,gp2,how='left',on=[feat])

train_unique = data[uniqueFeatureList][data.click != -1].values
test_unique = data[uniqueFeatureList][data.click == -1].values
trainCsr = sparse.hstack((trainCsr, train_unique), 'csr')
testCsr = sparse.hstack((testCsr, test_unique), 'csr')

############ 8、单一模型训练（LGBM） ############# 
def ModelTraining(trainCsr, testCsr, sed):
    LGB = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=141, reg_alpha=1, reg_lambda=2,
                         max_depth=-1, n_estimators=5000, objective='binary', min_child_weight=5, min_child_samples=35, 
                         subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                         learning_rate=0.02, random_state=sed)
    # 5折交叉检验
    skf = StratifiedKFold(n_splits=5, random_state=sed, shuffle=True)
    train_user = pd.Series()
    test_user = pd.Series(0, index=list(range(test.shape[0])))
    bestScore = []
    loss = 0
    for index, (trainIndex, testIndex) in enumerate(skf.split(trainCsr, y_train)):
        LGB.fit(trainCsr[trainIndex], y_train[trainIndex],
                  eval_names =['train','valid'],
                  eval_metric='logloss',
                  eval_set=[(trainCsr[trainIndex], y_train[trainIndex]),
                            (trainCsr[testIndex], y_train[testIndex])], early_stopping_rounds=100)
        bestScore.append(LGB.best_score_['valid']['binary_logloss'])
        print('bestScore:', bestScore)
        train_user = train_user.append(pd.Series(LGB.predict_proba(trainCsr[testIndex], num_iteration=LGB.best_iteration_)[:, 1],index=testIndex))
        test_user = test_user+pd.Series(LGB.predict_proba(testCsr, num_iteration=LGB.best_iteration_)[:, 1])
        loss += LGB.best_score_['valid']['binary_logloss']
    print('mean logloss:', loss/5)
    return (test_user/5).values

res = ModelTraining(trainCsr, testCsr, 520)

############ 9、保存结果 ############# 
print('mean:', res.mean())
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
y_test['predicted_score'] = y_test['predicted_score'] + res
if not os.path.exists('result'):
    os.mkdir('result')
y_test[['instance_id', 'predicted_score']].to_csv("./result/hankin_lgb_%s.csv" % now, index=False)
print('done!')