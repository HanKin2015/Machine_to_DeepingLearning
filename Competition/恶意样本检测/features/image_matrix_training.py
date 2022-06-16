# -*- coding: utf-8 -*-
"""
文 件 名: training.py
文件描述: 图像矩阵训练模型
作    者: HeJian
创建日期: 2022.06.15
修改日期：2022.06.15
Copyright (c) 2022 HeJian. All rights reserved.
"""

from common import *

DATA_PATH                                = './data/'                                        # 数据路径
DATASET_PATH                             = './dataset/'                                     # 数据集路径
TRAIN_DATASET_PATH                       = DATASET_PATH+'train_dataset.csv'                 # 训练集样本数据集路径
MODEL_PATH                               = './model/'                                       # 模型路径
MALICIOUS_SAMPLE_DETECTION_MODEL_PATH    = MODEL_PATH+'malicious_sample_detection.model'    # 恶意样本检测训练模型路径
MALICIOUS_SAMPLE_DETECTION_SELECTOR_PATH = MODEL_PATH+'malicious_sample_detection.selector' # 恶意样本检测特征选择器路径
TRAIN_WHITE_IMAGE_MATRIX_PATH            = DATA_PATH+'train_white_image_matrix.csv'         # 训练集白样本图像矩阵数据集存储路径
TRAIN_BLACK_IMAGE_MATRIX_PATH            = DATA_PATH+'train_black_image_matrix.csv'         # 训练集黑样本图像矩阵数据集存储路径
TEST_IMAGE_MATRIX_PATH                   = DATA_PATH+'test_image_matrix.csv'                # 测试集样本图像矩阵数据集存储路径
MODEL_SCORE_PATH                         = MODEL_PATH+'score'                               # 模型分数路径
IAMGE_MATRIX_RFC_MODEL_PATH              = MODEL_PATH+'image_matrix_rfc.model'              # RF模型路径
IAMGE_MATRIX_XGB_MODEL_PATH              = MODEL_PATH+'image_matrix_xgb.model'              # XGB模型路径
IAMGE_MATRIX_LGB_MODEL_PATH              = MODEL_PATH+'image_matrix_lgb.model'              # LGB模型路径

# 创建模型文件夹
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
    
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

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logger.info([X_train.shape, X_test.shape, y_train.shape, y_test.shape])

    RFC = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    RFC.fit(X_train, y_train)
    y_pred = RFC.predict(X_test)

    score = model_score('RandomForestClassifier', y_test, y_pred)
    
    save_test_pred(X_test, y_test, y_pred, score)
    return RFC, score

def save_training_model(model, score):
    """保存训练模型
    """
    
    before_score = 0
    with open(MODEL_SCORE_PATH, 'r') as fd:
        before_score = fd.read()

    if score > float(before_score):
        buffer = pickle.dumps(model)
        with open(IAMGE_MATRIX_RFC_MODEL_PATH, "wb+") as fd:
            fd.write(buffer)
        with open(MODEL_SCORE_PATH, 'w') as fd:
            fd.write(str(score))

def main():
    # 获取数据集
    train_white_dataset = pd.read_csv(TRAIN_WHITE_IMAGE_MATRIX_PATH)
    train_black_dataset = pd.read_csv(TRAIN_BLACK_IMAGE_MATRIX_PATH)
    logger.info('train white dataset shape: ({}, {}).'.format(train_white_dataset.shape[0], train_white_dataset.shape[1]))
    logger.info('train black dataset shape: ({}, {}).'.format(train_black_dataset.shape[0], train_black_dataset.shape[1]))
    
    # 添加标签
    train_black_dataset['Label'] = 0
    train_white_dataset['Label'] = 1
    
    # 黑白样本合并
    train_dataset = pd.concat([train_black_dataset, train_white_dataset], ignore_index=True, sort=False)
    logger.info('train dataset shape: ({}, {}).'.format(train_dataset.shape[0], train_dataset.shape[1]))
    
    # 填充缺失值
    train_dataset = train_dataset.fillna(0)
    
    # 划分训练集和测试集
    X = train_dataset.drop(['Label', 'FileName'], axis=1).values
    y = train_dataset['Label'].values
        
    # 模型训练
    model, score = random_forest_model(X, y)
    save_training_model(model, score)

if __name__ == '__main__':
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))

