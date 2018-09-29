# Python机器学习及实践个人笔记           
从零开始通往Kaggle竞赛之路
date: 2018-04-22 10:47:41  

## 第一章 简介篇
### 1、python数据分析常用包文件
- NumPy & Scipy
- Matplotlib
- Scikit-learn
- Pandas

#### 相关软件
- Anaconda
- IDLE
- IPython
- PyCharm
- Terminal
- DOS

### 2、python中的注意点
True  False
python3使用print()，而python2使用print，不用加括号输出。
幂指数运算**
没有自增和自减
主函数 if __name__ == '__main__':
逻辑运算 ans or not in
if else elif后面需要接冒号，可以不用括号
print输出连接用逗号隔开
math.exp函数（自然指数）
```
# method 1
import math
math.exp(2)

# method 2
from math import exp
exp(2)

# method 3
from math import exp as ep
ep(2)
```

### 3、良/恶性乳腺肿瘤预测
题目：数据总共四列，分别是编号、Clump Thickness（肿块厚度或密度）、Cell Size（细胞大小）、Type（类型，0为负分类即为良性阴性，1为恶性），根据测试集的数据预测该病人的肿瘤属于哪种类型？
数据：

CODE：[良/恶性乳腺肿瘤预测]()

## 第二章 基础篇
### 1、监督学习
分类学习：二分类、多类分类、多标签分类
线性分类器：一种假设特征与分类结果存在线性关系的模型。

#### 1-1、Logistic Function
逻辑斯蒂(Logistic)函数：g(z)=1/(1+e^-z)   值只取0、1。
模型的参数parameters：系数w和截距b

数据描述：1列检索的id，9列与肿瘤相关的医学特征(均被量化为1~10之间的数字)，以及1列表征肿瘤类型的数值(2指代良性，4指代恶性)。注意有缺失值。
数据集地址：https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data
数据集地址：https://www.kaggle.com/thulani96/breastcancer-wisconsin/data

通常情况下，25%的数据会作为测试集，其余75%的数据用于训练。
validation确认；批准；生效

假设我们手上有60个正样本，40个负样本，我们要找出所有的正样本，系统查找出50个，其中只有40个是真正的正样本，计算上述各指标。

TP: 将正类预测为正类数  40
FN: 将正类预测为负类数  20
FP: 将负类预测为正类数  10
TN: 将负类预测为负类数  30

准确率(accuracy) = 预测对的/所有 = (TP+TN)/(TP+FN+FP+TN) = 70%
精确率(precision) = TP/(TP+FP) = 80%
召回率(recall) = TP/(TP+FN) = 2/3
F值  = 正确率 * 召回率 * 2 / (正确率 + 召回率) （F 值即为正确率和召回率的调和平均值）
白话：精确率就是在所有预测为正例中有多少是预测正确的，召回率就是在全部本身就是正样本中有多少预测正确。

作者：Charles Xiao
链接：https://www.zhihu.com/question/19645541/answer/91694636
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

stochastic gradient descent 随机梯度下降参数估计法
机器学习里面，梯度下降法可以说是随处可见，虽然它不是什么高大上的机器学习算法，但是它却是用来解决机器学习算法的良药。我们经常会用到梯度下降法来对机器学习算法进行训练。
   在很多介绍梯度下降的书籍里，我们看到这样的几个英文单词缩写，BGD，SGD，MBGD。也就是批量梯度下降法BGD，随机梯度下降法SGD，小批量梯度下降法MBGD。

#### 1-2、Support Vector Classifier
支持向量机分类器根据训练样本的分布，搜索所有可能的线性分类器中最佳的那个。但不是绝对的最佳，如果位置的待测数据也如驯良数据一样分布，那么的确支持向量机找到的分类器就是最佳的。

数据描述：Scikit-learn内部集成的手写体数字图片数据集
使用基于线性假设的支持向量积分类器LinearSVC
from sklearn.svm import LinearSVC

召回率、准确率和F1指标最先适用于二分类任务，可以逐一评估某个类别的这三个性能指标：把所有其他的类别看作阴性（负）样本即可。

适用情况：海量甚至高维度的数据中，筛选对预测任务最为有效的少数`训练样本`。

#### 3、Naive Bayes
特别适合文本分类的任务。

数据描述：即时从互联网下载数据，20类新闻文本。固定的数据。
```
# 从sklearn.feature_extraction.text里导入用于文本特征向量转化模块。
from sklearn.feature_extraction.text import CountVectorizer
# 从sklearn.naive_bayes里导入朴素贝叶斯模型。
from sklearn.naive_bayes import MultinomialNB
```
#### 4、K近邻(分类)
待分类的样本在特征空间中距离最近的K个已标记样本作为参考，少数服从多数原则进行分类标记。

随着K的不同，我们会获得不同效果的分类器，即答案不同。

数据描述：鸢尾(Iris)数据集
该模型没有参数训练过程，属于无参数模型中非常简单一种。

#### 5、决策树
需要考虑特征节点的选取顺序：信息熵和基尼不纯性

题目：泰坦尼克号

- 选取特征：pclass、age、sex
- age有缺失值，用平均数或中位数填充
- sex和pclass类别型转数值特征，用0/1代替

X['age'].fillna(X['age'].mean(), inplace=True)

#### 6、集成模型(分类)
综合考虑多个分类期的预测结果从而做出决策。
##### 随机森林Random Forest Classifier
随机选取特征，搭建多个独立的分类模型，然后通过投票的方式，以少数服从多数做出决策。

##### 梯度提升决策树Gradient Tree Boosting
按照一定次序搭建多个分类模型，生成过程中都会尽可能降低整体集成模型在训练集上拟合误差。


gbc > rfc >单一决策树
经常使用随机森林分类模型作为基线系统Baseline System

### 回归预测
线性回归模型
线性回归器LinearRegression和SGDRegressor。
#### 评价
- LinearRegression模型自带的评估模块
- 从sklearn.metrics依次导入r2_score、mean_squared_error以及mean_absoluate_error用于回归性能的评估。
- 平均绝对误差和均方误差


##### 支持向量机(回归)
三种不同核函数：linear线性、ploy多项式、rbf径向基


K近邻(回归)
回归树：树叶节点的数据类型不是离散型，而是连续型。严格来讲不能成为“回归算法”


集成模型(回归) 极端随机森林Extremely Randomized Trees: 在分裂节点时，先随机收集一部分特征，然后利用

## 无监督学习
数据聚类，保留最具有区分性的低纬度特征
K均值(K-means)算法
# 从sklearn导入度量函数库metrics。
from sklearn import metrics
# 使用ARI进行KMeans聚类性能评估。
print metrics.adjusted_rand_score(y_test, y_pred)

轮廓系数凝聚度核分离度

特征降维：主成分分析
降低维度、信息冗余、PCA
矩阵线性相关，则秩为1




