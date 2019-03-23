https://www.lintcode.com/ai


from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from loadFile import load_data
from plot_classifier import plot_classifier
import numpy as np
import matplotlib.pyplot as plt
 
input_file = 'data_multivar_imbalance.txt'
X, y = load_data(input_file)
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
#通过交叉验证设置参数
parameter_grid = [  {'kernel': ['linear'], 'C': [1, 10, 50, 600]},
                    {'kernel': ['poly'], 'degree': [2, 3]},
                    {'kernel': ['rbf'], 'gamma': [0.01, 0.001], 'C': [1, 10, 50, 600]},
                 ]
#定义需要使用的指标
metrics = ['precision', 'recall_weighted']
#为每个指标搜索最优超参数
for metric in metrics:
    print('Searching optimal hyperparameters for',metric)
    classifier = GridSearchCV(SVC(C=1),parameter_grid, cv=5, scoring=metric)
    classifier.fit(X_train, y_train)
    print("\nScores across the parameter grid:")
    for params, avg_score, _ in classifier.grid_scores_:
        print(params, '-->', round(avg_score, 3))
 
    print("\nHighest scoring parameter set:", classifier.best_params_)
 
    y_true, y_pred = y_test, classifier.predict(X_test)
    print("\nFull performance report:\n")
    print(classification_report(y_true, y_pred))
--------------------- 
作者：远去的栀子花 
来源：CSDN 
原文：https://blog.csdn.net/u012967763/article/details/79231948 
版权声明：本文为博主原创文章，转载请附上博文链接！

寻找最优参数