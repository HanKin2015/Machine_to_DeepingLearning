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
#ͨ��������֤���ò���
parameter_grid = [  {'kernel': ['linear'], 'C': [1, 10, 50, 600]},
                    {'kernel': ['poly'], 'degree': [2, 3]},
                    {'kernel': ['rbf'], 'gamma': [0.01, 0.001], 'C': [1, 10, 50, 600]},
                 ]
#������Ҫʹ�õ�ָ��
metrics = ['precision', 'recall_weighted']
#Ϊÿ��ָ���������ų�����
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
���ߣ�Զȥ�����ӻ� 
��Դ��CSDN 
ԭ�ģ�https://blog.csdn.net/u012967763/article/details/79231948 
��Ȩ����������Ϊ����ԭ�����£�ת���븽�ϲ������ӣ�

Ѱ�����Ų���