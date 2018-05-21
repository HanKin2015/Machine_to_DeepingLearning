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
题目：
数据：

CODE：良/恶性乳腺肿瘤预测
