# 恶意样本检测

## 1、评分方法
black_is_black
black_is_white
white_is_black
white_is_white

### 精确率
precision = black_is_black / (black_is_black + white_is_black)

### 召回率
recall = black_is_black / (black_is_black + black_is_white)

### 准确率
accuracy = (black_is_black + white_is_white) / (black_is_black + black_is_white + white_is_black + white_is_white)

### 误报率
error_ratio = white_is_black / (white_is_black + white_is_white)

### 惩罚系数
alpha = 1.2

### 分数
score = recall - alpha * error_ratio

## 2、pe文件

### 参考资料
百度百科：https://baike.baidu.com/item/pe%E6%96%87%E4%BB%B6/6488140?fr=aladdin
神操作：教你用Python识别恶意软件：https://zhuanlan.zhihu.com/p/174260322

### 分析有用的特征
（时间戳）PE头里包括了时间戳字段，这个字段可以给出恶意软件作者编译文件的时间。通常恶意软件作者会使用伪造的值替换这个字段，但是有时恶意软件作者会忘记替换，就会发生这种情况。
（可选头）定义了PE文件中程序入口点的位置




