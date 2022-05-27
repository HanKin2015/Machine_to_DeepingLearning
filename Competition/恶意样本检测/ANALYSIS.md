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
PE文件全解析：https://mp.weixin.qq.com/s/iNLTcmlGsw8jWtTtn3zXKQ

### 分析有用的特征
（时间戳）PE头里包括了时间戳字段，这个字段可以给出恶意软件作者编译文件的时间。通常恶意软件作者会使用伪造的值替换这个字段，但是有时恶意软件作者会忘记替换，就会发生这种情况。
（可选头）定义了PE文件中程序入口点的位置
（IAT表内容，其显示了这个恶意软件使用的库函数）这个输出对于恶意软件分析很有价值，因为它列出了恶意软件声明和将引用的丰富的函数数组。
（检查恶意软件的图片）要了解恶意软件是如何设计来捉弄攻击目标的，让我们看看在它的.rsrc节中所包含的图标。例如，恶意软件二进制文件常常被设计成伪装的Word文档、游戏安装程序、PDF文件等常用软件的图标来欺骗用户点击它们。
你还可以在恶意软件中找到攻击者自己感兴趣程序中的图像，例如攻击者为远程控制受感染机器而运行的网络攻击工具和程序。

### 多层感知器
根据维基百科的词条解释：
多层感知器（multilayer perceptron，MLP）是一种前馈人工神经网络模型，它将输入数据集映射为一组适当的输出集。MLP由有向图中的多层节点组成，每层节点都与下一层节点完全相连。除了输入节点之外，每个节点都是具有非线性激活功能的神经元（或处理单元）。MLP使用了反向传播（back propagation）这种监督学习技术（supervised learning technique）来训练神经网络。MLP是标准线性感知器的修改版，可以用来区分不能线性分离的那些数据。
