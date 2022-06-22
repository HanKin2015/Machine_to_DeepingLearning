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

## 3、经验教训
- 注意提交结果可能不包含头部，即列名
- 居然在最后关头才发现文件名没有获取全
- 还是建议使用logging库保存日志，方便后续查找，这是一个漫长的比赛
- 注意模型保存到本地，使用固定随机数的方案不是很保险吧
- k交叉验证
- 使用全部特征会带来过拟合，缺少特征会欠拟合
- 87.03的模型最终得到90.80分数，90.11的模型最终得到90.77分数，说明不能一味的最求高分模型，可能过拟合



## 4、特征工程
恶意程序的检测过程是个典型的二分类问题，工业界和学术届经常使用的方法包括多层感知机（MLP）、卷积神经网络（CNN）、梯度提升决策树（GBDT）和XGBoost。

通过md5值分析不现实
```
md5值相同：
6：39
5：51

错误值分类：
0：9899  
1：1     f31a0a1f7f97ec5e845d50daa5363064（1）
2：3     018b28bffc4299ebf859a1cc27425ac1（0）    eac7606f578658c0e1c12d093be5bb64（1）    ee7dd7c4e288c2f5d44f77ef7a87fd64（1）
3：1     e63fffab440d9011b2eb8bb6ff767364（1）
4：0
5：51    c0fc377ea0b6d4e6cbc12827d2fe1546（0，只有3个是1）
6：41    8eec510e57f5f732fd2cce73df7b73ef（0，只有3个是1）    f3b9f713c4d6f3355e25fb2be47d6ad1（0）    463b726d9ed494b98fba88f02fad2c9d（0） 0141f601f68ab16936797ef54dda53af（0）
7：4     00b34c3d99e56ec8279cb393a295df39（0）     01401fa8db30a478ae65e5f9250053af（0）     020ebcff7a9fad60ba043d89a2c633af（0） 02efba69a72be68b896f9679d65f6baf	（0）
```


# <center>重新获取数据特征</center>

- 搞了几天分数愣是没有提高，最高分数居然是全部特征训练，删减特征反而低了
- 只眼睁睁看着一个个大佬在提高分数
- 20220608重新来过
- 拼命地尽可能多的提取特征，别人有的我要有，别人没有的我也要有
- 前期整体框架可以借鉴参考
- 转换思想，数据集是自己通过pefile提取出来的，提供的数据实实在在的是二进制可执行的文件，不用怀疑
- 不是pe文件可以删除，不在考虑范围内，不用怀疑数据异常值



https://www.malware-traffic-analysis.net/
https://amzn.to/2EpqJS2
https://github.com/chihebchebbi/Mastering-Machine-Learning-for-Penetration-Testing/tree/master/Chapter03



什么是文件扩展名 LE?
Microsoft Corporation 在最初发行 DOS/4GW PM 32X DOS Extender Executable Image 时开发了MS-DOS文件类型。 内部网站统计显示，LE 文件最受United States和运行Windows 10 操作系统的用户欢迎。 这些用户中的大多数正在运行Google Chrome web浏览器。

NE （Win16 Windows 3.x文件格式）
播报 编辑 上传视频
本词条缺少概述图，补充相关内容使词条更完整，还能快速升级，赶紧来编辑吧！
NE是Win16 Windows 3.x文件的格式。
为了保持对DOS的兼容性和保证Windows的需要，在Win 3.x中出现的NE格式的可执行文件中保留了MZ格式的头（具体原因后面会说，这里就不赘述了），同时NE文件又加了一个自己的头，之后才是可执行文件的可执行代码。Win 3.x中的16位Windows程序或OS/2程序都有可能是NE格式的。NE类型包括了.exe、.dll、.drv和.fon四种类型的文件。NE格式的关键特性是：它把程序代码、数据、资源隔离在不同的可加载区块中。它也藉由符号输入和输出，实现所谓的执行时期动态链接。16位的NE格式文件装载程序（NE Loader）读取部分磁盘文件，并生成一个完全不同的数据结构，在内存中建立模块。当代码或数据需要装入时，装载程序必须从全局内存中分配出一块，查找原始数据在文件的什么地方，找到位置后再读取原始的数据，最后再进行一些修整。还有，每一个16位的Module要负责记住现在使用的所有段选择符，该选择符表示该段是否已经被抛弃等等。


https://zhuanlan.zhihu.com/p/59768106


```
%WINDIR%\system32
\SystemRoot\System32
C:\WINDOWS\system32
:\Windows\System32
%windir%\system32
System32\
%SystemRoot%\System32

download
wget  curl
http://

ftp
smtp

```


这次比赛确实属于入门很容易的，网上有大量的资源，非常适合初学者了解ai的流程，也真正响应了公司举办比赛的初衷。不过我想说的是简单并不代表真的简单，如果要做到终端然后再做好真的很不容易，简单举几个栗子，大家有兴趣也可以一同思考一下呀，比如大家抽取特征保存的的模型是不是都快百兆级别了，而且这才是1w训练样本哦，如果千万、亿级别以上又当如何？能放到端侧运行吗？资源占用情况？其次训练时间，超大量的样本训练ai，是得设计分布式训练的，资源如何调度？第三，加壳(几十种加壳类别)、特殊pe格式(GO、Pyinstaller、.net、安装包)怎么处理？保证误报(万分之一)又该怎么解决？等等问题，因此XXX目前还是国内唯一能把PE AI检测做到终端的主流安全厂商。

 



使用lief分析的pe文件预测：https://github.com/elastic/ember/blob/master/ember/features.py
imphash分析：https://malwology.com/2018/08/24/python-for-malware-analysis-getting-started/
上面的翻译：https://www.cnblogs.com/17bdw/p/9894947.html
时间戳获取地区：https://cloud.tencent.com/developer/article/1927823

论文解读：https://cloud.tencent.com/developer/article/1910311
https://github.com/eastmountyxz/AI-Security-Paper
pefile官方文档：https://pefile.readthedocs.io/en/latest/modules/pefile.html
找到一个github，作者根据《基于数据科学的恶意软件分析》学习：https://github.com/hija/MalwareDataScience/blob/master/chapter08/MalwareClassifier.ipynb
可以找到很多关于pefile的代码：https://codingdict.com/sources/py/pefile/17317.html
https://vimsky.com/examples/detail/python-attribute-pefile.DIRECTORY_ENTRY.html

机器学习方法检测恶意文件：https://blog.csdn.net/zourzh123/article/details/81607330

# 模型融合
https://blog.csdn.net/weixin_52561314/article/details/124480003

#方向又错了
其实也不怪，主要是找不到可以优化的方法了。不过最近又发现了新的方法。

基于深度学习的恶意软件分类器：https://bbs.pediy.com/thread-271558-1.htm

## 原理
通过将程序的16进制序列按照一定的规则转换为灰度图作为输入数据，随后使用VGG16深度卷积神经网络作为模型，利用深度卷积神经网络在图像识别上的准确性来来构建恶意程序分类器。该实验将会在两个公开数据集上进行测试，两个公开数据集的数据分别属于不同的恶意程序家族。因此，构建的分类器要解决的问题将是一个多分类问题。
Pytorch版本：1.8.1
CUDA版本：11.4

灰度图的生成：《Malware Images: Visualization and Automatic  Classification》
https://www.researchgate.net/profile/Shanmugavadivel-Karthikeyan/publication/228811247_Malware_Images_Visualization_and_Automatic_Classification/links/0deec53bee6c992ef1000000/Malware-Images-Visualization-and-Automatic-Classification.pdf

当程序被编译器编译完成以后，将会以二进制的形式保存在磁盘中。但是程序解析的时候，都是按照一个字节（8比特）来解析，所以解析器一般是按照16进制形式展现一个程序的二进制文件。

而每个字节的大小在0-255的范围内，这个范围和像素点的范围是一样的。因此，可以考虑将程序的每个字节当成一个像素点，转换成一个二维的数组，然后生成相应的灰度图。

这里就有一个问题，每个程序的字节数是不同的。因此，需要按照一定的规则来指定生成的图像的宽和高，作者在论文中给出的参考：
![](https://bbs.pediy.com/upload/attach/202202/835440_UYUB8CGVDBBNCJC.jpg)

根据文件大小的不同来固定生成的图像的宽度，剩余的字节就填充到高度去，以此，来生成相应的灰度图来表征程序。此外，作者认为，生成灰度图的时候，应当把程序的PE头部分删除，只留下节区，因为节区中的内容才保存了程序要执行的指令，使用到的数据等到。如下图所示，可以看到不同节区生成的灰度图的纹理存在着明显的不同。

最终生成的灰度图如下图所示，其中上面三张图是其中一个类别的恶意程序生成的灰度图，而下面的三张图则是另一类别恶意程序生成的灰度图。从中可以看到，不同家族的恶意程序所生成的灰度图的纹理存在非常明显的差异，这就给模型的训练带来的便利。




利用机器学习进行恶意代码分类：https://zhuanlan.kanxue.com/article-13391.htm
方法简述
Kaggle比赛中最重要的环节就是特征工程，特征的好坏直接决定了比赛成绩。在这次Kaggle的比赛中冠军队伍选取了三个“黄金”特征：恶意代码图像、OpCode n-gram和Headers个数，其他一些特征包括ByteCode n-gram，指令频数等。机器学习部分采用了随机森林算法，并用到了xgboost和pypy加快训练速度。

本文主要关注恶意代码图像和OpCode n-gram，以及随机森林算法的应用。
《基于纹理指纹的恶意代码变种检测方法研究》

# 反汇编
相似性：https://github.com/hija/MalwareDataScience/blob/master/chapter05/similarity.py

# 一个不错的博客博主
https://blog.csdn.net/weixin_47038938






