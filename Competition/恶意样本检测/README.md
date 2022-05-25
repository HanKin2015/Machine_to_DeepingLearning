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

### 2-1、百度百科
PE文件的全称是Portable Executable，意为可移植的可执行的文件，常见的EXE、DLL、OCX、SYS、COM都是PE文件，PE文件是微软Windows操作系统上的程序文件（可能是间接被执行，如DLL）

一个操作系统的可执行文件格式在很多方面是这个系统的一面镜子。

### 2-2、其他
[神操作：教你用Python识别恶意软件](https://zhuanlan.zhihu.com/p/174260322)

静态分析是对程序文件的反汇编代码、图形图像、可打印字符串和其他磁盘资源进行分析，是一种不需要实际运行程序的逆向工程。虽然静态分析技术有欠缺之处，但是它可以帮助我们理解各种各样的恶意软件。

PE格式最初的设计是用来进行下面的操作。
1）告诉Windows如何将程序加载到内存中
2）为运行程序提供在执行过程中可能使用的媒体（或资源）
3）提供安全数据，例如数字代码签名

PE文件格式包括一系列头（header），用来告诉操作系统如何将程序加载到内存中。它还包括一系列节（section）用来包含实际的程序数据。Windows将这些节加载到内存中，使其在内存中的偏移量与它们在磁盘上的显示位置相对应。

DOS头->PE头头->可选头->节头->.text节(程序代码)->.idata(导入库)->.rsrc节(字符串,图像...)->.reloc节(内存转换)

#### 1. PE头
在DOS头的上面是PE头，它定义了程序的一般属性，如二进制代码、图像、压缩数据和其他程序属性。它还告诉我们程序是否是针对32位或64位系统而设计的。

PE头为恶意软件分析师提供了基本但有用的情景信息。例如，头里包括了时间戳字段，这个字段可以给出恶意软件作者编译文件的时间。通常恶意软件作者会使用伪造的值替换这个字段，但是有时恶意软件作者会忘记替换，就会发生这种情况。

#### 2. 可选头
可选头实际上在今天的PE可执行程序中无处不在，恰恰与其名称的含义相反。它定义了PE文件中程序入口点的位置，该位置指的是程序加载后运行的第一个指令。

它还定义了Windows在加载PE文件、Windows子系统、目标程序（例如Windows GUI或Windows命令行）时加载到内存中的数据大小，以及有关该程序其他的高级详细信息。由于程序的入口点告诉了逆向工程师该从哪里开始进行逆向工程，这个头信息对逆向工程师来说是非常宝贵的。

#### 3. 节头
节（section）头描述了PE文件中包含的数据节。PE文件中的一个节是一块数据，它们在操作系统加载程序时将被映射到内存中，或者包含有关如何将程序加载到内存中的指令。
换句话说，一个节是磁盘上的字节序列，它要么成为内存中一串连续字节的字符串，要么告知操作系统关于加载过程的某些方面。
节头还告诉Windows应该授予节哪些权限，比如程序在执行时，是否应该可读、可写或可执行。例如，包含x86代码的.text节通常被标记为可读和可执行的，但是不可写的，以防止程序代码在执行过程中意外修改自身。
如.text和.rsrc。执行PE文件时，它们会被映射到内存中。其他如.reloc节的特殊节不会被映射到内存中。

## 3、pefile模块---解析pe
源码下载地址：http://code.google.com/p/pefile/
命令安装：python setup.py install
对于whl文件：pip install windnd-1.0.7-py3-none-any.whl

由Ero Carerra编写和维护的Python模块pefile已经成为解析PE文件的一个行业标准的恶意软件分析库。






















## 4、测试pefile
可以看出pefile对test.exe做了全面的解析从DOS_Header 到 OPTIONAL_HEADER 再到PE SECTIONS。每个结构都可以完全的取得。细心的朋友还可以发现，他甚至可以做对一个section header的hash运算，包括md5, sha1, sha-256, sha-512，对导入导出函数也做了列举。
当然大家会问，未必我们就直接一个print就行了，然后做字符串解析，匹配来获得我们想要的信息？那pefile肯定不至于那么愚昧，当然要提供更多的接口。比如得到entrypoint。

参考学习：https://blog.csdn.net/weixin_40596016/article/details/79304256

## 5、网上制作的pefile软件工具
https://blog.csdn.net/weixin_46625757/article/details/124088469

























