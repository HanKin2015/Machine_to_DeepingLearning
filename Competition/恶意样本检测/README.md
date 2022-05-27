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

### 2-2、进一步了解
[神操作：教你用Python识别恶意软件](https://mp.weixin.qq.com/s/kw5cOTIqisRuKYz9UNCbrQ)
[PE文件全解析](https://mp.weixin.qq.com/s/iNLTcmlGsw8jWtTtn3zXKQ)

静态分析是对程序文件的反汇编代码、图形图像、可打印字符串和其他磁盘资源进行分析，是一种不需要实际运行程序的逆向工程。虽然静态分析技术有欠缺之处，但是它可以帮助我们理解各种各样的恶意软件。

PE是Windows下的可执行文件的格式。这是微软基于UNIX平台的COFF(Common Object File Format，通用文件格式)制成的。微软原本的意思是提高程序的移植型。但是想法是好的，但是实际上只用于Windows系列的操作系统下。

PE文件是指32位的可执行文件,也称PE32。注意:64位的可执行文件称为PE+或PE32+，是PE32的一种扩展，不叫PE64。

PE格式最初的设计是用来进行下面的操作。
1）告诉Windows如何将程序加载到内存中
2）为运行程序提供在执行过程中可能使用的媒体（或资源）
3）提供安全数据，例如数字代码签名

PE文件格式包括一系列头（header），用来告诉操作系统如何将程序加载到内存中。它还包括一系列节（section）用来包含实际的程序数据。Windows将这些节加载到内存中，使其在内存中的偏移量与它们在磁盘上的显示位置相对应。

DOS头->PE头头->可选头->节头->.text节(程序代码)->.idata(导入库)->.rsrc节(字符串,图像...)->.reloc节(内存转换)

#### RV&RVA的转换
- VA是指进程虚拟内存的绝对虚拟地址.
- RVA(相对虚拟地址),是指从某个基准位置(Image Base)开始的相对地址。

转换关系如下:
```
RVA+Image Base = VA
```
PE头内部信息大多以RVA的形式存在。因为PE文件加载到进程虚拟内存的特定位置，但是，这个位置可能已加载了其他的PE文件（DLL）。因此必须重定向到其他的空白位置。若PE头信息使用的是VA,则无法正常访问。所以，使用RVA来定位信息，即使发生了重定位，只要基准地址没有发生变化，就可以正常访问到指定的信息。

#### 0.DOS头
微软在创建PE 文件格式时，人们正在广泛使用DOS 文件，所以微软为了考虑兼容性的问题，所以在PE 头的最前边还添加了一个 IMAGE_DOS_HEADER 结构体，用来扩展已有的DOS EXE。

Dos结构体的大小为40个字节，这里面有两个重要的成员变量
- e_magic:DOS签名，这个属性的值基本都是4D5A=>ASCII值"MZ"
- e_lfanew:指示NT头的偏移量(根据不同的文件拥有的值就是不一样的)

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


https://mp.weixin.qq.com/s/xkJAZ0dSc19f_qNDJXNx0w

## 3、机器学习中的应用
分析的较为详细，可参考：https://m.freebuf.com/articles/system/205444.html
小字版本，但是代码乱序：https://m.sohu.com/a/320909910_354899

https://github.com/evilsocket/ergo-pe-av/blob/master/encoder.py#L22

理论研究（文字）：https://cloud.tencent.com/developer/article/1909166

传统的恶意软件检测引擎依赖于签名，这些签名是恶意软件分析人员提供的，在确定恶意代码的同时确保非恶意样本中没有冲突。

这种方法存在一些问题，这通常很容易被绕过（也取决于签名的类型，单个比特的改变或者恶意代码中几个字节的变更都可能导致无法检测），同时不利于大规模扩展。研究人员需要进行逆向工程分析，发现与写出独特的签名可能要数小时。

训练集：占全部样本的70%，用于训练模型；
验证集：占全部样本的15%，在训练时用于校准模型；
测试集：占全部样本的15%，训练模型后比较模型效果。

PE包含几个头，描述其属性与各种关于寻址的细节，如PE在内存中加载的基址以及入口点的位置
PE包含几个段，包含数据（常量、全局变量等）、代码（该段被标记为可执行）等
PE包含导入的API与系统库的声明

属性	描述
pe.has_configuration	如果PE加载配置文件
pe.has_debug	如果PE启用调试选项
pe.has_exceptions	如果PE使用Exception
pe.has_exports	如果PE使用任意导出符号
pe.has_imports	如果PE导入任意符号
pe.has_nx	如果PE启用了NX位
pe.has_relocations	如果PE启用重定位
pe.has_resources	如果PE使用任意Resource
pe.has_rich_header	如果存在一个Rich头
pe.has_signature	如果PE拥有数字签名
pe.has_tls	如果PE使用TLS

然后的六十四个元素，代表PE入口点函数前的六十四个字节，每个元素都要进行归一化，这将会帮助模型检测那些具有独特入口点的可执行文件，这些入口点在相同家族的不同样本间仅有轻微的差别。
```
ep_bytes  =  [0]  *  64
try:
    ep_offset = pe.entrypoint - pe.optional_header.imagebase
    ep_bytes = [int(b) for b in raw[ep_offset:ep_offset+64]]
except Exception as e:
    log.warning("can't get entrypoint bytes from %s: %s", filepath, e)
# ...
# ...
def encode_entrypoint(ep):
    while len(ep) < 64: # pad
        ep += [0.0]
    return np.array(ep) / 255.0 # normalize
```
然后是二进制文件中的ASCII表中每个字节重复次数的直方图，是有关文件原始内容的基本统计信息。
```
# 参数raw中有文件的全部内容
def encode_histogram(raw):
    histo = np.bincount(np.frombuffer(raw, dtype=np.uint8), minlength=256)
    histo = histo / histo.sum() # normalize
    return  histo
```
下一个特征是导入表，因为PE文件使用的API是非常有用的信息。为了做到这一点，我手动选择了数据集中150个最常见的库，每个PE使用相对应库的列值加一，创建另一个一百五十位的直方图，通过导入API的总量进行标准化。


## 4、pefile模块---解析pe
源码下载地址：http://code.google.com/p/pefile/
命令安装：python setup.py install
对于whl文件：pip install windnd-1.0.7-py3-none-any.whl

由Ero Carerra编写和维护的Python模块pefile已经成为解析PE文件的一个行业标准的恶意软件分析库。


### 4-1、测试pefile
可以看出pefile对test.exe做了全面的解析从DOS_Header 到 OPTIONAL_HEADER 再到PE SECTIONS。每个结构都可以完全的取得。细心的朋友还可以发现，他甚至可以做对一个section header的hash运算，包括md5, sha1, sha-256, sha-512，对导入导出函数也做了列举。
当然大家会问，未必我们就直接一个print就行了，然后做字符串解析，匹配来获得我们想要的信息？那pefile肯定不至于那么愚昧，当然要提供更多的接口。比如得到entrypoint。

参考学习：https://blog.csdn.net/weixin_40596016/article/details/79304256

### 4-2、网上制作的pefile软件工具
https://blog.csdn.net/weixin_46625757/article/details/124088469


## 5、实战
[机器学习方法检测恶意文件](https://blog.csdn.net/zourzh123/article/details/81607330)

另外参考：
https://github.com/elastic/ember/blob/master/ember/features.py
https://github.com/evilsocket/ergo-pe-av/blob/master/encoder.py#L22
理论研究（文字）：https://cloud.tencent.com/developer/article/1909166

### 信息熵
信息熵（information entropy）是信息论的基本概念。描述信息源各可能事件发生的不确定性。20世纪40年代，香农（C.E.Shannon）借鉴了热力学的概念，把信息中排除了冗余后的平均信息量称为“信息熵”，并给出了计算信息熵的数学表达式。信息熵的提出解决了对信息的量化度量问题。
信息是个很抽象的概念。人们常常说信息很多，或者信息较少，但却很难说清楚信息到底有多少。比如一本五十万字的中文书到底有多少信息量。
信息论之父克劳德·艾尔伍德·香农第一次用数学语言阐明了概率与信息冗余度的关系。

https://baike.baidu.com/item/%E4%BF%A1%E6%81%AF%E7%86%B5/7302318?fr=aladdin

### 减函数
函数f(x)的定义域为I，如果对于定义域I内的某个区间D上的任意两个自变量的值x1,x2 ，当x1<x2时，都有f(x1)> f(x2)，那么就说f(x)在这个区间上是减函数，并称区间D为递减区间。减函数的图像从左往右是下降的，即函数值随自变量的增大而减小。判断一个函数是否为减函数可以通过定义法、图像法、直观法或利用该区间内导数值的正负来判断。

### 5-1、特征直方图
本质上PE文件也是二进制文件，可以当作一连串字节组成的文件。字节直方图又称为ByteHistogram，它的核心思想是，定义一个长度为256维的向量，每个向量依次为0x00，0x01一直到0xFF，分别代表 PE文件中0x00，0x01一直到0xFF对应的个数。例如经过统计，0x01有两个，0x03和0x05对应的各一个，假设直方图维度为8，所以对应的直方图为：[0,2,0,1,0,0,1,0,0]

实际使用时，单纯统计直方图非常容易过拟合，因为字节直方图对于PE文件的二进制特征过于依赖，PE文件增加一个无意义的0字节都会改变直方图；另外PE文件中不同字节的数量可能差别很大，数量占优势的字节可能会大大弱化其他字节对结果的影响，所以需要对直方图进行标准化处理。一种常见的处理方式是，增加一个维度的变量，用于统计PE文件的字节总数，同时原有直方图按照字节总数取 平均值。

### 5-2、字节熵直方图
若信源符号有n种取值：U1 ,U2 ,U3 ,…,Un ，对应概率为：P1 ,P2 ,P3 ,…,Pn ，且各种符号的出现彼此独立。
通常，log是以2为底。
PE文件同样可以使用字节的信息熵来当作特征。我们把PE文件当作一个字节组成的数组，如图8-5所示，在这个数组上以2048字节为窗口，以1024字节为步长计算熵。

### 5-3、文本特征
- 可读字符串个数
- 平均可读字符串长度
- 可读字符直方图，由于可读字符的个数为96个，所以我们定义一个长度为96的向量统计其直方图。
- 可读字符信息熵.
- C盘路径字符串个数，恶意程序通常对被感染系统的根目录有一定的文件操作行为，表现在可读字符串中，可能会包含硬编码的C盘路径，我们将这类字符串的个数作为一个维度。
注册表字符串个数，恶意程序通常对被感染系统的注册表有一定的文件操作行为，表现在可读字符串中，可能会包含硬编码的注册表值，我们将这类字符串的个数作为一个维度。注册表字符串一般包含"HKEY_"字串，例如：self._registry = re.compile(b'HKEY_')
- URL字符串个数，恶意程序通常从指定URL下载资源，最典型的就是下载者病毒，表现在可读字符串中，可能会包含硬编码的URL，我们将这类字符串的个数作为一个维度：self._urls = re.compile(b'https?://', re.IGNORECASE)
- MZ头的个数，例如：self._mz = re.compile(b'MZ')

### 5-4、解析PE结构特征
上面提到的字节直方图、字节熵直方图和文本特征直方图都可以把PE文件当作字节数组处理即可获得。但是有一些特征我们必须按照PE文件的格式进行解析后才能获得，比较典型的就是文件信息。我 们定义需要关注的文件信息包括以下几种：
- 是否包含debug信息。
- 导出函数的个数。
- 导入函数的个数。
- 是否包含资源文件。
- 是否包含信号量。
- 是否启用了重定向。
- 是否启用了TLS回调函数。
- 符号个数。

#### LIEF-用于检测可执行格式的库
我的环境直接可以引用lief库，无可以进行安装：pip install lief
















