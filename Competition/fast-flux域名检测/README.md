# fast-flux域名检测

## 1、四步走
获取数据
特征工程
训练模型
预测结果

pip install xgboost
pip install lightgbm
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install catboost
pip install graphviz
https://pypi.tuna.tsinghua.edu.cn/simple/graphviz/
https://pypi.tuna.tsinghua.edu.cn/simple/catboost/
https://pypi.tuna.tsinghua.edu.cn/simple/lightgbm/
https://pypi.tuna.tsinghua.edu.cn/simple/xgboost/

nlpia这个库需要安装很多依赖，建议在线安装，只是例子中使用到。

## 2、Bailiwick
居然百度和必应几乎搜索不出来相关资料，这。。。。。。
https://zhuanlan.zhihu.com/p/92899876/

官方资料解释：https://www.farsightsecurity.com/blog/txt-record/what-is-a-bailiwick-20170321/

## 3、浏览互联网电话簿：被动 DNS 简介
Navigating the Internet’s Phone Book: An Introduction to Passive DNS

“被动 DNS”或“被动 DNS 复制”是 Florian Weimer 于 2004 年发明的一种技术，用于将全球域名系统中可用数据的部分视图机会性地重建到中央数据库中，以便对其进行索引和查询。 被动 DNS 允许威胁追踪者将各个入侵指标 (IOC) 与 Internet 上的几乎每个活动域和 IP 地址连接起来。 了解被动 DNS 数据如何帮助您将网络活动映射到攻击者基础设施。

常规 DNS：“常规”域名系统 (“DNS”) 通常被称为 Internet 的“电话簿”。 它将一个符号域名（如 Farsight Security、网络安全情报解决方案）解析为计算机和网络实际需要的 IP 地址（如 104.244.14.108）。 

多个站点通常会共享一个公共网络服务器、一个公共邮件服务器或公共名称服务器。 这意味着，如果我们能找到一个不良站点，我们通常可以使用最初的“潜在客户”来查找其他不良站点。

不能要求“常规 DNS”查看具有特定网络范围内 IP 地址的所有域名——常规 DNS 根本不知道如何回答此类问题。
也不能要求查看特定“二级域”（例如“oregon.gov”）下存在的所有完全限定域名（“主机名”）的列表。
幸运的是，如果您使用的是被动 DNS，您可以回答这类问题。

https://zhuanlan.zhihu.com/p/451551110 翻译后还是无法理解

另参考：https://bbs.csdn.net/topics/607189232
PDNS（被动DNS），也称为Passive DNS。被动DNS是利用与DNS查询的方式相反，被动DNS属于反向获取或查询DNS数据信息。
那么主动DNS，就是我们日常请求一个域名的常规流程。

被动DNS优点：数据量大，收集范围广。
主动DNS优点：精准发现潜在的域名或子域名。
被动DNS缺点：在互联网安全世界中只依靠被动DNS获取数据是远远不够的，时效性不及时，发现的数据有限，数据过于庞大，重复数据太多，无效数据太多（比如攻击爆破等等）。
主动DNS缺点：不能大量数据收集，收集范围狭小。

最好的文章：https://developer.aliyun.com/article/764940

发现有一些问题，光靠DNS是很难解决的，例如：
- 一个域名过去曾指向到何处？
- 特定网络范围内的IP地址对应的所有域名有哪些？
- 对于一个给定的名称服务器，它托管了哪些域名？
- 有哪些域名指向给定的IP网络？
- 一个域名下存在哪些子域名？
这些DNS很难解决的问题，利用被动DNS（Passive DNS）技术就可以迎刃而解。
https://zhuanlan.zhihu.com/p/29162387

## 4、子域名查询、DNS记录查询
https://zhuanlan.zhihu.com/p/402676726

## 5、最常见的三种DNS记录
https://zhuanlan.zhihu.com/p/92899876/

DNS记录是用于解析的数据，最常见的3种记录为：NS记录、A记录、CNAME记录。

### 5-1、第一种：NS记录
如果DNS给你回应一条NS记录，就是告诉你，这个家伙是某个域的权威DNS，有事你去问它。

### 5-2、第二种：A记录
A记录就是最经典的域名和IP的对应，在http://ns1.baidu.com里面，记录着百度公司各产品的域名和IP的对应关系，每一个这样的记录，就是一个A记录。

### 5-3、第三种：CNAME记录
这种记录比较有趣，你问DNS一个域名，它回CNAME记录，意思是说，你要解析的这个域名，还有另一个域名(也就是别名)，你去解析那个好了。

```
baidu.com.      86400   IN  NS  ns3.baidu.com.
```
注意，域名后面会比我们平时见到的多一个“.”，这就代表了根，严格地说，所有域名后面都应该有这一个“.”的，比如完整的http://www.baidu.com域名应该是www.baidu.com.，也可以把所有域名都看作有一个.root的后缀，比如www.baidu.com.root，但由于每个域名都有这个后缀，所以干脆就都省略了。（本文后面也都会省略这个“.”）

其中的86400就是TTL，是以秒为单位的，86400也就是24小时即1天。

其中的“IN”指的是互联网类型的域名，通常都是如此，还有一种类型是“CHAOS”，几乎没用。

## 6、DNS报文内容
DNS报文结构：
```
|QID|问题区|应答区|权威区|附加区|
```
查询报文（以下简称查询包）只是填写QID和问题区（也即要问什么），后面几个区不用填写内容。

响应报文（以下简称响应包）会填写QID（和对应查询包中的QID一致）、问题区（和对应查询包中的问题一致），还会填写后面几个区：应答区、权威区（也即填写权威DNS的区域，）、附加区。

这几个区的主要作用是：
问题区（Question Section）：这里填写所要查询的问题（可以是一个或多个），主要填查询内容和查询类型，比如想要查http://www.baidu.com的A记录，就填http://www.baidu.com和A。

应答区（Answer Section）：这里给出查询问题的答案，对于A记录类型的问题，这里会给出一个或多个A记录，也可能给出一个或多个CNAME记录。当然，如果问题问的是NS记录，这里就会应答NS记录。

权威区（Authority Section）：这里给出一个或多个NS记录，其实就是那些referral，也即告诉LDNS问谁会更接近答案一些，比如com的DNS会在这里给出http://baidu.com的权威DNS（可以给出多个）。注意，在必要情况下，权威DNS在应答区给出回答的同时，还会在权威区给出NS记录。

附加区（Additional Section）：这里存放附加的一些记录。比如在给出权威NS记录的同时，会把它的A记录放在这里（在NS记录中是不会有IP的），这样做的好处是可以解决鸡生蛋蛋生鸡的问题，比如你查http://www.baidu.com，权威告诉你去找http://ns.baidu.com，如果没有这个附加的A记录（又称胶水记录，glue record），你就得去问http://ns.baidu.com的A记录，然后权威又告诉你去找http://ns.baidu.com，这就进入一个无解的循环之中。

## 7、前面没有提及的bailiwick检查
这个攻击之所以出人意料，在于他能通过bailiwick检查，DNS投毒（或者说缓存污染）并不是新鲜事，bailiwick检查被设计出来专门防范缓存污染，主要原则是：如果检查发现附加区中的记录和问题区中的问题不在同一个域管辖之下，就会格外谨慎而不会采信（更不会记入缓存）此记录，这可以防范恶意权威DNS发出虚假的记录以污染缓存。

## 8、dnsdb.py
https://github.com/dnsdb/dnsdb-query
只能在linux上面运行。




https://github.com/asgoel/Fast-Flux-Detect
https://github.com/staaldraad/fastfluxanalysis
https://mp.weixin.qq.com/s/_VrzoPMhc8PIyFnbN3YaNw


Fast-Flux技术

在正常的DNS服务器中，用户对同一个域名做DNS查询，在较长的一段时间内，无论查询多少次返回的结果基本上是不会改变的，而Fast-Flux技术为一个域名配置了多个IP地址，通过动态变化IP地址完成和域名的映射，这种情况下用户在每次访问某个域名时实际上访问不是同一个主机。攻击者将这些感染的主机仅仅作为代理将需求转发给实际的控制者，完成控制者和被控机器的通信。

Fast-Flux域名检测



■ Fast-Flux特征分析

基于Fast-Flux技术的僵尸网络，有以下几个特征：

1）返回的IP数量

Fast-Flux网络返回的IP地址是不断变化的，在一段时间内查询得到的IP地址数量会持续增加，而合法网站的域名查询返回的IP数一般比较稳定。

2）IP所属国家特征

由于受感染主机的不同区域分布，基于Fast-Flux技术的僵尸网络的IP地址分布在不同的国家（对应不同的时区，混乱度较大），而正常的域名IP一般都在一个国家或少数几个国家中。



■ Fast-Flux检测方法

针对上述特征，首先考虑从输入的数据中统计每个域名响应的IP地址序列和对应的时区数据，并将每一个域名的统计结果和上一个周期进行累加更新，最后使用AI算法对数据进行异常的检测，并上报异常结果。



35  91.2530
25  91.4593
45  86.4233
15  91.1931

25  85.1017







