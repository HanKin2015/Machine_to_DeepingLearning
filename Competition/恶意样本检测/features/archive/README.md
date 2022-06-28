# 文件说明

## 1、公共文件
log.py：日志文件
common.py：公共文件库，包含python库引用、公共变量等

## 2、获取特征
get_data.py：通过pefile库获取pe文件信息
get_custom_string.sh：使用strings命令获取指定字符串
get_opcode_n_gram.py：获取二进制文件的n-gram特征(使用pefile库反汇编获取操作指令码)

## 3、特征工程
opcode_n_gram_features.py：操作指令码n-gram特征，由于实验环境内存太小，无法一次性获取完整，需要拼接
features.py：特征工程，将pe文件信息特征和字符串特征结合（由于n-gram特征是后面增加的，未在这个文件中结合）

## 4、训练
combine.py：将pe文件信息特征和字符串特征，以及n-gram特征结合，使用LGBMClassifier进行训练

## 5、预测判断
predict.py：加载模型对测试集进行预测判断输出结果

## 6、其他文件
main.py： 主程序入口
model：模型文件夹
result.csv：最终提交文件

总体思路：
使用pe文件信息特征+汇编语言操作指令3-gram特征+strings命令提取的字符串特征，使用LGBMClassifier进行训练。
