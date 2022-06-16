# 恶意样本检测代码分类
- 比赛说明 公司内部
- 数据下载 无

# 代码说明
|代码文件名          | 文件描述                                               |
|:-------------------| :------------------------------------------------------|
AIFirst_data.zip     | 随意找的几个exe文件
get_data.py          | 获取训练集和测试集的pe文件基本结构特征（使用pefile库）
file.py              | 获取训练集和测试集文件的文件信息
get_gray_images.py   | 获取灰度图像
get_image_matrix.py  | 获取图像矩阵数据集
get_opcode_n_gram.py | 获取执行文件的操作指令码的n-gram特征数据集
gray_image_n_gram.py | 初稿调试获取图像矩阵、灰度图像和操作指令码的n-gram特征
features.py          | 特征工程
predict.py           | 加载训练模型进行预测
main.py              | 主函数
pefile_info.txt      | pe文件使用pefile库解析出来的pe信息
strings.py           | 使用strings命令获取全部字符串特征数据集
get_custom_strings.sh| 使用strings命令获取指定字符串特征数据集

# 运行说明
将完整的训练数据集解压，运行main.py，耐心等待即可看到结果














