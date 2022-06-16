#!/bin/bash
#
# 文 件 名: get_custom_strings.sh
# 文件描述: 操作摄像头工具合集
# 作    者: HeJian
# 创建日期: 2022.06.10
# 修改日期：2022.06.15
# 
# Copyright (c) 2022 HeJian. All rights reserved.
#

TRAIN_WHITE_PATH='./AIFirst_data/train/white/'                 # 训练集白样本路径
TRAIN_BLACK_PATH='./AIFirst_data/train/black/'                 # 训练集黑样本路径
TEST_PATH='./AIFirst_data/test/'                               # 测试集样本路径
DATA_PATH='./data/'                                            # 数据路径
TRAIN_WHITE_STRINGS_PATH=${DATA_PATH}'train_white_strings.csv' # 训练集白样本数据集路径
TRAIN_BLACK_STRINGS_PATH=${DATA_PATH}'train_black_strings.csv' # 训练集黑样本数据集路径
TEST_STRINGS_PATH=${DATA_PATH}'test_strings.csv'               # 测试集样本数据集路径

# 调试路径
IN='/media/sangfor/vdb/study/test/'
OUT='./result.csv'

STRING_FEATURES=(
    "C:\\\Windows\\\System32"
    "http://"
    "https://"
    "download"
    "HKEY_"
    "wget"
    "curl"
    "%SystemRoot%\\\System32"
    "%windir%\\\system32"
    "SystemRoot\\\System32"
    "ftp"
)

function get_strings()
{
    data_path=$1
    strings_path=$2
    echo ${data_path} ${strings_path}

    for file in `ls ${data_path}`; do
        echo -n ${file} >> ${strings_path}
        for string_feature in ${STRING_FEATURES[*]}; do
            #echo ${string_feature}
            if strings ${data_path}${file} | grep -qi ${string_feature}; then
                echo -n ',1' >> ${strings_path}
            else
                echo -n ',0' >> ${strings_path}
            fi
        done
        echo '' >> ${strings_path}
    done
}

#get_strings ${TRAIN_WHITE_PATH} ${TRAIN_WHITE_STRINGS_PATH}
#get_strings ${TRAIN_BLACK_PATH} ${TRAIN_BLACK_STRINGS_PATH}
#get_strings ${TEST_PATH} ${TEST_STRINGS_PATH}
get_strings ${IN} ${OUT}